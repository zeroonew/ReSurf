#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import os
import torch
import random
from random import randint
from utils.loss_utils import l1_loss, ssim, loss_depth_smoothness
from utils.feat_utils import compute_reference_view_feature_penalty, FeatExt
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.vis_utils import apply_depth_colormap, save_points, colormap, apply_normal_colormap, pca_feature
from utils.point_utils import depths_to_points, depth_to_normal, inpaint_and_resample
import torchvision
from utils.graphics_utils import patch_offsets, patch_warp
from utils.graphics_utils import normal_from_depth_image
import numpy as np
from torch import nn
import torch.nn.functional as F
import cv2
from utils.foundationstereo_utils import Foundation, disp2depth, left_right_check
from utils.confidence_utils import (
    left_right_confidence,
    multi_baseline_fusion,
    normal_guided_depth_propagation,
)
from utils.floater_utils import (
    get_scheduler_lambdas,
    SpatialConstraint,
    compute_pixel_outlier_scores,
    attribute_scores_to_gaussians,
    OutlierTracker,
    identify_prune_mask,
    collect_reliable_surface_points,
    prepare_new_gaussian_tensors,
)
from sklearn.decomposition import PCA
import math
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

class CNN_decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1).cuda()


    def forward(self, x):    
        x = self.conv(x)
        return x

def pca_func(feature, n=3):
    device = feature.device
    B, C, H, W = feature.shape
    assert B == 1, B
    pca = PCA(n_components=n)
    feature = feature.permute(0, 2, 3, 1).contiguous().reshape(H*W, C).detach().cpu().numpy()
    feature_map = pca.fit_transform(feature)  # (H*W, 3)
    feature_map = torch.from_numpy(feature_map).to(device)
    feature_map = feature_map.reshape(H, W, n).permute(2, 0, 1).unsqueeze(0)
    return feature_map

def ranking_loss(error, penalize_ratio=0.7, extra_weights=None , type='mean'):
    error, indices = torch.sort(error)
    # only sum relatively small errors
    s_error = torch.index_select(error, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
    if extra_weights is not None:
        weights = torch.index_select(extra_weights, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
        s_error = s_error * weights

    if type == 'mean':
        return torch.mean(s_error)
    elif type == 'sum':
        return torch.sum(s_error)

def render_normal_func(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def apply_reference_feature_penalty(loss, scene, name2idx, viewpoint_cam, surf_depth, mask, dataset, opt, iteration):
    if iteration <= opt.featloss_from_iter:
        return loss

    paired_train_views = scene.getTrainCamerasSource(name2idx[viewpoint_cam.image_name]).copy()
    train_feat_loss = compute_reference_view_feature_penalty(
        depths_to_points(viewpoint_cam, surf_depth),
        viewpoint_cam,
        paired_train_views,
        mask,
        resolution=dataset.resolution,
    )
    return loss + opt.lambda_feat * train_feat_loss


@torch.no_grad()
def update_stereo_for_camera(t_cam, scene, name2idx, gaussians, pipe, background,
                             foundation, opt, iteration, conf_collector):
    """ConfSurf stereo update for one training camera.

    Differences vs SparseSurf:
      A2 - run stereo matching on multiple baselines and fuse the depths
           with confidence-weighted averaging (cross-baseline agreement
           discounts inconsistent pixels);
      A1 - store a continuous confidence map (exp of the LR residual)
           instead of only a binary left-right-check mask;
      B  - propagate high-confidence depth into low-confidence regions
           along the surface tangent, guided by rendered normals.
    """
    current_idx = name2idx[t_cam.image_name]
    closest_cam = scene.getTrainCamerasSource(current_idx)[0]
    pos_current = t_cam.c2w[:3, 3]
    right_current = t_cam.c2w[:3, 0]
    pos_neighbor = closest_cam.c2w[:3, 3]

    direction = pos_neighbor - pos_current
    dot = torch.dot(right_current, direction)
    cam_is_left = dot > 0  # t_cam acts as the left camera of the pair

    primary_baseline = scene.cameras_extent * opt.stereo_baseline_percent
    if opt.adaptive_baseline:
        # Two baselines by default (primary + large): the cross-baseline
        # agreement term in multi_baseline_fusion discounts inconsistent
        # pixels, giving a per-pixel uncertainty estimate at ~2x stereo cost.
        # Add opt.stereo_baseline_percent_small to the set for a 3rd baseline.
        baselines = sorted({
            primary_baseline,
            scene.cameras_extent * opt.stereo_baseline_percent_large,
        })
    else:
        baselines = [primary_baseline]

    depths, confs = [], []
    for bi, baseline in enumerate(baselines):
        if cam_is_left:
            l_image = t_cam.original_image.cuda()
            r_cam = t_cam.get_stereo_pair(baseline)
            r_render = render(r_cam, gaussians, pipe, background)
            r_image = r_render["render"]
            r_cam.original_image = r_image.clone().detach().to(r_cam.data_device)
            L2R_disparity = foundation.predict_disparity(l_image.unsqueeze(0), r_image.unsqueeze(0)).squeeze(0)
            R2L_disparity = foundation.predict_disparity(l_image.unsqueeze(0), r_image.unsqueeze(0), flip=True).squeeze(0)
            stereo_depth = disp2depth(L2R_disparity, baseline, t_cam)
            partner_depth = disp2depth(R2L_disparity, baseline, t_cam)
            conf = left_right_confidence(L2R_disparity, R2L_disparity,
                                         tau=opt.conf_tau, hard_threshold=opt.conf_hard_threshold)
            stereo_depth_mask = left_right_check(L2R_disparity, R2L_disparity)
            partner_cam, partner_image = r_cam, r_image
        else:
            r_image = t_cam.original_image.cuda()
            l_cam = t_cam.get_stereo_pair_(baseline)
            l_render = render(l_cam, gaussians, pipe, background)
            l_image = l_render["render"]
            l_cam.original_image = l_image.clone().detach().to(l_cam.data_device)
            L2R_disparity = foundation.predict_disparity(l_image.unsqueeze(0), r_image.unsqueeze(0)).squeeze(0)
            R2L_disparity = foundation.predict_disparity(l_image.unsqueeze(0), r_image.unsqueeze(0), flip=True).squeeze(0)
            stereo_depth = disp2depth(R2L_disparity, baseline, t_cam)
            partner_depth = disp2depth(L2R_disparity, baseline, t_cam)
            conf = left_right_confidence(R2L_disparity, L2R_disparity,
                                         tau=opt.conf_tau, hard_threshold=opt.conf_hard_threshold)
            stereo_depth_mask = left_right_check(R2L_disparity, L2R_disparity)
            partner_cam, partner_image = l_cam, l_image

        # keep the bookkeeping of the primary baseline (matches SparseSurf)
        if baseline == primary_baseline:
            t_cam.set_last_right_depth(partner_depth)
            partner_cam.set_last_stereo_depth(partner_depth, iteration)
            t_cam.set_last_stereo_depth_mask(stereo_depth_mask)
            t_cam.set_last_right_transforms(partner_cam.world_view_transform[None])
            t_cam.set_last_right_image(partner_image.clone().detach())
            t_cam.right_cam = partner_cam

        depths.append(stereo_depth)
        confs.append(conf)

    if len(depths) > 1:
        fused_depth, fused_conf = multi_baseline_fusion(depths, confs, tau_rel=opt.baseline_rel_tau)
    else:
        fused_depth, fused_conf = depths[0], confs[0]

    # A3 quality tracking: raw matching confidence before propagation
    valid = fused_depth.squeeze(0) > 0
    if conf_collector is not None and valid.any():
        conf_collector.append(float(fused_conf.squeeze(0)[valid].mean()))

    # B: normal-guided propagation into low-confidence / occluded regions
    if opt.enable_depth_propagation and iteration >= opt.prop_from_iter:
        t_render = render(t_cam, gaussians, pipe, background)
        rend_normal = t_render["rendered_normal"].detach()  # (3, H, W), camera space
        prop_depth, prop_conf = normal_guided_depth_propagation(
            fused_depth.squeeze(0), fused_conf.squeeze(0), rend_normal,
            t_cam.Fx, t_cam.Fy, t_cam.Cx, t_cam.Cy,
            n_iters=opt.prop_n_iters, conf_high=opt.prop_conf_high, gamma=opt.prop_gamma)
        fused_depth = prop_depth.unsqueeze(0)
        fused_conf = prop_conf.unsqueeze(0)

    t_cam.set_last_stereo_depth(fused_depth, iteration)
    t_cam.set_last_stereo_confidence(fused_conf)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    foundation = Foundation(dataset.foundation_stereo_ckpt)
    feat_ext = FeatExt().cuda()
    feat_ext.eval()
    for p in feat_ext.parameters():
        p.requires_grad = False
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.feat_dim)
    scene = Scene(dataset, gaussians)
    temp_trainCam = scene.getTrainCameras().copy()
    name2idx = {}
    for idx, view in enumerate(temp_trainCam):
        name = view.image_name
        name2idx.update({name: int(idx)})
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # ---- SparseSurf floater pipeline init ----
    floater_enabled = bool(getattr(opt, "enable_floater", 0))
    spatial_constraint = None
    outlier_tracker = None
    if floater_enabled:
        init_pts = gaussians.get_xyz.detach().cpu().numpy()
        spatial_constraint = SpatialConstraint(init_pts, scene.cameras_extent)
        outlier_tracker = OutlierTracker(
            gaussians.get_xyz.shape[0], beta=getattr(opt, "ema_beta", 0.95))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    virtualCameras = scene.getVirtualCameras().copy()
    # allCameras = trainCameras + testCameras
    allCameras = trainCameras

    # feature map CNN decoder
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    gt_feat_map = viewpoint_cam.feat[0].cuda()
    feature_out_dim = gt_feat_map.shape[0]
    feature_in_dim = dataset.feat_dim
    cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
    cnn_decoder_optimizer = torch.optim.Adam(cnn_decoder.parameters(), lr=0.0001)


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    viewpoint_stack2 = None
    unseen_viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    stereo_quality_ema = 0.0  # ConfSurf A3: EMA of mean stereo confidence
    # ConfSurf A3: under the confidence-aware schedule we start producing stereo
    # priors earlier; the quality gate (not a fixed iteration) controls how much
    # they influence the loss.
    stereo_start_iter = opt.stereo_min_warmup_iter if opt.confidence_aware_schedule else opt.stereofrom_iterations

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # update stereo depth  (ConfSurf: multi-baseline + confidence + propagation)
        with torch.no_grad():
            if iteration == stereo_start_iter or (iteration > stereo_start_iter and iteration % opt.stereosetup_interval == 0 and iteration < 7000) or (iteration > 7000 and iteration % 1000 == 0):
                conf_collector = []
                for t_cam in scene.getTrainCameras():
                    update_stereo_for_camera(t_cam, scene, name2idx, gaussians, pipe, background,
                                             foundation, opt, iteration, conf_collector)
                # A3: track stereo quality (mean confidence) for adaptive scheduling
                if len(conf_collector) > 0:
                    mean_conf = sum(conf_collector) / len(conf_collector)
                    if stereo_quality_ema == 0.0:
                        stereo_quality_ema = mean_conf
                    else:
                        stereo_quality_ema = opt.stereo_quality_ema * stereo_quality_ema + (1 - opt.stereo_quality_ema) * mean_conf
                    for t_cam in scene.getTrainCameras():
                        t_cam.set_last_stereo_quality(stereo_quality_ema)
                torch.cuda.empty_cache()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        if not unseen_viewpoint_stack:
            unseen_viewpoint_stack = scene.getVirtualCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_vircam = unseen_viewpoint_stack.pop(randint(0, len(unseen_viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        # ---- Stage 6: color down-weighting (per-pixel L1 weighted by outlier) ----
        lambda_space, lambda_decay, gamma = get_scheduler_lambdas(iteration, opt)
        if floater_enabled and outlier_tracker is not None and iteration >= opt.detect_from_iter:
            with torch.no_grad():
                _, H, W = image.shape
                pixel_outlier = outlier_tracker.project_to_pixels(
                    viewpoint_cam, gaussians.get_xyz.detach(), H, W)
                color_weights = torch.clamp(1.0 - gamma * pixel_outlier, min=0.1).detach()
            per_pix_l1 = (image - gt_image).abs().mean(dim=0)  # (H, W)
            Ll1 = (color_weights * per_pix_l1).mean()
        else:
            Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # ---- Stage 1: spatial constraint L_space ----
        if floater_enabled and spatial_constraint is not None:
            loss = loss + lambda_space * spatial_constraint.loss(
                gaussians.get_xyz, gaussians.get_opacity)
       
        # regularization
        lambda_normal = opt.lambda_normal if iteration > opt.lambda_normal_from_iter else 0.0
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.scale_loss_weight * min_scale_loss.mean()

        rend_normal  = render_pkg['rendered_normal']
        surf_normal = render_pkg['depth_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        rend_alpha = render_pkg["rendered_alpha"]

        surf_depth = render_pkg["plane_depth"]
        feature_map = render_pkg["feature_map"]

        _, gt_image_gray = viewpoint_cam.get_image()


        mask = (surf_depth.view(-1) > 0)

        # feature-distillation loss
        gt_feature_map = viewpoint_cam.feat[0].cuda()
        if iteration > opt.splat_feature_loss_iter:
            feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) 
            if dataset.feat_dim == 32:
                feature_map_d = feature_map
            else:
                feature_map_d = cnn_decoder(feature_map)
            feature_error = (1 - F.cosine_similarity(gt_feature_map, feature_map_d, dim=0)).mean()
            loss += feature_error * opt.lambda_splat_feat     

        # Stereo Loss (ConfSurf: continuous confidence weighting + quality-gated schedule)
        if opt.confidence_aware_schedule:
            stereo_active = (iteration > opt.stereo_min_warmup_iter) and (viewpoint_cam.last_stereo_depth is not None)
        else:
            stereo_active = (iteration > opt.stereofrom_iterations) and (viewpoint_cam.last_stereo_depth is not None)

        if stereo_active:
            raft_depth = viewpoint_cam.get_last_stereo_depth()
            prior_valid_mask = viewpoint_cam.get_last_stereo_depth_mask()

            use_conf = opt.use_confidence_weighting and (viewpoint_cam.last_stereo_confidence is not None)
            if use_conf:
                # A3: gate the whole stereo supervision by the tracked stereo quality
                if opt.confidence_aware_schedule:
                    quality_gate = float(min(max(stereo_quality_ema / max(opt.stereo_quality_target, 1e-6), 0.0), 1.0))
                else:
                    quality_gate = 1.0
                weight_map = (viewpoint_cam.get_last_stereo_confidence() * quality_gate).detach()  # (1,H,W)
                weight_flat = weight_map.squeeze(0)
                w_sum = weight_map.sum().clamp(min=1e-8)

                # confidence-weighted depth supervision (replaces masked mean)
                stereo_depth_loss = (torch.abs(surf_depth - raft_depth) * weight_map).sum() / w_sum
                loss += opt.lambda_stereo_depth_sup * stereo_depth_loss

                stereo_depth_normal = render_normal_func(viewpoint_cam, raft_depth.squeeze())
                stereo_depth_normal = stereo_depth_normal.cuda().detach()
                stereo_depth_normal = rend_alpha.clone().detach() * stereo_depth_normal
                normal_prior_error = (1 - F.cosine_similarity(stereo_depth_normal, rend_normal, dim=0)) + \
                                        (1 - F.cosine_similarity(stereo_depth_normal, surf_normal, dim=0))
                normal_prior_loss = opt.lambda_normal_prior * (normal_prior_error * weight_flat).sum() / weight_flat.sum().clamp(min=1e-8)
                loss += normal_prior_loss
            else:
                # fallback: original binary-mask behaviour
                stereo_depth_loss = torch.abs((surf_depth - raft_depth))[prior_valid_mask.bool()].mean()
                loss += opt.lambda_stereo_depth_sup * stereo_depth_loss
                stereo_depth_normal = render_normal_func(viewpoint_cam, raft_depth.squeeze())
                stereo_depth_normal = stereo_depth_normal.cuda().detach()
                stereo_depth_normal = rend_alpha.clone().detach() * stereo_depth_normal
                normal_prior_error = (1 - F.cosine_similarity(stereo_depth_normal, rend_normal, dim=0)) + \
                                        (1 - F.cosine_similarity(stereo_depth_normal, surf_normal, dim=0))
                normal_prior_error = ranking_loss(normal_prior_error[prior_valid_mask.squeeze(0).bool()], 
                                                    penalize_ratio = 1.0, type='mean')
                normal_prior_loss = opt.lambda_normal_prior * normal_prior_error
                loss += normal_prior_loss
            # smooth loss
            normal_smooth_loss = loss_depth_smoothness(rend_normal, stereo_depth_normal) + loss_depth_smoothness(surf_normal, stereo_depth_normal)
            loss += normal_smooth_loss * opt.lambda_normal_smooth

        # Pseudo-view Feature Loss
        if iteration > opt.pesudo_featpgsr_iter:
            if viewpoint_vircam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.pesudo_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ncc_scale = 1.0
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W, device="cuda"), torch.arange(H, device="cuda"), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float()

                nearest_render_pkg = render(viewpoint_vircam, gaussians, pipe, background)

                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_viewpoint_vircam = pts @ viewpoint_vircam.world_view_transform[:3,:3] + viewpoint_vircam.world_view_transform[3,:3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(viewpoint_vircam, nearest_render_pkg['plane_depth'], pts_in_viewpoint_vircam)
                
                pts_in_viewpoint_vircam = pts_in_viewpoint_vircam / (pts_in_viewpoint_vircam[:,2:3])
                pts_in_viewpoint_vircam = pts_in_viewpoint_vircam * map_z.squeeze()[...,None]
                R = torch.tensor(viewpoint_vircam.R).float().cuda()
                T = torch.tensor(viewpoint_vircam.T).float().cuda()
                pts_ = (pts_in_viewpoint_vircam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                C, _, _ = gt_feature_map.shape                
                d_mask = d_mask & (pixel_noise < pixel_noise_th)
                weights = (1.0 / torch.exp(pixel_noise)).detach()
                weights[~d_mask] = 0
                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    c_2 = sample_num 
                    with torch.no_grad():
                        ## sample mask
                        d_mask = d_mask.reshape(-1)
                        valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                        if d_mask.sum() > sample_num:
                            index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                            valid_indices = valid_indices[index]
                        if d_mask.sum() < sample_num:
                            c_2 = d_mask.sum()
                        
                        weights = weights.reshape(-1)[valid_indices]
                        ## sample ref frame patch
                        pixels = pixels.reshape(-1,2)[valid_indices]
                        
                        offsets = patch_offsets(patch_size, pixels.device)
                        ori_pixels_patch = pixels.reshape(-1, 1, 2) / ncc_scale + offsets.float()
                        # _, H, W = gt_feature_map.shape
                        pixels_patch = ori_pixels_patch.clone()
                        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                        ref_gray_val = F.grid_sample(gt_feature_map.unsqueeze(0), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                        ref_gray_val = ref_gray_val.reshape(C, c_2, total_patch_size)

                        ref_to_neareast_r = viewpoint_vircam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                        ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + viewpoint_vircam.world_view_transform[3,:3]
                    
                    ## compute Homography
                    ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                    ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                    ref_local_d = render_pkg['rendered_distance'].squeeze()
                    ref_local_d = ref_local_d.reshape(-1)[valid_indices]

                    H_ref_to_neareast = ref_to_neareast_r[None] - \
                        torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                    ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                    H_ref_to_neareast = torch.matmul(viewpoint_vircam.get_k(ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                    H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(ncc_scale)
                    
                    ## compute neareast frame patch
                    grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                    grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                    grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                    nearest_feature_map = nearest_render_pkg["feature_map"]
                    if dataset.feat_dim == 32:
                        nearest_feature_map = nearest_feature_map
                    else:
                        nearest_feature_map = cnn_decoder(nearest_feature_map)
                    sampled_gray_val = F.grid_sample(nearest_feature_map.unsqueeze(0), grid.reshape(1, -1, 1, 2), align_corners=True)
                    sampled_gray_val = sampled_gray_val.reshape(C, c_2, total_patch_size)
                    ## compute loss
                    c, bs, tps = sampled_gray_val.shape
                    ps = int(np.sqrt(tps))
                    nea_pool = torch.mean(sampled_gray_val, dim=2)
                    ref_pool = torch.mean(ref_gray_val, dim=2)
                    ncc = (1 - F.cosine_similarity(nea_pool, ref_pool, dim=0))
                    ncc_mask = (ncc < opt.ncc_mask_ratio)
                    ncc_mask = ncc_mask.reshape(-1)
                    ncc = ncc.reshape(-1) * weights
                    ncc = ncc[ncc_mask].squeeze()
                    if ncc_mask.sum() > 0:
                        ncc_loss = ncc_weight * ncc.mean()
                        loss += ncc_loss

        # Posudo-View Feature Regularization
        loss = apply_reference_feature_penalty(
            loss,
            scene,
            name2idx,
            viewpoint_cam,
            surf_depth,
            mask,
            dataset,
            opt,
            iteration,
        )
        
        
        # ---- Stage 2: depth-diff outlier detection (no_grad) ----
        if floater_enabled and outlier_tracker is not None and iteration >= opt.detect_from_iter:
            with torch.no_grad():
                stereo_depth = viewpoint_cam.get_last_stereo_depth()
                valid_mask = viewpoint_cam.get_last_stereo_depth_mask().bool()
                if stereo_depth is not None and valid_mask.any():
                    _, H, W = image.shape
                    det = compute_pixel_outlier_scores(
                        surf_depth.detach(), stereo_depth.detach(), valid_mask,
                        depth_thresh=opt.prune_depth_diff_thresh,
                        rel_floor=opt.prune_rel_floor,
                        mad_k=opt.prune_mad_k)
                    pixel_score = det["score"]
                    pixel_badness = det["badness"]
                    pixel_threshold = det["abs_thresh"]
                    g_score, g_count, g_bad = attribute_scores_to_gaussians(
                        pixel_score, viewpoint_cam, gaussians.get_xyz.detach(), H, W,
                        pixel_badness=pixel_badness,
                        stereo_depth=stereo_depth.detach(),
                        pixel_threshold=pixel_threshold)
                    outlier_tracker.update(g_score, g_count, g_bad)

                    # ---- depth difference statistics (log every 500 iters) ----
                    if iteration % 500 == 0:
                        stats = det.get("stats", {})
                        M = valid_mask.float().squeeze(0)
                        depth_diff = (surf_depth.detach() - stereo_depth.detach()).abs().squeeze(0)
                        valid_diff = depth_diff[M > 0.5]
                        if valid_diff.numel() > 0:
                            dmax = valid_diff.max().item()
                            dmean = valid_diff.mean().item()
                            dmed = valid_diff.median().item()
                            n_total = valid_diff.numel()
                            n_cand = int((pixel_score > 0.5).sum().item())
                            pct_cand = 100.0 * n_cand / max(n_total, 1)
                            n_g = int(g_score.shape[0])
                            n_g_cand = int((g_score > 0.5).sum().item())
                            med_r = stats.get("median_rel_err", float("nan"))
                            rel_t = stats.get("rel_threshold", float("nan"))
                            print(f"[DepthDiff iter={iteration}] "
                                  f"max={dmax:.4f}m mean={dmean:.4f}m median={dmed:.4f}m | "
                                  f"pix_cand={n_cand}/{n_total} ({pct_cand:.1f}%) | "
                                  f"gauss_cand={n_g_cand}/{n_g} ({100.0*n_g_cand/max(n_g,1):.1f}%) | "
                                  f"med_rel={med_r:.4f} rel_thr={rel_t:.4f}")

        # ---- Stage 3: soft opacity decay loss ----
        if floater_enabled and outlier_tracker is not None and iteration >= opt.soft_from_iter:
            L_decay = outlier_tracker.opacity_decay_loss(
                gaussians.get_opacity, min_obs=opt.min_obs_soft)
            loss = loss + lambda_decay * L_decay

        # loss
        total_loss = loss + normal_loss

        # ---- NaN diagnostics ----
        if not torch.isfinite(total_loss):
            with torch.no_grad():
                xyz = gaussians.get_xyz
                op = gaussians.get_opacity
                sc = gaussians.get_scaling
                rot = gaussians.get_rotation
                fd = gaussians.get_features[:, :1]
                print(f"[NaN iter={iteration}] total_loss={total_loss.item()} "
                      f"loss={loss.item() if torch.is_tensor(loss) else loss} "
                      f"normal={normal_loss.item()}")
                print(f"[NaN] xyz nan={int(torch.isnan(xyz).sum())} "
                      f"op nan={int(torch.isnan(op).sum())} range=[{op.min().item():.3f},{op.max().item():.3f}] "
                      f"sc nan={int(torch.isnan(sc).sum())} range=[{sc.min().item():.3f},{sc.max().item():.3f}] "
                      f"rot nan={int(torch.isnan(rot).sum())} "
                      f"fd nan={int(torch.isnan(fd).sum())}")

        total_loss.backward()

        iter_end.record()
       
        # save images
        if iteration % 500 == 0:
            with torch.no_grad():
                visualize(scene, opt, iteration, gaussians, pipe, background, allCameras, virtualCameras, args.model_path)
               

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    keep_mask = gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)
                    # keep outlier tracker in sync after densify/prune
                    if floater_enabled and outlier_tracker is not None:
                        outlier_tracker.sync_after_densify(keep_mask)

                    # ---- NaN guard: sanitize any non-finite gaussian attributes ----
                    with torch.no_grad():
                        nan_mask = (~torch.isfinite(gaussians.get_xyz)).any(dim=1)
                        if nan_mask.any():
                            print(f"[NaN-guard iter={iteration}] removing {int(nan_mask.sum())} NaN gaussians")
                            keep = ~nan_mask
                            gaussians.prune_points(keep)
                            if floater_enabled and outlier_tracker is not None:
                                outlier_tracker.resize(gaussians.get_xyz.shape[0], keep_mask=keep)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # ---- Stage 4: hard pruning + Stage 5: resampling ----
            if (floater_enabled and outlier_tracker is not None
                    and iteration >= opt.hard_from_iter
                    and iteration % opt.prune_interval == 0):
                with torch.no_grad():
                    # --- debug: render before prune ---
                    debug_dir = os.path.join(scene.model_path, "debug_prune")
                    os.makedirs(debug_dir, exist_ok=True)
                    cam_tag = getattr(viewpoint_cam, "image_name", f"cam{iteration}")
                    torchvision.utils.save_image(
                        image.clamp(0, 1),
                        os.path.join(debug_dir, f"iter{iteration:05d}_{cam_tag}_00_before_prune.png"))

                    # Stage 4: hard pruning (gated by enable_hard_prune)
                    prune_mask = None
                    if opt.enable_hard_prune:
                        opacity = gaussians.get_opacity.detach()
                        prune_mask = identify_prune_mask(
                            outlier_tracker.last_score, opacity, opt,
                            badness=outlier_tracker.last_badness)
                        n_prune = int(prune_mask.sum().item())
                        if prune_mask.any():
                            gaussians.prune_points(prune_mask)
                            outlier_tracker.resize(gaussians.get_xyz.shape[0],
                                                   keep_mask=~prune_mask)
                        print(f"[Prune iter={iteration}] "
                              f"pruned {n_prune} / {prune_mask.shape[0]} gaussians "
                              f"({100.0*n_prune/prune_mask.shape[0]:.1f}%)")

                    # --- debug: render after prune, before resample ---
                    render_after_prune = render(viewpoint_cam, gaussians, pipe, background)
                    torchvision.utils.save_image(
                        render_after_prune["render"].clamp(0, 1),
                        os.path.join(debug_dir, f"iter{iteration:05d}_{cam_tag}_01_after_prune.png"))

                    # Stage 5A: stereo-based surface resample
                    if opt.resample_enabled and viewpoint_cam.last_stereo_depth is not None:
                        stereo_depth = viewpoint_cam.get_last_stereo_depth().detach()
                        conf = getattr(viewpoint_cam, "last_stereo_confidence", None)
                        if conf is not None:
                            conf = conf.detach()
                        pts, rgb = collect_reliable_surface_points(
                            viewpoint_cam, stereo_depth, conf,
                            gt_image.detach(),
                            conf_thresh=opt.resample_conf_thresh,
                            max_points=opt.resample_max_points)
                        if pts is not None:
                            new_t = prepare_new_gaussian_tensors(
                                pts, rgb, scene.cameras_extent,
                                gaussians.max_sh_degree, dataset.feat_dim)
                            gaussians.densification_postfix(
                                new_t["xyz"], new_t["knn_f"], new_t["f_dc"],
                                new_t["f_rest"], new_t["surf_feat"],
                                new_t["opacity"], new_t["scaling"], new_t["rotation"])
                            outlier_tracker.resize(gaussians.get_xyz.shape[0])
                            print(f"[Resample5A iter={iteration}] added {pts.shape[0]} points (N={gaussians.get_xyz.shape[0]})")
                        else:
                            print(f"[Resample5A iter={iteration}] added 0 points (N={gaussians.get_xyz.shape[0]})")

                    # Stage 5B: 2D inpainting resample
                    # (uses outlier pixels as holes; works with or without Stage 4)
                    if opt.inpaint_enabled:
                        hole_mask = torch.zeros(
                            surf_depth.shape[-2:], dtype=torch.bool, device="cuda")
                        if outlier_tracker is not None:
                            ph, pw = hole_mask.shape
                            pixel_outlier = outlier_tracker.project_to_pixels(
                                viewpoint_cam, gaussians.get_xyz.detach(), ph, pw)
                            hole_mask = pixel_outlier > 0.5
                        pts2, rgb2 = inpaint_and_resample(
                            viewpoint_cam, surf_depth.detach(),
                            viewpoint_cam.get_last_stereo_depth().detach(),
                            hole_mask,
                            inpaint_max_points=opt.inpaint_max_points)
                        if pts2 is not None:
                            new_t2 = prepare_new_gaussian_tensors(
                                pts2, rgb2, scene.cameras_extent,
                                gaussians.max_sh_degree, dataset.feat_dim)
                            gaussians.densification_postfix(
                                new_t2["xyz"], new_t2["knn_f"], new_t2["f_dc"],
                                new_t2["f_rest"], new_t2["surf_feat"],
                                new_t2["opacity"], new_t2["scaling"], new_t2["rotation"])
                            outlier_tracker.resize(gaussians.get_xyz.shape[0])
                            print(f"[Resample5B iter={iteration}] added {pts2.shape[0]} points (N={gaussians.get_xyz.shape[0]})")
                        else:
                            print(f"[Resample5B iter={iteration}] added 0 points (N={gaussians.get_xyz.shape[0]})")

                    # --- debug: render after resample ---
                    render_after_resample = render(viewpoint_cam, gaussians, pipe, background)
                    torchvision.utils.save_image(
                        render_after_resample["render"].clamp(0, 1),
                        os.path.join(debug_dir, f"iter{iteration:05d}_{cam_tag}_02_after_resample.png"))
                    # also save GT for reference
                    torchvision.utils.save_image(
                        gt_image.clamp(0, 1),
                        os.path.join(debug_dir, f"iter{iteration:05d}_{cam_tag}_03_gt.png"))

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.vis_utils import colormap
                        depth = render_pkg["plane_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rendered_alpha']
                            rend_normal = render_pkg["rendered_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["depth_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    if viewpoint.gt_alpha_mask is not None:
                        object_mask = (viewpoint.gt_alpha_mask > 0.5).view(-1)
                        image = image.view(3, -1)[:, object_mask]
                        gt_image = gt_image.view(3, -1)[:, object_mask]
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

def visualize(scene, opt, iteration, gaussians, pipe, background, allCameras, virtualCameras, save_path):
    eval_cam = allCameras[random.randint(0, len(allCameras) - 1)]
    gt_image = eval_cam.original_image.cuda()
    test_render_pkg = render(eval_cam, gaussians, pipe, background)
    image = test_render_pkg['render']
    _, imgH, imgW = image.shape
    depth = test_render_pkg['plane_depth']  # [1, h, w]
    depth_normal = test_render_pkg['depth_normal']  # 3, h, w

    render_normal = test_render_pkg['rendered_normal']

    accumlated_alpha = test_render_pkg['rendered_alpha']  # 1, h, w
    colored_accum_alpha = accumlated_alpha.repeat(3, 1, 1)

    color_percent = 0.10
    min_depth, max_depth = float(torch.quantile(depth, color_percent)), float(torch.quantile(depth, 0.99))
    colored_depth = apply_depth_colormap(depth.squeeze(0), None, near_plane=min_depth, far_plane=max_depth).permute(2, 0, 1).contiguous()

    colored_depth_normal = apply_normal_colormap(depth_normal)
    colored_render_normal = apply_normal_colormap(render_normal)
    
    images = [
        gt_image, image, colored_accum_alpha, colored_depth, colored_depth_normal, colored_render_normal
    ]
   
    prior_feature_scale1 = eval_cam.feat[0].cuda()
    prior_feature_scale1_vis = pca_feature(prior_feature_scale1)
    images.append(prior_feature_scale1_vis)
    rend_feature_map = test_render_pkg["feature_map"]
    render_feature_vis = pca_feature(rend_feature_map)
    images.append(render_feature_vis)
    
    ## visual stereo pair
    if eval_cam.last_stereo_depth is not None:
        stereo_depth = eval_cam.get_last_stereo_depth().cuda()
        stereo_depth_mask = eval_cam.get_last_stereo_depth_mask().cuda().float()
        colored_stereo_depth = apply_depth_colormap(stereo_depth.squeeze(0), None, near_plane=min_depth, far_plane=max_depth)
        images.append(colored_stereo_depth.permute(2, 0, 1).contiguous())
        stereo_depth_normal = render_normal_func(eval_cam, stereo_depth.squeeze())
        images.append((stereo_depth_normal + 1.) / 2.)
        images.append(stereo_depth_mask.repeat(3, 1, 1))
        right_image_vis = eval_cam.get_last_right_image()
        images.append(right_image_vis)

    ## visual virtual cams
    vir_cam = virtualCameras[random.randint(0, len(virtualCameras) - 1)] 
    vir_render_pkg = render(vir_cam, gaussians, pipe, background)
    vir_render_image = vir_render_pkg['render']
    images.append(vir_render_image)

    def image_grid(xs, n_col=5):
        """ n x [3, h, w]"""
        if len(xs) % n_col > 0:
            emptys = (n_col - len(xs) % n_col) * [torch.ones_like(xs[0]), ]
            xs = xs + emptys
        rets = []
        for i in range(len(xs) // n_col):
            rets.append(torch.cat(xs[i * n_col:(i + 1) * n_col], dim=2))
        return torch.cat(rets, dim=1)

    image_to_show = image_grid(images, n_col=5)
    image_to_show = torch.clamp(image_to_show, 0, 1)

    os.makedirs(f"{save_path}/log_images", exist_ok=True)
    torchvision.utils.save_image(image_to_show, f"{save_path}/log_images/{iteration}.jpg")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
