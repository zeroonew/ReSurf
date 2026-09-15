# SparseSurf six-stage floater suppression pipeline.
#
# Stages:
#   1  Prevention      SpatialConstraint L_space (bbox from init point cloud)
#   2  Detection       multi-signal pixel outlier score + gaussian attribute
#                      anomaly, accumulated into OutlierTracker via EMA
#   3  Soft Penalty    opacity decay loss L_decay = mean(phi * opacity)
#   4  Hard Pruning    3-condition prune with max_prune_ratio + ramp-up
#   5  Resampling      (A) stereo high-conf back-projection, (B) 2D IDW inpaint
#   6  Scheduling      color down-weighting + loss-weight schedules
#
# All Stage-2 bookkeeping runs under torch.no_grad() so Stage 3/4 backward
# never re-enters the detection graph.
import math
import torch
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
# Stage 6  Schedulers
# --------------------------------------------------------------------------- #
def get_scheduler_lambdas(iteration, opt):
    """Return (lambda_space, lambda_decay, gamma) for the current iteration."""
    # Stage 1: spatial constraint, ramps from 0 to lambda_space_max over first
    # 5k iters.
    lambda_space = min(opt.lambda_space_max,
                       opt.lambda_space_max * (iteration / 5000.0))

    # Stage 3: opacity decay. 0 before soft_from_iter, linear ramp to
    # lambda_decay_max over (soft_from_iter -> soft_from_iter+2000).
    if iteration < opt.soft_from_iter:
        lambda_decay = 0.0
    else:
        ramp = min(1.0, (iteration - opt.soft_from_iter) / 2000.0)
        lambda_decay = opt.lambda_decay_max * ramp

    # Stage 6: color down-weighting gamma. Same ramp as opacity decay.
    if iteration < opt.soft_from_iter:
        gamma = 0.0
    else:
        ramp = min(1.0, (iteration - opt.soft_from_iter) / 2000.0)
        gamma = opt.gamma_max * ramp

    return lambda_space, lambda_decay, gamma


def smoothstep(edge0, edge1, x):
    """Hermite smoothstep; clamps x to [edge0, edge1]."""
    t = torch.clamp((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- #
# Stage 1  Spatial constraint
# --------------------------------------------------------------------------- #
class SpatialConstraint:
    """Valid-region bbox built from the COLMAP initial point cloud."""

    def __init__(self, points3d, scene_radius, margin_ratio=0.4):
        pts = np.asarray(points3d)
        if pts.size == 0:
            self.bbox_min = torch.zeros(3, device="cuda")
            self.bbox_max = torch.ones(3, device="cuda")
            self.scene_radius = float(scene_radius)
            return
        p5 = np.percentile(pts, 5, axis=0)
        p95 = np.percentile(pts, 95, axis=0)
        extent = p95 - p5
        self.bbox_min = torch.from_numpy(p5 - extent * margin_ratio).float().cuda()
        self.bbox_max = torch.from_numpy(p95 + extent * margin_ratio).float().cuda()
        self.scene_radius = float(scene_radius)

    def distance_to_valid_region(self, means):
        """Per-gaussian L2 distance to the bbox (0 inside)."""
        below = (self.bbox_min - means).clamp(min=0.0)
        above = (means - self.bbox_max).clamp(min=0.0)
        return torch.norm(below + above, dim=-1)

    def loss(self, means, opacity):
        allowed_margin = self.scene_radius * 0.08
        dist = self.distance_to_valid_region(means)
        penalty = F.relu(dist - allowed_margin) ** 2
        return (opacity.squeeze() * penalty).mean()


# --------------------------------------------------------------------------- #
# Stage 2  Multi-signal pixel outlier detection
# --------------------------------------------------------------------------- #
def compute_dynamic_threshold(E_depth, valid):
    """MAD-based per-view adaptive threshold. Returns (tau_low, tau_high)."""
    vals = E_depth[valid]
    if vals.numel() < 10:
        return torch.tensor(1e6, device=E_depth.device), torch.tensor(1e6, device=E_depth.device)
    median = vals.median()
    mad = (vals - median).abs().median()
    sigma = 1.4826 * mad
    tau_low = median + 2.0 * sigma
    tau_high = median + 4.0 * sigma
    return tau_low, tau_high


@torch.no_grad()
def compute_pixel_outlier_scores(render_depth, stereo_depth, valid_mask,
                                 depth_thresh=0.3, rel_floor=0.01, mad_k=2.5,
                                 depth_min=1e-6, min_valid_pixels=1024):
    """Adaptive relative-error outlier detection.

    Instead of a fixed absolute threshold (e.g. 1 cm), this computes a
    per-pixel absolute threshold tau(p) = max(abs_floor, D_s(p) * theta_t),
    where theta_t = max(rel_floor, median(r) + mad_k * 1.4826 * MAD(r)) and
    r(p) = |D_r(p) - D_s(p)| / D_s(p) is the relative depth error.

    Returns a dict with:
      - score:      (H, W) binary candidate mask in {0, 1}
      - badness:    (H, W) exceedance ratio max(e/tau - 1, 0), for ranking
      - abs_thresh: (H, W) per-pixel absolute threshold tau(p)
      - stats:      dict of diagnostic numbers
    """
    M = valid_mask.bool().squeeze(0)
    rd = render_depth.squeeze(0)
    sd = stereo_depth.squeeze(0)

    valid = (M
             & torch.isfinite(rd) & torch.isfinite(sd)
             & (rd > depth_min) & (sd > depth_min))

    H, W = sd.shape
    empty_score = torch.zeros((H, W), device=sd.device)
    empty_bad = torch.zeros((H, W), device=sd.device)

    if int(valid.sum()) < min_valid_pixels:
        return {
            "score": empty_score,
            "badness": empty_bad,
            "abs_thresh": empty_bad.clone(),
            "stats": {"enabled": False, "reason": "insufficient_valid_pixels",
                      "valid_pixels": int(valid.sum())},
        }

    abs_error = (rd - sd).abs()
    rel_error = abs_error / sd.clamp_min(depth_min)

    values = rel_error[valid]
    median = values.median()
    mad = (values - median).abs().median()
    robust_scale = 1.4826 * mad

    relative_threshold = torch.maximum(
        median + mad_k * robust_scale,
        median.new_tensor(rel_floor),
    )

    abs_floor = torch.full_like(sd, float(depth_thresh))
    abs_thresh = torch.maximum(sd * relative_threshold, abs_floor)

    candidates = valid & (abs_error > abs_thresh)
    score = candidates.float()

    # exceedance ratio for ranking (>=0, higher = more anomalous)
    badness = torch.clamp(abs_error / abs_thresh.clamp_min(depth_min) - 1.0, min=0.0)
    badness = badness * candidates.float()  # zero out non-candidates

    return {
        "score": score,
        "badness": badness,
        "abs_thresh": abs_thresh,
        "stats": {
            "enabled": True,
            "median_rel_err": float(median.item()),
            "mad_scale": float(robust_scale.item()),
            "rel_threshold": float(relative_threshold.item()),
            "valid_pixels": int(valid.sum()),
            "candidate_ratio": float((candidates.sum().float() / valid.sum()).item()),
        },
    }


@torch.no_grad()
def gaussian_attribute_anomaly(gaussians):
    """Per-gaussian attribute anomaly in [0, 1] (scale / anisotropy / opacity
    cheat / high-SH). Returns (N,) tensor."""
    scaling = gaussians.get_scaling  # (N, 3)
    scale_mag = scaling.norm(dim=-1)
    scale_anom = smoothstep(
        scale_mag.quantile(0.95), scale_mag.quantile(0.99), scale_mag)

    sorted_s, _ = torch.sort(scaling, dim=-1)
    aniso = sorted_s[:, -1] / (sorted_s[:, 0] + 1e-6)
    aniso_anom = smoothstep(aniso.quantile(0.95), aniso.quantile(0.99), aniso)

    opacity = gaussians.get_opacity.squeeze(-1)
    op_anom = smoothstep(0.995, 1.0, opacity)

    features = gaussians.get_features  # (N, k, 3)
    sh_rest = features[:, 1:, :] if features.shape[1] > 1 else features[:, :0, :]
    if sh_rest.numel() > 0:
        sh_anom = smoothstep(
            sh_rest.norm(dim=(1, 2)).quantile(0.95),
            sh_rest.norm(dim=(1, 2)).quantile(0.99),
            sh_rest.norm(dim=(1, 2)))
    else:
        sh_anom = torch.zeros_like(scale_anom)

    combined = (0.4 * scale_anom + 0.3 * aniso_anom
                + 0.2 * op_anom + 0.1 * sh_anom)
    p99 = combined.quantile(0.99).clamp(min=1e-6)
    return (combined / p99).clamp(0.0, 1.0)


@torch.no_grad()
def attribute_scores_to_gaussians(pixel_score, viewpoint_cam, means, H, W,
                                  pixel_badness=None, stereo_depth=None,
                                  pixel_threshold=None, depth_min=1e-6):
    """Project per-pixel outlier score onto the gaussians visible in this view.

    If ``stereo_depth`` and ``pixel_threshold`` are provided, the gaussian's
    own camera-space depth is compared against the stereo depth at its
    projected pixel.  A gaussian is flagged only when it lies clearly IN FRONT
    of the stereo surface (f_i = D_s - z_i > tau), which is the geometric
    signature of a floater.  Gaussians behind the surface (possibly occluded
    structure) are NOT flagged.

    Returns (gaussian_score, gaussian_count, gaussian_badness) each of shape
    (N,).  Without stereo_depth it falls back to copying the pixel score.
    """
    w2c = viewpoint_cam.world_view_transform  # (4, 4)
    pts = means @ w2c[:3, :3] + w2c[3, :3]
    fx, fy = viewpoint_cam.Fx, viewpoint_cam.Fy
    cx, cy = viewpoint_cam.Cx, viewpoint_cam.Cy
    u = pts[:, 0] * fx / (pts[:, 2] + 1e-8) + cx
    v = pts[:, 1] * fy / (pts[:, 2] + 1e-8) + cy
    in_front = pts[:, 2] > 0.1
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid = in_front & in_img

    N = means.shape[0]
    g_score = torch.zeros(N, device=means.device)
    g_count = torch.zeros(N, device=means.device, dtype=torch.long)
    g_bad = torch.zeros(N, device=means.device)
    if not valid.any():
        return g_score, g_count, g_bad

    ui = u[valid].long().clamp(0, W - 1)
    vi = v[valid].long().clamp(0, H - 1)

    if stereo_depth is not None and pixel_threshold is not None:
        # Gaussian-geometry front-conflict: f_i = D_s(p_i) - z_i
        z_i = pts[valid][:, 2]
        sd = stereo_depth.squeeze(0)[vi, ui]
        tau = pixel_threshold[vi, ui]
        f_i = sd - z_i
        is_candidate = (sd > depth_min) & (f_i > tau)
        g_score[valid] = is_candidate.float()
        g_bad[valid] = torch.clamp(
            f_i / tau.clamp_min(depth_min) - 1.0, min=0.0) * is_candidate.float()
    else:
        g_score[valid] = pixel_score[vi, ui]
        if pixel_badness is not None:
            g_bad[valid] = pixel_badness[vi, ui]

    g_count[valid] = 1
    return g_score, g_count, g_bad


# --------------------------------------------------------------------------- #
# Stage 2 -> 3 bridge : OutlierTracker (latest binary score, no EMA)
# --------------------------------------------------------------------------- #
class OutlierTracker:
    def __init__(self, num_gaussians, device="cuda", beta=0.95):
        self.device = device
        # last_score: latest binary outlier flag per gaussian (0 / 1)
        self.last_score = torch.zeros(num_gaussians, device=device)
        # last_badness: latest exceedance ratio per gaussian (for ranking)
        self.last_badness = torch.zeros(num_gaussians, device=device)
        self.obs_count = torch.zeros(num_gaussians, dtype=torch.long, device=device)

    def update(self, new_scores, new_counts, new_badness=None):
        """Just store the latest score; no EMA accumulation."""
        new_scores = new_scores.detach()
        new_counts = new_counts.detach()
        mask = new_counts > 0
        self.last_score[mask] = new_scores[mask]
        self.obs_count[mask] += new_counts[mask].to(torch.long)
        if new_badness is not None:
            self.last_badness[mask] = new_badness.detach()[mask]

    def resize(self, new_N, keep_mask=None):
        if keep_mask is not None:
            self.last_score = self.last_score[keep_mask]
            self.last_badness = self.last_badness[keep_mask]
            self.obs_count = self.obs_count[keep_mask]
        else:
            cur = self.last_score.shape[0]
            extra = new_N - cur
            if extra > 0:
                self.last_score = torch.cat(
                    [self.last_score, torch.zeros(extra, device=self.device)])
                self.last_badness = torch.cat(
                    [self.last_badness, torch.zeros(extra, device=self.device)])
                self.obs_count = torch.cat(
                    [self.obs_count,
                     torch.zeros(extra, dtype=torch.long, device=self.device)])
            elif extra < 0:
                self.last_score = self.last_score[:new_N]
                self.last_badness = self.last_badness[:new_N]
                self.obs_count = self.obs_count[:new_N]

    def sync_after_densify(self, keep_mask):
        full_N = keep_mask.shape[0]
        cur = self.last_score.shape[0]
        pad = full_N - cur
        if pad > 0:
            self.last_score = torch.cat(
                [self.last_score, torch.zeros(pad, device=self.device)])
            self.last_badness = torch.cat(
                [self.last_badness, torch.zeros(pad, device=self.device)])
            self.obs_count = torch.cat(
                [self.obs_count,
                 torch.zeros(pad, dtype=torch.long, device=self.device)])
        self.last_score = self.last_score[keep_mask]
        self.last_badness = self.last_badness[keep_mask]
        self.obs_count = self.obs_count[keep_mask]

    def opacity_decay_loss(self, opacities, min_obs=3):
        """L_decay = mean(phi * opacity) for flagged outliers."""
        valid = (self.obs_count >= min_obs).float()
        phi = valid * self.last_score.detach()
        return (phi * opacities.squeeze()).mean()

    @torch.no_grad()
    def project_to_pixels(self, viewpoint_cam, means, H, W):
        """Project per-gaussian outlier flag back to a 2D pixel map."""
        w2c = viewpoint_cam.world_view_transform
        pts = means @ w2c[:3, :3] + w2c[3, :3]
        fx, fy = viewpoint_cam.Fx, viewpoint_cam.Fy
        cx, cy = viewpoint_cam.Cx, viewpoint_cam.Cy
        u = pts[:, 0] * fx / (pts[:, 2] + 1e-8) + cx
        v = pts[:, 1] * fy / (pts[:, 2] + 1e-8) + cy
        in_front = pts[:, 2] > 0.1
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid = in_front & in_img
        pixel_outlier = torch.zeros(H, W, device=means.device)
        if valid.any():
            ui = u[valid].long().clamp(0, W - 1)
            vi = v[valid].long().clamp(0, H - 1)
            pixel_outlier[vi, ui] = self.last_score[valid]
        return pixel_outlier


# --------------------------------------------------------------------------- #
# Stage 4  Hard pruning (simple depth-diff threshold)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def identify_prune_mask(score, opacity, opt, badness=None):
    """Prune if the gaussian was flagged as a depth outlier (score==1) or is
    nearly transparent. Candidates are ranked by exceedance badness when
    available, so the most anomalous gaussians are removed first.

    score:    (N,) latest binary outlier flag (0/1)
    opacity:  (N, 1)
    badness:  (N,) exceedance ratio from adaptive threshold (higher = worse)
    Returns boolean prune_mask (N,).
    """
    cond1 = score > 0.5
    cond2 = opacity.squeeze() < 0.005
    prune_mask = cond1 | cond2

    total = int(prune_mask.sum().item())
    ratio_eff = float(np.clip(opt.max_prune_ratio, 0.005, 0.25))
    max_allowed = int(score.shape[0] * ratio_eff)
    if total > max_allowed and max_allowed > 0:
        # Rank by exceedance badness if available; fall back to opacity.
        if badness is not None:
            rank_bad = badness.detach()
        else:
            rank_bad = 1.0 - opacity.squeeze()
        order = torch.argsort(rank_bad[prune_mask], descending=True)
        keep_from = order[max_allowed:]
        global_idx = torch.nonzero(prune_mask, as_tuple=False).squeeze(-1)
        prune_mask[global_idx[keep_from]] = False

    return prune_mask


# --------------------------------------------------------------------------- #
# Stage 5A  Stereo-based surface resampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_reliable_surface_points(viewpoint_cam, stereo_depth, confidence,
                                    gt_rgb, conf_thresh=0.7, max_points=30000):
    """Back-project high-confidence stereo pixels to world points.

    Returns (xyz_world (M,3), rgb (M,3)) or (None, None) if none.
    """
    valid = (stereo_depth.squeeze(0) > 0)
    if confidence is not None:
        valid = valid & (confidence.squeeze(0) > conf_thresh)
    ys, xs = torch.nonzero(valid, as_tuple=True)
    if xs.numel() == 0:
        return None, None
    if xs.numel() > max_points:
        idx = torch.randperm(xs.numel(), device=xs.device)[:max_points]
        xs, ys = xs[idx], ys[idx]

    fx, fy = viewpoint_cam.Fx, viewpoint_cam.Fy
    cx, cy = viewpoint_cam.Cx, viewpoint_cam.Cy
    depth = stereo_depth.squeeze(0)[ys, xs]
    xs_c = (xs.float() - cx) / fx
    ys_c = (ys.float() - cy) / fy
    pts_cam = torch.stack([xs_c * depth, ys_c * depth, depth], dim=-1)

    R = torch.tensor(viewpoint_cam.R, device=depth.device).float()
    T = torch.tensor(viewpoint_cam.T, device=depth.device).float()
    pts_world = pts_cam @ R.transpose(-1, -2) + T

    rgb = gt_rgb[:, ys, xs].transpose(0, 1)  # (M, 3)
    return pts_world, rgb


def prepare_new_gaussian_tensors(pts_world, rgb, scene_extent, sh_degree, feat_dim=8):
    """Build parameter tensors for new gaussians from resampled points."""
    from utils.sh_utils import RGB2SH
    n = pts_world.shape[0]
    device = pts_world.device
    new_xyz = pts_world + torch.randn_like(pts_world) * 0.001
    scale = torch.log(torch.full((n, 3), 0.005, device=device))
    rot = torch.zeros((n, 4), device=device)
    rot[:, 0] = 1.0
    opacity = torch.logit(torch.full((n, 1), 0.3, device=device))
    f_dc = RGB2SH(rgb).unsqueeze(1)  # (N, 1, 3)
    f_rest = torch.zeros((n, max((sh_degree + 1) ** 2 - 1, 0), 3), device=device)
    knn_f = torch.randn((n, 6), device=device) * 0.01
    surf_feat = torch.zeros((n, feat_dim), device=device)
    return {
        "xyz": new_xyz,
        "knn_f": knn_f,
        "f_dc": f_dc,
        "f_rest": f_rest,
        "surf_feat": surf_feat,
        "opacity": opacity,
        "scaling": scale,
        "rotation": rot,
    }
