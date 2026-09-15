# ConfSurf: confidence-driven utilities
# A1: continuous stereo confidence (replaces binary LR-check mask)
# A2: multi-baseline confidence-weighted depth fusion
# B:  normal-guided depth propagation for occluded / low-confidence regions

import torch
import torch.nn.functional as F


def lr_residual(L2R_disparity, R2L_disparity):
    """Continuous left-right reprojection residual (in pixels).

    Same geometry as the binary left_right_check, but returns the raw
    residual instead of thresholding it.

    Args:
        L2R_disparity: (B, H, W) left-to-right disparity
        R2L_disparity: (B, H, W) right-to-left disparity
    Returns:
        residual: (B, H, W) float reprojection error, -1 where invalid
    """
    batch_size, height, width = L2R_disparity.shape
    x_grid = torch.arange(width, device=L2R_disparity.device).view(1, 1, -1).repeat(batch_size, height, 1)

    x_projected = (x_grid - L2R_disparity).long()
    valid = (x_projected >= 0) & (x_projected < width)
    x_projected_clipped = torch.clamp(x_projected, 0, width - 1)

    x_reprojected = x_projected_clipped + R2L_disparity.gather(2, x_projected_clipped)
    x_reprojected_clipped = torch.clamp(x_reprojected, 0, width - 1)

    residual = torch.abs(x_grid - x_reprojected_clipped).float()
    residual[~valid] = -1.0
    return residual


def left_right_confidence(L2R_disparity, R2L_disparity, tau=1.0, hard_threshold=None):
    """Continuous per-pixel stereo confidence in [0, 1].

    conf = exp(-residual / tau), with conf = 0 where the reprojection
    falls outside the frame or the disparity is non-positive.

    Args:
        L2R_disparity: (B, H, W)
        R2L_disparity: (B, H, W)
        tau: softness of the exponential falloff (pixels)
        hard_threshold: if given, additionally zero out pixels whose
            residual exceeds this value (safety valve against gross mismatches)
    Returns:
        confidence: (B, H, W) float in [0, 1]
    """
    residual = lr_residual(L2R_disparity, R2L_disparity)
    conf = torch.exp(-residual.clamp(min=0.0) / tau)
    invalid = (residual < 0) | (L2R_disparity <= 0)
    if hard_threshold is not None:
        invalid = invalid | (residual > hard_threshold)
    conf[invalid] = 0.0
    return conf


def multi_baseline_fusion(depths, confs, tau_rel=0.02):
    """Confidence-weighted fusion of multi-baseline stereo depths.

    Args:
        depths: list of N tensors (1, H, W) or (H, W), metric depths
        confs:  list of N tensors, same shape, confidence in [0, 1]
        tau_rel: relative depth disagreement tolerance for the agreement term
    Returns:
        fused_depth: same shape as input
        fused_conf:  same shape, fused confidence discounted by cross-baseline agreement
    """
    depths = [d.unsqueeze(0) if d.dim() == 2 else d for d in depths]
    confs = [c.unsqueeze(0) if c.dim() == 2 else c for c in confs]

    depth_stack = torch.stack(depths, dim=0)      # (N, 1, H, W)
    conf_stack = torch.stack(confs, dim=0)        # (N, 1, H, W)

    conf_sum = conf_stack.sum(dim=0).clamp(min=1e-8)
    fused_depth = (depth_stack * conf_stack).sum(dim=0) / conf_sum

    # cross-baseline agreement: relative weighted std of depths
    diff = (depth_stack - fused_depth.unsqueeze(0)) ** 2
    var = (conf_stack * diff).sum(dim=0) / conf_sum
    rel_std = torch.sqrt(var.clamp(min=1e-12)) / fused_depth.clamp(min=1e-6)
    agreement = torch.exp(-rel_std / tau_rel)

    fused_conf = (conf_stack.sum(dim=0) / len(depths)) * agreement
    fused_conf[conf_sum <= 1e-7] = 0.0
    return fused_depth, fused_conf


def _backproject(depth, fx, fy, cx, cy):
    """(H, W) depth -> (H, W, 3) camera-space points."""
    H, W = depth.shape
    device = depth.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=depth.dtype),
        torch.arange(W, device=device, dtype=depth.dtype),
        indexing='ij')
    X = (xs - cx) * depth / fx
    Y = (ys - cy) * depth / fy
    return torch.stack([X, Y, depth], dim=-1)


@torch.no_grad()
def normal_guided_depth_propagation(depth, confidence, normal, fx, fy, cx, cy,
                                    n_iters=16, conf_high=0.6, gamma=0.85,
                                    min_conf=1e-3):
    """Propagate high-confidence depth into low-confidence regions along the
    surface tangent direction, guided by rendered normals.

    For every low-confidence pixel we look at its 4 image neighbours. A
    neighbour votes with the depth of its tangent plane intersected by the
    pixel's viewing ray; the vote is weighted by the neighbour confidence and
    by how tangent the pixel direction is to the neighbour's surface
    (1 - |n . dir|). Propagated confidence decays by gamma per hop.

    Args:
        depth: (H, W) current depth map
        confidence: (H, W) confidence in [0, 1]
        normal: (3, H, W) camera-space unit normals
        fx, fy, cx, cy: intrinsics
        n_iters: number of propagation sweeps
        conf_high: pixels above this are treated as anchors (never overwritten)
        gamma: per-hop confidence decay
        min_conf: ignore votes below this confidence
    Returns:
        prop_depth: (H, W) depth with low-confidence regions filled
        prop_conf:  (H, W) confidence of the filled values (0 for anchors' original conf kept)
    """
    device = depth.device
    H, W = depth.shape

    cur_depth = depth.clone()
    cur_conf = confidence.clone()
    anchor = confidence >= conf_high

    normal = F.normalize(normal, dim=0)
    nx, ny, nz = normal[0], normal[1], normal[2]

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=depth.dtype),
        torch.arange(W, device=device, dtype=depth.dtype),
        indexing='ij')
    ray_x = (xs - cx) / fx
    ray_y = (ys - cy) / fy

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(n_iters):
        acc_depth = torch.zeros_like(cur_depth)
        acc_weight = torch.zeros_like(cur_depth)
        acc_conf = torch.zeros_like(cur_conf)

        for dy, dx in offsets:
            # neighbour slice
            sy0, sy1 = (max(0, -dy), H - max(0, dy))
            sx0, sx1 = (max(0, -dx), W - max(0, dx))
            ty0, ty1 = sy0 + dy, sy1 + dy
            tx0, tx1 = sx0 + dx, sx1 + dx

            nb_depth = cur_depth[sy0:sy1, sx0:sx1]
            nb_conf = cur_conf[sy0:sy1, sx0:sx1]
            nb_nx = nx[sy0:sy1, sx0:sx1]
            nb_ny = ny[sy0:sy1, sx0:sx1]
            nb_nz = nz[sy0:sy1, sx0:sx1]

            # neighbour 3D position
            nb_X = (xs[sy0:sy1, sx0:sx1] - cx) * nb_depth / fx
            nb_Y = (ys[sy0:sy1, sx0:sx1] - cy) * nb_depth / fy
            nb_Z = nb_depth

            # target ray direction
            r_x = ray_x[ty0:ty1, tx0:tx1]
            r_y = ray_y[ty0:ty1, tx0:tx1]

            # tangent-plane / ray intersection: t = n.P / (n.r)
            n_dot_p = nb_nx * nb_X + nb_ny * nb_Y + nb_nz * nb_Z
            n_dot_r = nb_nx * r_x + nb_ny * r_y + nb_nz
            valid_plane = n_dot_r.abs() > 1e-6
            cand_depth = torch.where(valid_plane, n_dot_p / n_dot_r.clamp(min=1e-6), nb_depth)
            cand_depth = cand_depth.clamp(min=1e-3)

            # tangent alignment: direction neighbour -> target pixel in 3D
            tgt_X = r_x * cand_depth
            tgt_Y = r_y * cand_depth
            dirx = tgt_X - nb_X
            diry = tgt_Y - nb_Y
            dirz = cand_depth - nb_Z
            dir_norm = torch.sqrt(dirx ** 2 + diry ** 2 + dirz ** 2).clamp(min=1e-8)
            cos = (nb_nx * dirx + nb_ny * diry + nb_nz * dirz) / dir_norm
            align = (1.0 - cos.abs()).clamp(min=0.0) ** 2

            w = nb_conf * align
            vote_conf = nb_conf * gamma * align

            good = (nb_conf > min_conf) & (nb_depth > 0) & (cand_depth > 0)
            w = torch.where(good, w, torch.zeros_like(w))
            vote_conf = torch.where(good, vote_conf, torch.zeros_like(vote_conf))

            acc_depth[ty0:ty1, tx0:tx1] += w * cand_depth
            acc_weight[ty0:ty1, tx0:tx1] += w
            acc_conf[ty0:ty1, tx0:tx1] = torch.maximum(acc_conf[ty0:ty1, tx0:tx1], vote_conf)

        has_vote = acc_weight > 1e-8
        new_depth = torch.where(has_vote, acc_depth / acc_weight.clamp(min=1e-8), cur_depth)
        new_conf = acc_conf

        # only fill non-anchor pixels where the propagated confidence improves
        fill = (~anchor) & has_vote & (new_conf > cur_conf)
        cur_depth = torch.where(fill, new_depth, cur_depth)
        cur_conf = torch.where(fill, new_conf, cur_conf)

    return cur_depth, cur_conf
