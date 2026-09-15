import torch


def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    intrins = view.intrinsic[:3, :3]
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


@torch.no_grad()
def inpaint_and_resample(viewpoint_cam, render_depth, stereo_depth, hole_mask,
                         inpaint_window=15, inpaint_hole_dilate_radius=3,
                         inpaint_min_healthy=8, inpaint_depth_err_thr=0.02,
                         inpaint_idw_p=2.0, inpaint_max_points=15000):
    """Stage 5.5: 2D neighbor inpainting resample.

    Fill 2D holes (from hard pruning) by IDW-interpolating depth from healthy
    neighbors (where render depth agrees with stereo depth), then back-project.

    Args:
        render_depth, stereo_depth: (1, H, W)
        hole_mask: (H, W) bool  -- pixels corresponding to pruned gaussians
    Returns:
        (pts_world (M,3), rgb (M,3)) or (None, None)
    """
    import torch.nn.functional as F
    H, W = hole_mask.shape
    rd = render_depth.squeeze(0)
    sd = stereo_depth.squeeze(0)

    # dilate the hole mask to avoid boundary noise
    if inpaint_hole_dilate_radius > 0:
        k = 2 * inpaint_hole_dilate_radius + 1
        hole = hole_mask.float()[None, None]
        hole = F.max_pool2d(hole, k, stride=1, padding=inpaint_hole_dilate_radius)
        hole_mask = (hole.squeeze() > 0.5)

    # healthy pixels: stereo valid and relative depth error small
    valid_stereo = sd > 0
    rel_err = (rd - sd).abs() / (sd + 1e-3)
    healthy = valid_stereo & (rel_err < inpaint_depth_err_thr)

    if not hole_mask.any() or not healthy.any():
        return None, None

    # gather hole pixels
    hy, hx = torch.nonzero(hole_mask, as_tuple=True)
    if hy.numel() > inpaint_max_points:
        idx = torch.randperm(hy.numel(), device=hy.device)[:inpaint_max_points]
        hy, hx = hy[idx], hx[idx]

    # build local neighbor offsets
    offs = torch.arange(-inpaint_window, inpaint_window + 1, device=hy.device)
    oy, ox = torch.meshgrid(offs, offs, indexing='ij')
    oy, ox = oy.reshape(-1), ox.reshape(-1)
    center = (oy == 0) & (ox == 0)
    oy, ox = oy[~center], ox[~center]

    ny = hy[:, None] + oy[None]
    nx = hx[:, None] + ox[None]
    in_bounds = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
    n_healthy = healthy[ny.clamp(0, H - 1), nx.clamp(0, W - 1)] & in_bounds

    # IDW weights
    dist = (oy[None] ** 2 + ox[None] ** 2).float().sqrt()
    weights = 1.0 / (dist ** inpaint_idw_p + 1e-8)
    weights = weights * n_healthy.float()

    enough = weights.sum(dim=1) > 0
    if not enough.any():
        return None, None

    # interpolate depth
    neighbor_depth = sd[ny[enough].clamp(0, H - 1), nx[enough].clamp(0, W - 1)]
    w = weights[enough]
    inpainted_depth = (w * neighbor_depth).sum(dim=1) / w.sum(dim=1).clamp(min=1e-8)

    # back-project
    fx, fy = viewpoint_cam.Fx, viewpoint_cam.Fy
    cx, cy = viewpoint_cam.Cx, viewpoint_cam.Cy
    hx_sel = hx[enough].float()
    hy_sel = hy[enough].float()
    xs_c = (hx_sel - cx) / fx
    ys_c = (hy_sel - cy) / fy
    pts_cam = torch.stack([xs_c * inpainted_depth, ys_c * inpainted_depth, inpainted_depth], dim=-1)
    R = torch.tensor(viewpoint_cam.R, device=pts_cam.device).float()
    T = torch.tensor(viewpoint_cam.T, device=pts_cam.device).float()
    pts_world = pts_cam @ R.transpose(-1, -2) + T

    # use stereo depth color at neighbor (approx); fall back to gray
    rgb = torch.full((pts_world.shape[0], 3), 0.5, device=pts_world.device)
    return pts_world, rgb

