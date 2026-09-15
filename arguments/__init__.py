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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.use_mask = False
        self.valid_depth_radius = 2.0
        self.total_virtual_num = 360
        self.feat_dim = 8
        self.n_views = 3
        self.stereo_init_num = -1
        self.reinit_save_path = ""
        self.foundation_stereo_ckpt = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        self.stage = 'render'
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 7_000 
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000 #origin 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 10000.0
        self.lambda_normal = 0.05
        self.lambda_dist_from_iter = 1500
        self.lambda_normal_from_iter = 3000
        self.lambda_feat = 1.5
        self.lambda_normal_smooth = 0.05
        self.lambda_normal_prior = 0.05
        self.opacity_cull = 0.1
        

        self.densification_interval = 100
        self.opacity_reset_interval = 1000 
        self.densify_from_iter = 500
        self.densify_until_iter = 15000 
        self.densify_grad_threshold = 0.0002
        self.lambda_stereo_depth_sup = 0.05

        self.multi_view_ncc_weight = 0.15
        self.multi_view_geo_weight = 0.03
        self.multi_view_patch_size = 3
        self.multi_view_sample_num = 25600

        self.stereosetup_interval = 300
        self.stereofrom_iterations = 500
        self.stereo_baseline_percent = 0.03
        self.pesudo_view_pixel_noise_th = 3.0

        self.lambda_splat_feat = 1.5
        self.splat_feature_loss_iter = -1
        self.pesudo_featpgsr_iter = 3000
        self.ncc_mask_ratio = 0.9
        self.featloss_from_iter = 0

        self.scale_loss_weight = 100.0
        self.abs_split_radii2D_threshold = 20
        self.max_abs_split_points = 50_000
        self.max_all_points = 6000_000
        self.opacity_cull_threshold = 0.005
        self.densify_abs_grad_threshold = 0.0008

        # ===== ConfSurf: A1 continuous stereo confidence =====
        # NOTE: these switches are int (0/1) so they can be toggled from the CLI.
        self.use_confidence_weighting = 1      # soft confidence weighting vs binary mask
        self.conf_tau = 1.0                       # LR-residual -> confidence softness (px)
        self.conf_hard_threshold = 5.0            # safety valve: zero conf above this residual

        # ===== ConfSurf: A2 adaptive multi-baseline stereo =====
        self.adaptive_baseline = 1             # fuse multiple baselines by confidence
        self.stereo_baseline_percent_small = 0.02
        self.stereo_baseline_percent_large = 0.05
        self.baseline_rel_tau = 0.02              # cross-baseline agreement tolerance

        # ===== ConfSurf: A3 confidence-aware stereo schedule =====
        self.confidence_aware_schedule = 1     # enable stereo by quality, not fixed iter
        self.stereo_min_warmup_iter = 200         # hard lower bound before stereo kicks in
        self.stereo_quality_target = 0.35         # mean conf needed for full stereo weight
        self.stereo_quality_ema = 0.7             # EMA factor for quality tracking

        # ===== ConfSurf: B normal-guided depth propagation =====
        self.enable_depth_propagation = 1      # fill low-confidence regions
        self.prop_n_iters = 16
        self.prop_conf_high = 0.6                 # anchor threshold
        self.prop_gamma = 0.85                    # per-hop confidence decay
        self.prop_from_iter = 1000

        # ===== SparseSurf Floater pipeline (6 stages) =====
        self.enable_floater = 1                # master switch for the floater pipeline
        # Stage 1 spatial constraint
        self.lambda_space_max = 0.05
        # Stage 2 detection
        self.detect_from_iter = 1000
        self.attr_anom_w = 0.15
        # Stage 3 soft opacity decay
        self.soft_from_iter = 3000
        self.lambda_decay_max = 2.0
        self.min_obs_soft = 3
        self.ema_beta = 0.95
        # Stage 4 hard pruning
        self.enable_hard_prune = 1            # master switch for Stage 4 hard pruning
        self.hard_from_iter = 5000
        self.prune_depth_diff_thresh = 0.3    # abs floor epsilon_abs (m); tau = max(eps, D_s * theta)
        self.prune_rel_floor = 0.01           # min relative threshold eta_min (1%)
        self.prune_mad_k = 2.5                # theta = max(eta_min, median + k * 1.4826 * MAD)
        self.max_prune_ratio = 0.1
        self.prune_interval = 500
        # Stage 5 resampling
        self.resample_enabled = 1
        self.resample_max_points = 30000
        self.resample_conf_thresh = 0.7
        self.inpaint_enabled = 1
        self.inpaint_max_points = 15000
        # Stage 6 color down-weighting
        self.gamma_max = 0.5

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
