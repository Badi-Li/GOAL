import warnings
from typing import List, Optional, Union

from habitat.config import Config as CN
from habitat_baselines.config.default import _C, CONFIG_FILE_SEPARATOR

from . import default_hlab

_C = _C.clone()

_C.defrost()

_C.BASE_TASK_CONFIG_PATH = "configs/objectnav_mp3d.yaml"
_C.VERBOSE = False
_C.NUM_ENVIRONMENTS = 1
################################################################################
# Planner
################################################################################
_C.PLANNER = CN()
_C.PLANNER.collision_threshold = 0.20
# --------------------------------------------------------
# NOTE: These will be set automatically. Do NOT modify.
# --------------------------------------------------------
_C.PLANNER.n_planners = 5
_C.PLANNER.map_size_cm = 2400
_C.PLANNER.map_resolution = 5
_C.PLANNER.turn_angle = 30
_C.PLANNER.change_goal_thr_up = 240
_C.PLANNER.change_goal_thr_down = 10
# action mappings
_C.PLANNER.ACTION = CN()
_C.PLANNER.ACTION.stop = 0
_C.PLANNER.ACTION.move_forward = 1
_C.PLANNER.ACTION.turn_left = 2
_C.PLANNER.ACTION.turn_right = 3
# Downsampling factor for speed
_C.PLANNER.stg_downsampling = 1
_C.PLANNER.conseq_replan_thresh = 5
_C.PLANNER.stg_disk_size = 10
_C.PLANNER.move_as_close_as_possible = True
_C.PLANNER.move_close_limit = 25
# Weighted planner configs
_C.PLANNER.enable_weighted = False
_C.PLANNER.weighted_scale = 4.0
_C.PLANNER.weighted_niters = 1
################################################################################
# Scene segmentation
################################################################################
_C.SCENE_SEGMENTATION = CN()
_C.SCENE_SEGMENTATION.sem_pred_weights = './pretrained_models/spconv_state.pth'
_C.SCENE_SEGMENTATION.in_channels = 6
_C.SCENE_SEGMENTATION.out_channels = 22
_C.SCENE_SEGMENTATION.base_channels = 32
_C.SCENE_SEGMENTATION.channels = (32, 64, 128, 256, 256, 128, 96, 96)
_C.SCENE_SEGMENTATION.layers = (2, 3, 4, 6, 2, 2, 2, 2)
_C.SCENE_SEGMENTATION.D = 3
_C.SCENE_SEGMENTATION.seg_pred_thr = 0.7
# --------------------------------------------------------
# NOTE: These will be set automatically. Do NOT modify.
# --------------------------------------------------------
################################################################################
# Semantic Mapping
################################################################################
_C.PROJECTOR = CN()
_C.PROJECTOR.vision_range = 100
_C.PROJECTOR.du_scale = 1
_C.PROJECTOR.cat_pred_threshold = 1.0
_C.PROJECTOR.map_pred_threshold = 1.0
_C.PROJECTOR.exp_pred_threshold = 1.0
################################################################################
# Flow Mathching Model
################################################################################
_C.FM = CN()
# TODO: Adjust config and weights path
_C.FM.fm_weights = './pretrained_models/fm_model.pth'
_C.FM.input_size = 256 
_C.FM.patch_size = 16 
_C.FM.in_channels = 23 
_C.FM.hidden_size = 768
_C.FM.num_heads = 12 
_C.FM.depth = 12 
_C.FM.condition = None
_C.FM.thr = 0.35
################################################################################
# Flow Solver
################################################################################
_C.FLOW = CN()
_C.FLOW.ODE_method = 'euler'
_C.FLOW.num_steps = 96
_C.FLOW.condition = None
_C.FLOW.expand_ratio = 1.2 
_C.FLOW.noise_std = 0.01
################################################################################
# Global exploration agent
################################################################################
_C.GLOBAL_AGENT = CN()
_C.GLOBAL_AGENT.dataset = "mp3d"
_C.GLOBAL_AGENT.name = "SemExp"
_C.GLOBAL_AGENT.map_size_cm = 4800
_C.GLOBAL_AGENT.map_resolution = 5
_C.GLOBAL_AGENT.global_downscaling = 2
_C.GLOBAL_AGENT.frame_width = 640
_C.GLOBAL_AGENT.frame_height = 480
# TODO: Adjust num_local_steps
_C.GLOBAL_AGENT.num_local_steps = 25
_C.GLOBAL_AGENT.smart_local_boundaries = False
# TODO: Adjust number of categories
_C.GLOBAL_AGENT.num_sem_categories = 22  # Note: This is n_classes + 1
_C.GLOBAL_AGENT.reset_map_upon_replan = True
# TODO: Adjust seg_interval 
_C.GLOBAL_AGENT.seg_interval = 1
_C.GLOBAL_AGENT.stop_upon_replan = False
_C.GLOBAL_AGENT.stop_upon_replan_thresh = 2
_C.GLOBAL_AGENT.step_test = 5
_C.GLOBAL_AGENT.use_fm = True
_C.GLOBAL_AGENT.checker_length = 15
_C.GLOBAL_AGENT.deadlock_thre = 0.5
_C.GLOBAL_AGENT.changegoal_steps = 1
_C.GLOBAL_AGENT.visualize_maps = False
_C.GLOBAL_AGENT.visualize_interval = 25
# --------------------------------------------------------
# NOTE: These will be set automatically. Do NOT modify.
# --------------------------------------------------------
_C.GLOBAL_AGENT.env_frame_width = 640
_C.GLOBAL_AGENT.env_frame_height = 480
_C.GLOBAL_AGENT.camera_height = 0.88
_C.GLOBAL_AGENT.visualize = False
################################################################################
# SemExp policy
################################################################################
_C.SEM_EXP_POLICY = CN()
_C.SEM_EXP_POLICY.global_hidden_size = 256
_C.SEM_EXP_POLICY.use_recurrent_global = False
_C.SEM_EXP_POLICY.main_model = "simple_cnn"
_C.SEM_EXP_POLICY.pretrained_weights = ""
################################################################################
# PFExp policy
################################################################################
_C.PF_EXP_POLICY = CN()
_C.PF_EXP_POLICY.pf_model_path = ""
_C.PF_EXP_POLICY.map_resolution = 5
_C.PF_EXP_POLICY.visualize = 0
_C.PF_EXP_POLICY.print_images = 1
_C.PF_EXP_POLICY.use_egocentric_transform = False
_C.PF_EXP_POLICY.add_agent2loc_distance = False
_C.PF_EXP_POLICY.add_agent2loc_distance_v2 = False
_C.PF_EXP_POLICY.pf_masking_opt = "none"
_C.PF_EXP_POLICY.mask_nearest_locations = True
_C.PF_EXP_POLICY.mask_size = 1.0
_C.PF_EXP_POLICY.area_weight_coef = 0.5
_C.PF_EXP_POLICY.dist_weight_coef = 0.3
################################################################################
# NFExp policy
################################################################################
_C.NF_EXP_POLICY = CN()
_C.NF_EXP_POLICY.mask_nearest_locations = True
_C.NF_EXP_POLICY.mask_size = 1.0
################################################################################


_C.register_renamed_key


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :ref:`config_paths` and overwritten by options from :ref:`opts`.

    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, ``opts = ['FOO.BAR',
        0.5]``. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        for k, v in zip(opts[0::2], opts[1::2]):
            if k == "BASE_TASK_CONFIG_PATH":
                config.BASE_TASK_CONFIG_PATH = v

    config.TASK_CONFIG = default_hlab.get_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    if config.NUM_PROCESSES != -1:
        warnings.warn(
            "NUM_PROCESSES is depricated and will be removed in a future version."
            "  Use NUM_ENVIRONMENTS instead."
            "  Overwriting NUM_ENVIRONMENTS with NUM_PROCESSES for backwards compatibility."
        )

        config.NUM_ENVIRONMENTS = config.NUM_PROCESSES

    config.freeze()
    return config