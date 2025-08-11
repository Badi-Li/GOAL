import math
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import skimage.morphology

import torch
import torch.nn as nn
import torch.nn.functional as F
import nav.utils.pose as pu
from nav.utils.prepare_pcd import Prepare_data
from einops import rearrange

from habitat import logger
from nav.utils.fmm_planner import FMMPlanner
from nav.utils.planners import PlannerActorSequential, PlannerActorVector
from nav.utils.prepare_models import get_seg_model, get_fm_model
from nav.utils.projector import Final_Projector as Projector
from nav.utils.semantic_mapping import get_semantic_map
from nav.utils.model_pf import RL_Policy
import nav.utils.pts_utils as ptsu 
from nav.utils.location_checker import LocationChecker
from nav.utils.set_location import (
    set_target_loc,
    set_target_loc_base
)
from nav.utils.model_inference import (
    data_precompute,
    cal_res
)
HM3D2MP3D_MAPPING = {
    0: 0,
    1: 6,
    2: 8,
    3: 10,
    4: 13,
    5: 5
}

class GlobalAgent(object):

    def __init__(self, cfg, device):
        self.cfg = cfg
        # Sanity check. The codes are designed to work only with single thread
        assert cfg.NUM_ENVIRONMENTS == 1, "The codes are designed to work only with single thread"
        self.device = device
        self.dataset = cfg.GLOBAL_AGENT.dataset
        # Create segmentation model
        self.sem_seg_model = get_seg_model(cfg.SCENE_SEGMENTATION)
        self.sem_seg_model.to(device)
        self.sem_seg_model.eval()

        self.seg_pred_thr = cfg.SCENE_SEGMENTATION.seg_pred_thr
        # Flow Matching model
        self.fm_model = get_fm_model(cfg.FM)
        self.fm_model.to(device)
        self.fm_model.eval()
        # Create semantic mapping model
        self.projector = Projector.from_config(cfg, self.device)
        # Preprocessing point clouds 
        self.preparer = Prepare_data()
        
        # PONI, only area potential is used for frontier based exploration 
        self.g_policy = RL_Policy(cfg.PF_EXP_POLICY, cfg.PF_EXP_POLICY.pf_model_path)
        self.g_policy.to(device)
        self.g_policy.eval()

        self.needs_dist_maps = cfg.PF_EXP_POLICY.add_agent2loc_distance or \
                                    cfg.PF_EXP_POLICY.add_agent2loc_distance_v2
        # Efficient mapping
        self.seg_interval = cfg.GLOBAL_AGENT.seg_interval
        self.step_test = cfg.GLOBAL_AGENT.step_test
        self.num_conseq_fwds = 0

        self.stop_upon_replan = cfg.GLOBAL_AGENT.stop_upon_replan
        if self.stop_upon_replan:
            assert cfg.PLANNER.n_planners == 1
            self.replan_count = 0

        self.kornia = None

        if cfg.PLANNER.n_planners > 1:
            self.planners = PlannerActorVector(cfg.PLANNER)
        else:
            self.planners = PlannerActorSequential(cfg.PLANNER)
        # Useful pre-computation
        gcfg = self.cfg.GLOBAL_AGENT
        full_size = gcfg.map_size_cm // gcfg.map_resolution
        local_size = int(full_size / gcfg.global_downscaling)
        self._full_map_size = (full_size, full_size)
        self._local_map_size = (local_size, local_size)
        max_h = int(360 / gcfg.map_resolution)
        min_h = int(-40 / gcfg.map_resolution)
        self.height = max_h - min_h 
        self.min_z = int(25 / gcfg.map_resolution  - min_h)
        self.max_z = int((gcfg.camera_height * 100. + 1) / gcfg.map_resolution  - min_h)
        self.color_palette = None
        self.time_benchmarks = defaultdict(lambda: deque(maxlen=50))
        self.selem = skimage.morphology.disk(3)

    def act(self, batched_obs, agent_states, steps, g_masks, l_masks, infos):
        gcfg = self.cfg.GLOBAL_AGENT
        l_step = steps % gcfg.num_local_steps

        full_map = agent_states["full_map"]
        full_pose = agent_states["full_pose"]
        lmb = agent_states["lmb"]
        local_map = agent_states["local_map"]
        local_pose = agent_states["local_pose"]
        semantic_map = agent_states['semantic_map']
        planner_pose_inputs = agent_states["planner_pose_inputs"]
        origins = agent_states["origins"]
        wait_env = agent_states["wait_env"]
        finished = agent_states["finished"]
        global_orientation = agent_states["global_orientation"]
        global_input = agent_states["global_input"]
        global_goals = agent_states["global_goals"]
        extras = agent_states["extras"]
    
        local_coords = agent_states['local_coords']
        local_feats = agent_states['local_feats']
        global_coords = agent_states['global_coords']
        global_feats = agent_states['global_feats']
        
        local_goal_maps = agent_states['local_goal_maps']
        global_goal_maps = agent_states['global_goal_maps']
        local_invalid_goal = agent_states['local_invalid_goal']
        global_invalid_goal = agent_states['global_invalid_goal']
        location_checker = agent_states['location_checker']
        goal_steps =agent_states['goal_steps']
        local_w, local_h = self.local_map_size
        full_w, full_h = self.full_map_size

        if infos is not None:
            self.past_observations.append(batched_obs)
        ########################################################################
        # Semantic Mapping
        ########################################################################
        poses = self._get_poses_from_obs(batched_obs, agent_states, g_masks)
        state = self.preprocess_obs(batched_obs)

        local_coords, local_feats, local_map, local_pose = self.projector(
            state, poses, local_coords, local_feats, local_pose, local_map
            )

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.0)  # Resetting current location channel
        for e in range(self.cfg.NUM_ENVIRONMENTS):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / gcfg.map_resolution),
                int(c * 100.0 / gcfg.map_resolution),
            ]
            local_map[e, 2:4, loc_r - 2 : loc_r + 3, loc_c - 2 : loc_c + 3] = 1.0

           
        ########################################################################
        # Global policy
        ########################################################################
        if (steps == 0) or (l_step == gcfg.num_local_steps - 1):
            start_time = time.time()
            local_coords, indices = ptsu.unique(local_coords, dim = 0)
            local_feats = local_feats[indices]
            global_coords, indices = ptsu.unique(global_coords, dim = 0)
            global_feats = global_feats[indices]
            updated_local_coords = []
            updated_local_feats = []

            # For every global step, update the full and local maps
            for e in range(self.cfg.NUM_ENVIRONMENTS):
                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.0

                full_map[
                    e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                ] = local_map[e]

                global_goal_maps[
                    e, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                ] = local_goal_maps[e]

                global_invalid_goal[
                    e, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                ] = local_invalid_goal[e]

                batch_mask = (local_coords[:, 0] == e)

                # Before merging to global points, coordinates of the origin should be added
                batch_coords = local_coords[batch_mask]
                batch_feats = local_feats[batch_mask]
                batch_coords[:, 1] += lmb[e, 0] 
                batch_coords[:, 2] += lmb[e, 2]
            


                global_coords = torch.cat([global_coords, batch_coords], dim=0)
                global_feats = torch.cat([global_feats, batch_feats], dim=0)


                full_pose[e] = local_pose[e] + \
                        torch.from_numpy(origins[e]).to(self.device).float()

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [
                    int(r * 100.0 / gcfg.map_resolution),
                    int(c * 100.0 / gcfg.map_resolution),
                ]

                location_checker[e].insert((r, c))


                if not self.cfg.GLOBAL_AGENT.smart_local_boundaries:
                    update_local_boundaries = True
                else:
                    local_r, local_c = local_pose[e, 1].item(), local_pose[e, 0].item()
                    local_loc_r, local_loc_c = [
                        int(local_r * 100.0 / gcfg.map_resolution),
                        int(local_c * 100.0 / gcfg.map_resolution),
                    ]
                    if local_loc_r < (local_w * 0.2) or local_loc_r > (local_w * 0.8):
                        update_local_boundaries = True
                    elif local_loc_c < (local_h * 0.2) or local_loc_c > (local_h * 0.8):
                        update_local_boundaries = True
                    else:
                        update_local_boundaries = False

                if update_local_boundaries:
                    lmb[e] = self.get_local_map_boundaries(
                        (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
                    )

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [
                    lmb[e][2] * gcfg.map_resolution / 100.0,
                    lmb[e][0] * gcfg.map_resolution / 100.0,
                    0.0,
                ]

                local_map[e] = full_map[
                    e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                ]

                local_goal_maps[e] = global_goal_maps[
                    e, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                ]

                local_invalid_goal[e] = global_invalid_goal[
                    e, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                ]

                updated_local_coord, updated_local_feat, global_coords, global_feats, _ = \
                                    ptsu.get_local_points(global_coords, global_feats, e, lmb[e, 0], lmb[e, 1], lmb[e, 2], lmb[e, 3])
                updated_local_coord[:, 1] -= lmb[e, 0]
                updated_local_coord[:, 2] -= lmb[e, 2]

                updated_local_coords.append(updated_local_coord)
                updated_local_feats.append(updated_local_feat) 

                local_pose[e] = full_pose[e] - \
                        torch.from_numpy(origins[e]).to(self.device).float()
            
            local_coords = torch.cat(updated_local_coords, dim = 0)
            local_feats = torch.cat(updated_local_feats, dim = 0)
            self.time_benchmarks["map_update"].append(time.time() - start_time)
            
            change_goals = [0 for _ in range(self.cfg.NUM_ENVIRONMENTS)]
            for e in range(self.cfg.NUM_ENVIRONMENTS):
                if local_goal_maps[e].sum() == 0:
                    change_goals[e] = 1
                    goal_steps[e] = 0
                elif location_checker[e].deadlock():
                    # Current goal result in deadlock, mask out.
                    selem = skimage.morphology.disk(15)
                    local_invalid_goal[e] += skimage.morphology.binary_dilation(local_goal_maps[e].astype(bool),
                                                                                selem)
                    local_invalid_goal[e] = np.clip(local_invalid_goal[e], a_min = 0, a_max = 1)
                    global_invalid_goal[
                        e, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                    ] = local_invalid_goal[e]
                    location_checker[e].reset()
                    change_goals[e] = 1
                    goal_steps[e] = 0
                else:
                    map_pred = local_map[e, 0, :, :].cpu().numpy() > 0
                    goal_map = local_goal_maps[e]
                    pose = planner_pose_inputs[e]
                    dist_to_goal = self._get_fmm_dist_to_goal(map_pred, goal_map, pose)
                    if (dist_to_goal <= self.cfg.PLANNER.change_goal_thr_down) or (dist_to_goal >= self.cfg.PLANNER.change_goal_thr_up and \
                        goal_steps[e] >= gcfg.changegoal_steps):
                        change_goals[e] = 1
                        goal_steps[e] = 0


            # Perform scene segmentation 
            start_time = time.time()
            if goal_steps[0] % self.seg_interval == 0:
                # NOTE: Pad a little to avoid error
                x = self.preparer(local_coords.detach().clone(), local_feats.detach().clone(), 
                                    self.cfg.NUM_ENVIRONMENTS, device = self.device,
                                    spatial_shape = (local_w + 5, local_h + 5, self.height + 5))
                start_time = time.time()
                if x == 0:
                    print('empty points!')
                    sem_pred =torch.zeros((0, ))
                    self.time_benchmarks["semantic_prediction"].append(time.time() - start_time)
                    semantic_map = torch.zeros((self.cfg.NUM_ENVIRONMENTS, gcfg.num_sem_categories,
                                                * self._local_map_size))

                else:
                    with torch.no_grad():
                        seg_output = self.sem_seg_model(x)
                    sem_pred = self.process_segment(seg_output)
                    self.time_benchmarks["semantic_prediction"].append(time.time() - start_time)
                
                
                    # NOTE:Really need stairs?
                    semantic_map = get_semantic_map(local_coords.detach().clone(), sem_pred.detach().clone(), 
                                                    seg_output.shape[1], self._local_map_size, 
                                                    self.cfg.NUM_ENVIRONMENTS, self.min_z, self.max_z, self.cfg.PROJECTOR.cat_pred_threshold)
                    torch.cuda.empty_cache()
            
            self.time_benchmarks["semantic_mapping"].append(time.time() - start_time)

            locs = local_pose.cpu().numpy()
            for e in range(self.cfg.NUM_ENVIRONMENTS):
                global_orientation[e] = int((locs[e, 2] + 180.0) / 5.0)

            global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()
            global_input[:, 4:8, :, :] = nn.MaxPool2d(gcfg.global_downscaling)(
                full_map[:, 0:4, :, :]
            )
            global_input[:, 8:, :, :] = semantic_map.detach()
            goal_cat_id = batched_obs["objectgoal"]
            extras[:, 0] = global_orientation[:, 0]
            extras[:, 1] = goal_cat_id[:, 0]

            ####################################################################
            start_time = time.time()
            # Compute additional inputs if needed
            extra_maps = {
                "dmap": None,
                "umap": None,
                "fmap": None,
                "pfs": None,
                "agent_locations": None,
                "ego_agent_poses": None,
            }

            extra_maps["agent_locations"] = []
            for e in range(self.cfg.NUM_ENVIRONMENTS):
                pose_pred = planner_pose_inputs[e]
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
                gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                map_r, map_c = start_y, start_x
                map_loc = [
                    int(map_r * 100.0 / gcfg.map_resolution - gx1),
                    int(map_c * 100.0 / gcfg.map_resolution - gy1),
                ]
                map_loc = pu.threshold_poses(map_loc, global_input[e].shape[1:])
                extra_maps["agent_locations"].append(map_loc)

            if self.needs_dist_maps:
                planner_inputs = [{} for e in range(self.cfg.NUM_ENVIRONMENTS)]
                for e, p_input in enumerate(planner_inputs):
                    obs_map = local_map[e, 0, :, :].cpu().numpy()
                    exp_map = local_map[e, 1, :, :].cpu().numpy()
                    # set unexplored to navigable by default
                    p_input["map_pred"] = (obs_map * np.rint(exp_map)) > 0
                    p_input["pose_pred"] = planner_pose_inputs[e]
                masks = [1 for _ in range(self.cfg.NUM_ENVIRONMENTS)]
                _, dmap = self.planners.get_reachability_maps(planner_inputs, masks)
                dmap = torch.from_numpy(dmap).to(self.device)
                # Convert to float
                dmap = dmap.float().div_(100.0)  # cm -> m
                extra_maps["dmap"] = dmap

            extra_maps["umap"] = 1.0 - local_map[:, 1, :, :]


            if self.g_policy.needs_egocentric_transform:
                ego_agent_poses = []
                for e in range(self.cfg.NUM_ENVIRONMENTS):
                    map_loc = extra_maps["agent_locations"][e]
                    # Crop map about a center
                    ego_agent_poses.append(
                        [map_loc[0], map_loc[1], math.radians(start_o)]
                    )
                ego_agent_poses = torch.Tensor(ego_agent_poses).to(self.device)
                extra_maps["ego_agent_poses"] = ego_agent_poses

            self.time_benchmarks["extra_inputs"].append(time.time() - start_time)
            ####################################################################
            start_time = time.time()
            # Sample long-term goal from global policy
            _, g_action, _, _, prev_pfs, _, area_pfs = self.g_policy.act(
                global_input,
                None,
                g_masks,
                extras=extras.long(),
                deterministic=False,
                extra_maps=extra_maps,
            )
            if not self.g_policy.has_action_output:
                cpu_actions = g_action.cpu().numpy()
                if len(cpu_actions.shape) == 4:
                    # Output action map
                    global_goals = cpu_actions[:, 0]  # (B, H, W)
                elif len(cpu_actions.shape) == 3:
                    # Output action map
                    global_goals = cpu_actions  # (B, H, W)
                else:
                    # Output action locations
                    assert len(cpu_actions.shape) == 2
                    global_goals = [
                        [int(action[0] * local_w), int(action[1] * local_h)]
                        for action in cpu_actions
                    ]
                    global_goals = [
                        [min(x, int(local_w - 1)), min(y, int(local_h - 1))]
                        for x, y in global_goals
                    ]
            g_masks.fill_(1.0)
            self.time_benchmarks["goal_sampling"].append(time.time() - start_time)

        ########################################################################
        # Define long-term goal map
        ########################################################################
        start_time = time.time()
        found_goal = [0 for _ in range(self.cfg.NUM_ENVIRONMENTS)]
        if self.cfg.GLOBAL_AGENT.dataset == 'mp3d':
            cn = int(batched_obs["objectgoal"][e, 0].item()) 
        elif self.cfg.GLOBAL_AGENT.dataset == 'hm3d':
            cn = int(HM3D2MP3D_MAPPING[batched_obs["objectgoal"][e, 0].item()]) 
        elif self.cfg.GLOBAL_AGENT.dataset == 'gibson':
            pass
        for e in range(self.cfg.NUM_ENVIRONMENTS):
            cat_semantic_map = semantic_map[e, cn, :, :]
            if cat_semantic_map.sum() != 0.0:
                cat_semantic_map = cat_semantic_map.cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.0
                local_goal_maps[e] = cat_semantic_scores
                found_goal[e] = 1
        # NOTE: fix to 0 as we hypothesize single scene
        if not found_goal[0]:
            if not self.g_policy.has_action_output:
                for e in range(self.cfg.NUM_ENVIRONMENTS):
                    if change_goals[e] == 1:
                        local_goal_maps[e] = np.zeros((local_w, local_h))
                        
                        in_semmap = torch.zeros(self.cfg.GLOBAL_AGENT.num_sem_categories + 1, local_w, local_h).to(self.device)
                        in_semmap[0, :, :] = local_map[e, 1, :, :].detach().clone()
                        in_semmap[1, :, :] = local_map[e, 0, :, :].detach().clone()
                        in_semmap[2:, :, :] = semantic_map[e][:-1, :, :].detach().clone()
                        # TODO: Really need this precompute? 
                        _, locs = data_precompute(in_semmap, 0)
                        resmap, flag_c = cal_res(in_semmap, self.fm_model, locs[2], 
                                                        self.cfg.FLOW)
                        if flag_c == 0:
                            good_target_loc = set_target_loc(resmap, area_pfs[e], cn, self.cfg.FM.thr,
                                                            local_invalid_goal[e])
                        else:
                            good_target_loc = set_target_loc_base(area_pfs[e], local_invalid_goal[e])
                        sg_c = int(good_target_loc[1])
                        sg_r = int(good_target_loc[0])
                        local_goal_maps[e][sg_r, sg_c] = 1

            global_goal_maps[
                e, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
            ] = local_goal_maps[e]


        self.time_benchmarks["goal_map_building"].append(time.time() - start_time)
        ########################################################################
        # Plan and sample action
        ########################################################################
        start_time = time.time()
        planner_inputs = [{} for e in range(self.cfg.NUM_ENVIRONMENTS)]
        pf_visualizations = None
        if self.cfg.GLOBAL_AGENT.visualize:
            pf_visualizations = self.g_policy.visualizations
        for e, p_input in enumerate(planner_inputs):
            p_input["map_pred"] = local_map[e, 0, :, :].cpu().numpy() > 0
            p_input["pose_pred"] = planner_pose_inputs[e]
            p_input["goal"] = local_goal_maps[e] > 0  # global_goals[e]
            p_input["new_goal"] = l_step == gcfg.num_local_steps - 1
            p_input["found_goal"] = found_goal[e]
            if self.g_policy.has_action_output:
                p_input["wait"] = (not found_goal[e]) or wait_env[e] or finished[e]
            else:
                p_input["wait"] = wait_env[e] or finished[e]
            if self.cfg.GLOBAL_AGENT.visualize:
                p_input["exp_pred"] = local_map[e, 1, :, :].cpu().numpy() > 0
                # local_map[e, -1, :, :] = 1e-5
                p_input["sem_map_pred"] = semantic_map[e].cpu().numpy()
                p_input["pf_pred"] = pf_visualizations[e]

        actions, replan_flags = self.planners.plan_and_act(
            planner_inputs, l_masks.cpu().numpy()
        )  # (B, 1) ndarray
        goal_steps += 1
        actions = torch.from_numpy(actions)
        if self.g_policy.has_action_output:
            for e in range(self.cfg.NUM_ENVIRONMENTS):
                if not found_goal[e]:
                    actions[e] = g_action[e]
        for e, flag in enumerate(replan_flags):
            if flag and self.cfg.GLOBAL_AGENT.reset_map_upon_replan:
                # print(f'=====> Reseting map for process {e}')
                local_map[e].fill_(0)
            if flag and self.cfg.GLOBAL_AGENT.stop_upon_replan:
                self.replan_count += 1
                if self.replan_count >= self.cfg.GLOBAL_AGENT.stop_upon_replan_thresh:
                    print("========> Early stopping after 2 replans")
                    self.replan_count = 0
                    actions[e] = self.cfg.PLANNER.ACTION.stop
        # Cache for visualization purposes
        self._cached_planner_inputs = planner_inputs

        self.time_benchmarks["plan_and_act"].append(time.time() - start_time)
        if self.seg_interval > 0:
            if actions[0, 0].item() == self.cfg.PLANNER.ACTION.move_forward:
                self.num_conseq_fwds += 1
                self.num_conseq_fwds = self.num_conseq_fwds % self.seg_interval
            else:
                self.num_conseq_fwds = 0
            
            
        ########################################################################
        # Update agent states
        ########################################################################
        agent_states["full_map"] = full_map
        agent_states["full_pose"] = full_pose
        agent_states["lmb"] = lmb
        agent_states["local_map"] = local_map
        agent_states["local_pose"] = local_pose
        agent_states["semantic_map"] = semantic_map 

        agent_states['local_coords'] = local_coords
        agent_states['local_feats'] = local_feats
        agent_states['global_coords'] = global_coords 
        agent_states['global_feats'] = global_feats
        
        agent_states["planner_pose_inputs"] = planner_pose_inputs
        agent_states["origins"] = origins
        agent_states["wait_env"] = wait_env
        agent_states["finished"] = finished
        agent_states["global_orientation"] = global_orientation
        agent_states["global_input"] = global_input
        agent_states["global_goals"] = global_goals
        agent_states["extras"] = extras

        agent_states["local_goal_maps"] = local_goal_maps
        agent_states["global_goal_maps"] = global_goal_maps
        agent_states['location_checker'] = location_checker 
        agent_states['local_invalid_goal'] = local_invalid_goal
        agent_states['global_invalid_goal'] = global_invalid_goal 
        
        agent_states['goal_steps'] = goal_steps
        

        if steps % 50 == 0:
            logger.info("=====> Time benchmarks")
            for k, v in self.time_benchmarks.items():
                logger.info(f"{k:<20s} : {np.mean(v).item():6.4f} secs")

        return actions, agent_states

    def preprocess_obs(self, batched_obs):
        cfg = self.cfg.GLOBAL_AGENT
        rgb = batched_obs["rgb"]  # (B, H, W, 3) torch Tensor
        depth = batched_obs["depth"]  # (B, H, W, 1) torch Tensor
        # Process depth
        depth = self.preprocess_depth(depth)
        # Re-format observations
        rgb = rearrange(rgb, "b h w c -> b c h w").float()
        depth = rearrange(depth, "b h w c -> b c h w").float()

        # Downscale observations
        ds = cfg.env_frame_width // cfg.frame_width
        if ds != 1:
            rgb = F.interpolate(
                rgb,
                (cfg.frame_height, cfg.frame_width),
                mode="nearest",
            )
            depth = depth[:, :, (ds // 2) :: ds, (ds // 2) :: ds]

        state = torch.cat([rgb, depth], dim=1)

        return state

    def preprocess_depth(self, depth):
        # depth - (B, H, W, 1) torch Tensor
        task_cfg = self.cfg.TASK_CONFIG
        min_depth = task_cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = task_cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH

        depth = depth * 1.0

        mask1 = depth>0.99 
        mask2 = depth==0

        depth =  depth * (max_depth - min_depth) * 100 + min_depth * 100

        depth[mask1] = 0
        depth[mask2] = 0

        return depth

    def close(self):
        self.planners.close()

    def process_segment(self, seg_output):

        nc = self.cfg.GLOBAL_AGENT.num_sem_categories
        predictions = F.softmax(seg_output, dim = 1)
        is_confident = torch.any(
            predictions >= self.seg_pred_thr, dim = 1
            )
        pred = torch.argmax(predictions, dim = 1)

        pred[~is_confident] = nc - 1


        return pred
    
    def _get_fmm_dist_to_goal(self, map_pred, goal_map, pose):
        """
        Function responsible for computing the FMM distance from current 
        position to the goal position 
        """
        map_resolution = self.cfg.PLANNER.map_resolution 
        map_pred = np.rint(map_pred).astype(np.uint8)
        goal_map = goal_map

        # Get agent position + local boundaries
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        r, c = start_y, start_x
        start = [
            int(r * 100.0 / map_resolution - gx1),
            int(c * 100.0 / map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, map_pred.shape)


        _, fmm_dist = self._get_reachability(
            map_pred, goal_map, planning_window
        )

        return fmm_dist[start[0], start[1]]
    
    def _get_reachability(self, grid, goal, planning_window):
        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape

        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)


        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(5)
        goal = cv2.dilate(goal, selem)
        planner.set_multi_goal(goal)
        fmm_dist = planner.fmm_dist

        reachability = fmm_dist < fmm_dist.max()

        return reachability.astype(np.float32), fmm_dist.astype(np.float32)

    @property
    def full_map_size(self):
        return self._full_map_size

    @property
    def local_map_size(self):
        return self._local_map_size

    @property
    def cached_planner_inputs(self):
        return self._cached_planner_inputs

    def _get_poses_from_obs(self, batched_obs, agent_states, g_masks):
        curr_sim_location = torch.stack(
            [
                batched_obs["gps"][:, 0],  # -Z
                -batched_obs["gps"][:, 1],  # -X
                batched_obs["compass"][:, 0],  # Heading
            ],
            dim=1,
        )
        prev_sim_location = torch.from_numpy(agent_states["prev_sim_location"]).to(
            curr_sim_location.device
        )
        # Measure pose change
        pose = self._get_rel_pose_change(prev_sim_location, curr_sim_location)
        # If episode terminated in last step, set pose change to zero
        pose = pose * g_masks
        # Update prev_sim_location
        agent_states["prev_sim_location"] = curr_sim_location.cpu().numpy()
        return pose

    def _get_rel_pose_change(self, pos1, pos2):
        x1, y1, o1 = torch.unbind(pos1, dim=1)
        x2, y2, o2 = torch.unbind(pos2, dim=1)

        theta = torch.atan2(y2 - y1, x2 - x1) - o1
        dist = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        dx = dist * torch.cos(theta)
        dy = dist * torch.sin(theta)
        do = o2 - o1

        return torch.stack([dx, dy, do], dim=1)

    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.cfg.GLOBAL_AGENT.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose(self, agent_states):

        gcfg = self.cfg.GLOBAL_AGENT

        local_w, local_h = self.local_map_size
        full_w, full_h = self.full_map_size

        full_map = agent_states["full_map"]
        full_pose = agent_states["full_pose"]
        lmb = agent_states["lmb"]
        origins = agent_states["origins"]
        local_map = agent_states["local_map"]
        local_pose = agent_states["local_pose"]
        planner_pose_inputs = agent_states["planner_pose_inputs"]
        semantic_map = agent_states['semantic_map']

        goal_steps = agent_states['goal_steps']
        local_goal_maps = agent_states['local_goal_maps']
        global_goal_maps = agent_states['global_goal_maps']
        local_invalid_goal = agent_states['local_invalid_goal']
        global_invalid_goal = agent_states['global_invalid_goal']
        location_checker = agent_states['location_checker']

        global_coords = torch.empty((0, 4)).to(self.device)
        global_feats = torch.empty((0, 6)).to(self.device)
        local_coords = torch.empty((0, 4)).to(self.device)
        local_feats = torch.empty((0, 6)).to(self.device)

        self.past_observations = []

        goal_steps.fill(0)
        full_map.fill_(0.0)
        full_pose.fill_(0.0)
        full_pose[:, :2] = gcfg.map_size_cm / 100.0 / 2.0
        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs

        semantic_map.fill_(0.0)

        # numpy array, fill is in-place operation
        local_goal_maps.fill(0)
        global_goal_maps.fill(0)
        local_invalid_goal.fill(0)
        global_invalid_goal.fill(0)


        for e in range(self.cfg.NUM_ENVIRONMENTS):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / gcfg.map_resolution),
                int(c * 100.0 / gcfg.map_resolution),
            ]

            full_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

            lmb[e] = self.get_local_map_boundaries(
                (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
            )

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [
                lmb[e][2] * gcfg.map_resolution / 100.0,
                lmb[e][0] * gcfg.map_resolution / 100.0,
                0.0,
            ]

            location_checker[e].reset()

        for e in range(self.cfg.NUM_ENVIRONMENTS):
            local_map[e] = full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]]
            local_pose[e] = (
                full_pose[e] - torch.from_numpy(origins[e]).to(self.device).float()
            )

        # Update states (probably unnecessary)
        agent_states["full_map"] = full_map
        agent_states["full_pose"] = full_pose
        agent_states["lmb"] = lmb
        agent_states["origins"] = origins
        agent_states["local_map"] = local_map
        agent_states["local_pose"] = local_pose
        agent_states["planner_pose_inputs"] = planner_pose_inputs

        agent_states['local_goal_maps'] = local_goal_maps
        agent_states['global_goal_maps'] = global_goal_maps
        agent_states['local_invalid_goal'] = local_invalid_goal
        agent_states['global_invalid_goal'] = global_invalid_goal
        agent_states['location_checker'] = location_checker

        agent_states['goal_steps'] = goal_steps
        agent_states['semantic_map'] = semantic_map 
        agent_states['local_coords'] = local_coords 
        agent_states['local_feats'] = local_feats 
        agent_states['global_coords'] = global_coords 
        agent_states['global_feats'] = global_feats 


        return agent_states

    def init_map_and_pose_for_env(self, agent_states, e):

        gcfg = self.cfg.GLOBAL_AGENT

        local_w, local_h = self.local_map_size
        full_w, full_h = self.full_map_size

        full_map = agent_states["full_map"]
        full_pose = agent_states["full_pose"]
        lmb = agent_states["lmb"]
        origins = agent_states["origins"]
        local_map = agent_states["local_map"]
        local_pose = agent_states["local_pose"]
        planner_pose_inputs = agent_states["planner_pose_inputs"]

        local_goal_maps = agent_states['local_goal_maps']
        global_goal_maps = agent_states['global_goal_maps']
        local_invalid_goal = agent_states['local_invalid_goal']
        global_invalid_goal = agent_states['global_invalid_goal']
        location_checker = agent_states['location_checker']
        semantic_map = agent_states['semantic_map']
        global_coords = agent_states["global_coords"]
        global_feats = agent_states["global_feats"]
        local_coords = agent_states["local_coords"]
        local_feats = agent_states["local_feats"]
        goal_steps = agent_states['goal_steps']

        self.past_observations = []

        goal_steps[e] = 0


        full_map[e].fill_(0.0)
        full_pose[e].fill_(0.0)
        full_pose[e, :2] = gcfg.map_size_cm / 100.0 / 2.0

        local_goal_maps[e].fill(0)
        global_goal_maps[e].fill(0)
        local_invalid_goal[e].fill(0)
        global_invalid_goal[e].fill(0)
        location_checker[e].reset()

        semantic_map[e].fill_(0.0)

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [
            int(r * 100.0 / gcfg.map_resolution),
            int(c * 100.0 / gcfg.map_resolution),
        ]

        full_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

        lmb[e] = self.get_local_map_boundaries(
            (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
        )

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [
            lmb[e][2] * gcfg.map_resolution / 100.0,
            lmb[e][0] * gcfg.map_resolution / 100.0,
            0.0,
        ]

        local_map[e] = full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]]
        local_pose[e] = (
            full_pose[e] - torch.from_numpy(origins[e]).to(self.device).float()
        )

        batch_mask = global_coords[:, 0] == e 
        global_coords = global_coords[~batch_mask]
        global_feats = global_feats[~batch_mask]

        batch_mask = local_coords[:, 0] == e 
        local_coords = local_coords[~batch_mask]
        local_feats = local_feats[~batch_mask]


        # Update states (probably unnecessary)
        agent_states["full_map"] = full_map
        agent_states["full_pose"] = full_pose
        agent_states["lmb"] = lmb
        agent_states["origins"] = origins
        agent_states["local_map"] = local_map
        agent_states["local_pose"] = local_pose
        agent_states["planner_pose_inputs"] = planner_pose_inputs

        agent_states['semantic_map'] = semantic_map 
        agent_states['local_coords'] = local_coords 
        agent_states['local_feats'] = local_feats 
        agent_states['global_coords'] = global_coords 
        agent_states['global_feats'] = global_feats 

        agent_states['local_goal_maps'] = local_goal_maps
        agent_states['global_goal_maps'] = global_goal_maps
        agent_states['local_invalid_goal'] = local_invalid_goal
        agent_states['global_invalid_goal'] = global_invalid_goal
        agent_states['location_checker'] = location_checker

        agent_states['goal_steps'] = goal_steps

        return agent_states

    def get_new_agent_states(self):
        ########################################################################
        # Create agent states:
        ########################################################################
        # Full map consists of multiple channels containing the following:
        # 1. Obstacle Map
        # 2. Exploread Area
        # 3. Current Agent Location
        # 4. Past Agent Locations
        # 5,6,7,.. : Semantic Categories
        nc = self.cfg.GLOBAL_AGENT.num_sem_categories # num channels
        full_w, full_h = self.full_map_size
        local_w, local_h = self.local_map_size

        B = self.cfg.NUM_ENVIRONMENTS

        full_map = torch.zeros(B, 4, full_w, full_h).to(self.device)
        local_map = torch.zeros(B, 4, local_w, local_h).to(self.device)

        global_goal_maps = np.zeros((B, full_w, full_h))
        local_goal_maps = np.zeros((B, local_w, local_h))
        global_invalid_goal = np.zeros((B, full_w, full_h))
        local_invalid_goal = np.zeros((B, local_w, local_h))
        location_checker = [LocationChecker(self.cfg.GLOBAL_AGENT.checker_length,
                                            self.cfg.GLOBAL_AGENT.deadlock_thre) for _ in range(B)]

        semantic_map = torch.zeros(B, nc, local_w, local_h).to(self.device)

        goal_steps = np.zeros((B, ), dtype = np.int32)
        global_coords = torch.empty((0, 4)).to(self.device)
        global_feats = torch.empty((0, 6)).to(self.device)
        local_coords = torch.empty((0, 4)).to(self.device)
        local_feats = torch.empty((0, 6)).to(self.device)

        # Create pose estimates
        full_pose = torch.zeros(B, 3).to(self.device)
        local_pose = torch.zeros(B, 3).to(self.device)

        # Origins of local map
        origins = np.zeros((B, 3))

        # Local map boundaries
        lmb = np.zeros((B, 4), dtype=int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        planner_pose_inputs = np.zeros((B, 7))

        # Global policy states
        ngc = 8 + self.cfg.GLOBAL_AGENT.num_sem_categories
        es = 2
        global_input = torch.zeros(B, ngc, local_w, local_h).to(self.device)
        global_orientation = torch.zeros(B, 1).long().to(self.device)
        extras = torch.zeros(B, es).to(self.device)
        prev_sim_location = np.zeros((B, 3), dtype=np.float32)

        # Other states
        wait_env = np.zeros((B,))
        finished = np.zeros((B,))

        agent_states = {
            "full_map": full_map,
            "local_map": local_map,
            "full_pose": full_pose,
            "local_pose": local_pose,
            "local_coords": local_coords,
            "local_feats": local_feats, 
            "global_coords": global_coords, 
            "global_feats": global_feats,
            "semantic_map": semantic_map, 
            "origins": origins,
            "lmb": lmb,
            "planner_pose_inputs": planner_pose_inputs,
            "global_input": global_input,
            "global_orientation": global_orientation,
            "global_goals": None,
            "extras": extras,
            "prev_sim_location": prev_sim_location,
            "finished": finished,
            "wait_env": wait_env,
            "local_invalid_goal": local_invalid_goal, 
            "global_invalid_goal": global_invalid_goal,
            "local_goal_maps": local_goal_maps,
            "global_goal_maps": global_goal_maps,
            "location_checker": location_checker,
            "goal_steps": goal_steps,
        }

        return agent_states