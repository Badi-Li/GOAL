import nav.utils.poni_geometry as pgeo
import torch
import torch.nn as nn
from einops import asnumpy
from nav.configs.poni_default import get_cfg
from models.poni import get_semantic_encoder_decoder
import torch.nn.functional as F

class Potential_Function_Semantic_Policy(nn.Module):
    def __init__(self, pf_model_path):
        super().__init__()

        loaded_state = torch.load(pf_model_path, map_location="cpu")
        pf_model_cfg = get_cfg()
        pf_model_cfg.merge_from_other_cfg(loaded_state["cfg"])
        self.pf_model = PFModel(pf_model_cfg)
        # Remove dataparallel modules
        state_dict = {
            k.replace(".module", ""): v for k, v in loaded_state["state_dict"].items()
        }
        self.pf_model.load_state_dict(state_dict)
        self.eval()

    def forward(self, inputs, rnn_hxs, masks, extras):
        # inputs - (bs, N, H, W)
        # x_pf - (bs, N, H, W), x_a - (bs, 1, H, W)
        x_pf, x_a = self.pf_model.infer(inputs, avg_preds=False)
        return x_pf, x_a

    def add_agent_dists_to_object_dists(self, pfs, agent_dists):
        # pfs - (B, N, H, W)
        # agent_dists - (B, H, W)
        object_dists = self.convert_object_pf_to_distance(pfs)
        agent2obj_dists = agent_dists.unsqueeze(1) + object_dists
        # Convert back to pf
        return self.pf_model.convert_distance_to_pf(agent2obj_dists)

    def convert_object_pf_to_distance(self, pfs):
        return self.pf_model.convert_object_pf_to_distance(pfs)

    def convert_distance_to_pf(self, dists):
        return self.pf_model.convert_distance_to_pf(dists)

    @property
    def cfg(self):
        return self.pf_model.cfg


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):
    def __init__(self, args, pf_model_path):

        super(RL_Policy, self).__init__()

        self.args = args
        self.network = Potential_Function_Semantic_Policy(pf_model_path)
        self._cached_visualizations = None

        self.prev_maps = None 

    @property
    def is_recurrent(self):
        return False

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return 10  # Some random value

    @property
    def needs_egocentric_transform(self):
        cfg = self.network.pf_model.cfg
        output_type = "map"
        if hasattr(cfg.MODEL, "output_type"):
            output_type = cfg.MODEL.output_type
        return (
            output_type in ["dirs", "locs", "acts"]
        ) or self.args.use_egocentric_transform

    @property
    def has_action_output(self):
        cfg = self.network.pf_model.cfg
        return cfg.MODEL.output_type == "acts"

    def get_pf_cfg(self):
        return self.network.pf_model.get_pf_cfg()

    def forward(self, inputs, rnn_hxs, masks, extras):
        raise NotImplementedError

    def act(
        self, inputs, rnn_hxs, masks, extras=None, extra_maps=None, deterministic=False
    ):

        assert extra_maps is not None
        value = torch.zeros(inputs.shape[0], device=inputs.device)
        action_log_probs = torch.zeros(inputs.shape[0], device=inputs.device)
        # Convert inputs to appropriate format for self.network
        proc_inputs = self.do_proc(inputs)  # (B, N, H, W)

        # Perform egocentric transform if needed
        B, _, H, W = proc_inputs.shape
        t_ego_agent_poses = None
        t_proc_inputs = proc_inputs
        if self.needs_egocentric_transform:
            # Input conventions:
            # X is down, Y is right, origin is top-left
            # theta in radians from Y to X
            ego_agent_poses = extra_maps["ego_agent_poses"]  # (B, 3)
            # Convert to conventions appropriate for spatial_transform_map
            # Required conventions:
            # X is right, Y is down, origin is map center
            # theta in radians from new X to new Y (no changes in effect)
            t_ego_agent_poses = torch.stack(
                [
                    ego_agent_poses[:, 1] - W / 2.0,
                    ego_agent_poses[:, 0] - H / 2.0,
                    ego_agent_poses[:, 2],
                ],
                dim=1,
            )  # (B, 3)
            t_proc_inputs = pgeo.spatial_transform_map(t_proc_inputs, t_ego_agent_poses)

        with torch.no_grad():
            t_pfs, t_area_pfs = self.network(t_proc_inputs, rnn_hxs, masks, extras)

        if self.has_action_output:
            goal_cat_id = extras[:, 1].long()  # (bs, )
            out_actions = [
                t_pfs[e, gcat.item() + 2].argmax().item()
                for e, gcat in enumerate(goal_cat_id)
            ]
            return value, out_actions, action_log_probs, rnn_hxs, {}

        # Transform back the prediction if needed
        pfs = t_pfs
        area_pfs = t_area_pfs
        if self.needs_egocentric_transform:
            # Compute transform from t_ego_agent_poses -> origin
            origin_pose = torch.Tensor([[0.0, 0.0, 0.0]]).to(inputs.device)
            rev_ego_agent_poses = pgeo.subtract_poses(t_ego_agent_poses, origin_pose)
            pfs = pgeo.spatial_transform_map(pfs, rev_ego_agent_poses)  # (B, N, H, W)
            if area_pfs is not None:
                area_pfs = pgeo.spatial_transform_map(
                    area_pfs, rev_ego_agent_poses
                )  # (B, 1, H, W)

        # Add agent to location distance if needed
        if self.args.add_agent2loc_distance:
            agent_dists = extra_maps["dmap"]  # (B, H, W)
            pfs_dists = self.network.convert_object_pf_to_distance(pfs)  # (B, N, H, W)
            pfs_dists = pfs_dists + agent_dists.unsqueeze(1)
            # Convert back to a pf
            pfs = self.network.convert_distance_to_pf(pfs_dists)

        dist_pfs = None
        if self.args.add_agent2loc_distance_v2:
            agent_dists = extra_maps["dmap"].unsqueeze(1)  # (B, 1, H, W)
            dist_pfs = self.network.convert_distance_to_pf(agent_dists)

        # Take the mean with area_pfs
        init_pfs = pfs
        if area_pfs is not None:
            if dist_pfs is None:
                awc = self.args.area_weight_coef
                pfs = (1 - awc) * pfs + awc * area_pfs
            else:
                awc = self.args.area_weight_coef
                dwc = self.args.dist_weight_coef
                assert (awc + dwc <= 1) and (awc + dwc >= 0)
                pfs = (1 - awc - dwc) * pfs + awc * area_pfs + dwc * dist_pfs

        # Get action
        goal_cat_id = extras[:, 1].long()
        action = self.get_action(
            pfs,
            goal_cat_id,
            extra_maps["umap"],
            extra_maps["dmap"],
            extra_maps["agent_locations"],
        )
        pred_maps = {
            "pfs": pfs,
            "raw_pfs": init_pfs,
            "area_pfs": area_pfs,
        }
        pred_maps = {
            k: asnumpy(v) if v is not None else v for k, v in pred_maps.items()
        }

        
        self.prev_maps = pred_maps 
        return value, action, action_log_probs, rnn_hxs, pred_maps, t_pfs, t_area_pfs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        raise NotImplementedError

    def do_proc(self, inputs):
        """
        Map consists of multiple channels containing the following:
        ----------- For local map -----------------
        1. Obstacle Map
        2. Explored Area
        3. Current Agent Location
        4. Past Agent Locations
        ----------- For global map -----------------
        5. Obstacle Map
        6. Explored Area
        7. Current Agent Location
        8. Past Agent Locations
        ----------- For semantic local map -----------------
        9,10,11,.. : Semantic Categories
        """
        # The input to PF model consists of Free map, Obstacle Map, Semantic Categories
        # The last semantic map channel is ignored since it belongs to unknown categories.
        obstacle_map = inputs[:, 0:1]
        explored_map = inputs[:, 1:2]
        semantic_map = inputs[:, 8:-1]
        free_map = ((obstacle_map < 0.5) & (explored_map >= 0.5)).float()
        outputs = torch.cat([free_map, obstacle_map, semantic_map], dim=1)
        return outputs

    def get_action(self, pfs, goal_cat_id, umap, dmap, agent_locs):
        """
        Computes distance from (agent -> location) + (location -> goal)
        based on PF predictions. It then selects goal as location with
        least distance.

        Args:
            pfs = (B, N + 2, H, W) potential fields
            goal_cat_id = (B, ) goal category
            umap = (B, H, W) unexplored map
            dmap = (B, H, W) geodesic distance from agent map
            agent_locs = B x 2 list of agent positions
        """
        B, N, H, W = pfs.shape[0], pfs.shape[1] - 2, pfs.shape[2], pfs.shape[3]
        goal_pfs = []
        for b in range(B):
            goal_pf = pfs[b, goal_cat_id[b].item() + 2, :]
            goal_pfs.append(goal_pf)
        goal_pfs = torch.stack(goal_pfs, dim=0)
        agt2loc_dist = dmap
        if self.args.pf_masking_opt == "unexplored":
            # Filter out explored locations
            goal_pfs = goal_pfs * umap
        # Filter out locations very close to the agent
        if self.args.mask_nearest_locations:
            for i in range(B):
                ri, ci = agent_locs[i]
                size = int(self.args.mask_size * 100.0 / self.args.map_resolution)
                goal_pfs[i, ri - size : ri + size + 1, ci - size : ci + size + 1] = 0

        act_ixs = goal_pfs.view(B, -1).max(dim=1).indices
        # Convert action to (0, 1) values for x and y coors
        actions = []
        for b in range(B):
            act_ix = act_ixs[b].item()
            # Convert action to (0, 1) values for x and y coors
            act_x = float(act_ix % W) / W
            act_y = float(act_ix // W) / H
            actions.append((act_y, act_x))
        actions = torch.Tensor(actions).to(pfs.device)

        return actions

    
class PFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Define loss functions
        # Define models
        self.dirs_map = None
        self.inv_dists_map = None
        enable_directions = self.cfg.DATASET.enable_directions
        enable_locations = self.cfg.DATASET.enable_locations
        enable_actions = self.cfg.DATASET.enable_actions
        assert not (enable_locations and enable_directions)
        assert not (enable_actions and enable_directions)
        assert not (enable_actions and enable_locations)
        (
            self.encoder,
            self.object_decoder,
            self.area_decoder,
        ) = get_semantic_encoder_decoder(self.cfg)
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.OPTIM.lr)
        # Define scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=cfg.OPTIM.lr_sched_milestones,
            gamma=cfg.OPTIM.lr_sched_gamma,
        )
        # Define activation functions
        self.object_activation = get_activation_fn(self.cfg.MODEL.object_activation)
        self.area_activation = get_activation_fn(self.cfg.MODEL.area_activation)

    def forward(self, x):
        embedding = self.encoder(x)
        object_preds = self.object_activation(self.object_decoder(embedding))
        area_preds = None
        if self.area_decoder is not None:
            area_preds = self.area_activation(self.area_decoder(embedding))
        return object_preds, area_preds

    def get_inv_dists_map(self, x):
        # x - (bs, N, M, M)
        M = x.shape[2]
        assert x.shape[3] == M
        Mby2 = M // 2
        # Compute a directions map that stores the direction value in
        # degrees for each location on the map.
        x_values = torch.arange(0, M, 1).float().unsqueeze(0) - Mby2  # (1, M)
        y_values = torch.arange(0, M, 1).float().unsqueeze(1) - Mby2  # (M, 1)
        dirs_map = torch.atan2(y_values, x_values)  # (M, M)
        dirs_map = torch.rad2deg(dirs_map).view(1, 1, M, M)
        dirs_map = (dirs_map + 360) % 360
        # Compute a distances map that stores the distance value in unit
        # cells for each location on the map.
        dists_map = torch.sqrt(x_values**2 + y_values**2)  # (M, M)
        inv_dists_map = torch.exp(-dists_map).view(1, 1, M, M)
        return dirs_map.to(x.device), inv_dists_map.to(x.device)

    def infer(self, x, do_forward_pass=True, input_maps=None, avg_preds=True):
        if do_forward_pass:
            object_preds, area_preds = self(x)
        else:
            assert input_maps is not None
            object_preds, area_preds = x
            x = input_maps

        if self.cfg.MODEL.output_type == "dirs":
            ####################################################################
            # Convert predicted directions to points on a map.
            ####################################################################
            # Convert predictions to angles
            # object_preds - (B, N, D)
            angles = torch.Tensor(self.cfg.DATASET.prediction_directions)
            angles = angles.to(object_preds.device)  # (D, )
            pred_dir_ixs = torch.argmax(object_preds, dim=2)  # (B, N)
            B, N = pred_dir_ixs.shape
            pred_dir_ixs = pred_dir_ixs.view(-1)  # (B * N)
            pred_dirs = torch.gather(angles, 0, pred_dir_ixs)
            pred_dirs = pred_dirs.view(B, N, 1, 1)  # (B, N, 1, 1)
            # Identify frontiers
            frontiers = self.calculate_frontiers(x)  # (B, 1, H, W)
            # Select the nearest frontier point along the predicted direction
            if self.dirs_map is None:
                self.dirs_map, self.inv_dists_map = self.get_inv_dists_map(x)
            dirs_map = self.dirs_map
            inv_dists_map = self.inv_dists_map
            delta = torch.abs(angles[1] - angles[0]).item() / 2
            dirs_map = dirs_map.to(x.device)
            abs_diff = torch.abs(dirs_map - pred_dirs) % 360  # (B, N, H, W)
            diff = (abs_diff - 180) % 360 - 180  # Convert from (0, 360) to (-180, 180)
            is_within_angle = (torch.abs(diff) <= delta).float()
            inv_dists_map = inv_dists_map.to(x.device)  # (1, 1, H, W)
            complex_mask = frontiers * is_within_angle * inv_dists_map  # (B, N, H, W)
            _, _, H, W = complex_mask.shape
            fpoint = torch.argmax(complex_mask.view(B, N, -1), dim=2)  # (B, N)
            # If no frontier point exists, sample a random point
            # along the predicted direction.
            no_frontier_mask = torch.all(
                complex_mask.view(B, N, -1) == 0, dim=2
            )  # (B, N)
            angle_mask = is_within_angle.view(B * N, H * W)
            spoint = torch.multinomial(angle_mask, 1)  # (B * N, 1)
            spoint = spoint.view(B, N)
            fpoint[no_frontier_mask] = spoint[no_frontier_mask]
            # Create a map with these predictions
            preds_map = torch.zeros_like(complex_mask)  # (B, N, H, W)
            preds_map = preds_map.view(B, N, -1)  # (B, N, H * W)
            preds_map.scatter_(2, fpoint.unsqueeze(2), 1.0)
            preds_map = preds_map.view(B, N, H, W)
            object_preds = F.max_pool2d(preds_map, 7, stride=1, padding=3)
        elif self.cfg.MODEL.output_type == "locs":
            ####################################################################
            # Convert predicted locations to points on a map.
            ####################################################################
            # Convert predictions to map locations
            # preds - (B, N, 2)
            B, N, H, W = x.shape
            preds_x = torch.clamp(object_preds[:, :, 0] * W, 0, W - 1)  # (B, N)
            preds_y = torch.clamp(object_preds[:, :, 1] * H, 0, H - 1)  # (B, N)
            # Convert to row-major form
            preds_xy = (preds_y * W + preds_x).long()  # (B, N)
            # Create a map with these predictions
            preds_map = torch.zeros_like(x)  # (B, N, H, W)
            preds_map = preds_map.view(B, N, -1)
            preds_map.scatter_(2, preds_xy.unsqueeze(2), 1.0)
            preds_map = preds_map.view(B, N, H, W)
            object_preds = F.max_pool2d(preds_map, 7, stride=1, padding=3)
        elif self.cfg.MODEL.output_type == "acts":
            ####################################################################
            # Retain predicted actions as actions
            ####################################################################
            # object_preds - (B, N, 4)
            assert not avg_preds
        # By default, average the two predictions and return it.
        if avg_preds:
            outputs = object_preds
            if area_preds is not None:
                outputs = (object_preds + area_preds) / 2.0
            return outputs
        else:
            return object_preds, area_preds

    def calculate_frontiers(self, x):
        # x - semantic map of shape (B, N, H, W)
        free_map = (x[:, 0] >= 0.5).float()  # (B, H, W)
        exp_map = torch.max(x, dim=1).values >= 0.5  # (B, H, W)
        unk_map = (~exp_map).float()  # (B, H, W)
        # Compute frontiers (reference below)
        # https://github.com/facebookresearch/exploring_exploration/blob/09d3f9b8703162fcc0974989e60f8cd5b47d4d39/exploring_exploration/models/frontier_agent.py#L132
        unk_map_shiftup = F.pad(unk_map, (0, 0, 0, 1))[:, 1:]
        unk_map_shiftdown = F.pad(unk_map, (0, 0, 1, 0))[:, :-1]
        unk_map_shiftleft = F.pad(unk_map, (0, 1, 0, 0))[:, :, 1:]
        unk_map_shiftright = F.pad(unk_map, (1, 0, 0, 0))[:, :, :-1]
        frontiers = (
            (free_map == unk_map_shiftup)
            | (free_map == unk_map_shiftdown)
            | (free_map == unk_map_shiftleft)
            | (free_map == unk_map_shiftright)
        ) & (
            free_map == 1
        )  # (B, H, W)
        # Dilate the frontiers
        frontiers = frontiers.unsqueeze(1).float()  # (B, 1, H, W)
        frontiers = torch.nn.functional.max_pool2d(frontiers, 7, stride=1, padding=3)
        return frontiers

    def undo_memory_opts(self, batch):
        inputs, labels = batch
        inputs["semmap"] = inputs["semmap"].float()
        labels["semmap"] = labels["semmap"].float()
        labels["object_pfs"] = labels["object_pfs"].float() / 1000.0
        if "area_pfs" in labels:
            labels["area_pfs"] = labels["area_pfs"].float() / 1000.0
        return (inputs, labels)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def convert_to_data_parallel(self):
        self.encoder = nn.DataParallel(self.encoder)
        self.object_decoder = nn.DataParallel(self.object_decoder)
        if self.area_decoder is not None:
            self.area_decoder = nn.DataParallel(self.area_decoder)

    def convert_object_pf_to_distance(self, opfs, min_value=1e-20, max_value=1.0):
        """
        opfs - (bs, N, H, W)
        """
        opfs = torch.clamp(opfs, min_value, max_value)
        data_cfg = self.cfg.DATASET
        max_d = data_cfg.object_pf_cutoff_dist
        dists = max_d - opfs * max_d
        return dists

    def convert_distance_to_object_pf(self, dists):
        """
        dists - (bs, N, H, W)
        """
        data_cfg = self.cfg.DATASET
        max_d = data_cfg.object_pf_cutoff_dist
        opfs = torch.clamp((max_d - dists) / max_d, 0.0, 1.0)
        return opfs

    def get_pf_cfg(self):
        return {"dthresh": self.cfg.DATASET.object_pf_cutoff_dist}
    
def get_activation_fn(activation_type):
    assert activation_type in ["none", "sigmoid", "relu"]
    activation = nn.Identity()
    if activation_type == "sigmoid":
        activation = nn.Sigmoid()
    elif activation_type == "relu":
        activation = nn.ReLU()
    return activation