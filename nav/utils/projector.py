import torch 
from torch import nn 
import numpy as np 
from utils import depth_utils as du 
from utils import pts_utils as ptsu
from argparse import Namespace
def get_matrix(pose, device):
    """
    Get the transformation (rotation + translation) matrices according 
    to the pose
    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180.
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)

    rot_matrix = torch.zeros((bs, 3, 3), device=device)
    rot_matrix[:, 0, 0] = cos_t
    rot_matrix[:, 0, 1] = -sin_t
    rot_matrix[:, 1, 0] = sin_t
    rot_matrix[:, 1, 1] = cos_t
    rot_matrix[:, 2, 2] = 1

    trans_matrix = torch.stack([y, x, torch.zeros(bs).to(device)], dim=1)


    return rot_matrix, trans_matrix


def get_local_map(coords, grid_shape, min_z, max_z):
    """Get the local map based on coordinates of shape. Only explored area and obstacles
    channels are needed, the agent location and trajectory will be set to zero and be updated outside"""
    local_map = torch.zeros(grid_shape)
    
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]


    x_coords = torch.clamp(x_coords, min = 0, max = grid_shape[1] - 1)
    y_coords = torch.clamp(y_coords, min = 0, max = grid_shape[2] - 1)
    
    # Explored area
    local_map[1, x_coords, y_coords] = 1.

    agent_height_mask = (coords[:, 2] >= min_z) & (coords[:, 2] <= max_z)
    agent_height_coords = coords[agent_height_mask]

    agent_x = agent_height_coords[:, 0]
    agent_y = agent_height_coords[:, 1]

    agent_x = torch.clamp(agent_x, min = 0, max = grid_shape[1] - 1)
    agent_y = torch.clamp(agent_y, min = 0, max = grid_shape[2] - 1)

    # Obstacles (agent height projection)
    local_map[0, agent_x, agent_y] = 1.

    return local_map

    
class BackProjector(nn.Module):
    def __init__(self, args, contain_seg = False):
        """contain_seg specifies if the input feature contains segmentation labels, which is for collecting GT data
           keep_map specifies whether to return the 2D explored map, agent trajectories and so on"""
        super().__init__()
        self.args = args 
        self.device = args.device
        self.du_scale = args.du_scale
        self.agent_height = args.camera_height * 100.
        self.screen_h = args.frame_height 
        self.screen_w = args.frame_width
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, args.hfov)
        self.vr = args.vision_range 
        self.resolution = args.map_resolution 
        self.z_resolution = args.map_resolution 
        # self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        
        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)


        self.shift_loc = [self.vr * self.resolution //2, 0, np.pi/2.0]

        self.shift_loc2 = [self.vr * self.resolution // 2 / 100. , 0, np.pi/2.0]

        self.seg = contain_seg

    
    def forward(self, obs, pose_obs, coords_last, feats_last, poses_last, map_last):
        if self.seg:
            assert obs.shape[1] == 5, "obs should contain segmentation labels for collecting GT data"
        bs, feature_dim, _, _= obs.shape
        feature_dim = feature_dim - 1  # depth obs is not kept
        depth = obs[:, 3, :, :]
        if self.seg:
            rgb = obs[:, :3, :, :]
            labels = obs[:, -1:, :, :]
            feature = torch.cat([rgb, labels], dim=1)
        else:
            feature = obs[:, :3, :, :]

        # Feature may contain classes logits or other features, but projector will only be responsible for updating coord and rgb
        extra_dim = feats_last.shape[-1] - 6

        max_h = self.max_height 
        min_h = self.min_height 
        z_resolution = self.z_resolution 
        vr = self.vr 
        # TODO: Adjust min_z?
        min_z = int(25 / z_resolution - min_h)
        # min_z = int(40 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)

        bs, _, local_w, local_h = map_last.shape 

        # The boundary in whole local voxels for current observed voxels
        x1 = self.map_size_cm // (self.resolution * 2) - vr // 2

        y1 = self.map_size_cm // (self.resolution * 2)

        current_poses = get_new_pose_batch(poses_last, pose_obs)
        st_pose = current_poses.clone().detach()
        # Transform pose into local map coordinate
        # heading is 0 pointing the positive x axis 
        # And coordinates are relative to the center of the local_map instead of the origin 
        st_pose[:, :2] = st_pose[:, :2] * 100. / self.resolution 
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        coords = du.get_point_cloud_from_z_t(depth / 100., self.camera_matrix, self.device, scale=self.du_scale)

        # Transform the point clouds to egocentric coordinates taking account for elevation
        # Since 4-action setting is adopted here, elevation is set to 0. If 6-action is adopted, feel free to change
        agent_view_t = du.transform_camera_view_t(coords, self.agent_height / 100., 0, self.device)

        # Transform to geocentric coordinates taking account for rotation 
        # However, the origin here is not the origion of full map but the vr x vr rectangle 
        coords = du.transform_pose_t(agent_view_t, self.shift_loc2, self.device)

        rot, trans = get_matrix(st_pose, self.device)
        coords = coords.view(bs, -1, 3)
        feats = feature.permute(0, 2, 3, 1).view(bs, -1, feature_dim)
        sampled_coords = []
        sampled_feats = []
        local_maps = []
        min_coord = torch.tensor([0., 0. , min_h * self.resolution  / 100.]).to(self.device)
        max_coord = torch.tensor([vr * self.resolution / 100., vr * self.resolution / 100., max_h * self.resolution / 100.]).to(self.device)
        for i in range(bs):
            coord = coords[i]
            feat = feats[i]

            batch_mask = coords_last[:, 0] == i 
            coord_last = coords_last[batch_mask]
            feat_last = feats_last[batch_mask]

            # Get the points in the valid range
            valid_indices = torch.all((coord > min_coord) & (coord < max_coord), dim = 1)
            coord = coord[valid_indices]
            feat = feat[valid_indices]

            # Some envs may be waiting for other envs that there is no observation and hence no valid point clouds
            # Simply skip those envs
            if(coord.shape[0] <= 0.):
                sampled_coords.append(coord_last)
                sampled_feats.append(feat_last)
                local_maps.append(map_last[i])
                continue 


            # Downsampling the points 
            grid_size = self.resolution / 100. 
            indices, grid_coord = ptsu.voxel_downsample(coord, grid_size = grid_size, return_grid_coord = True)
            coord = coord[indices]
            feat = feat[indices]

            
            # Insert the observed vr x vr rectangle into local_w x local_h local map
            grid_coord = grid_coord[:, [1, 0, 2]]
            grid_coord[:, 0] += y1 
            grid_coord[:, 1] += x1 

            coord = coord[:, [1, 0, 2]]
            coord[:, 0] += y1 * grid_size
            coord[:, 1] += x1 * grid_size  

            # Apply transformation (rotation + translation) based on pose
            grid_coord = grid_coord.to(torch.float32)
            # rotation is relative to the center of the map, firstly translate all points to center
            grid_coord[:, :2] -= torch.tensor([local_w // 2., local_h //2.]).to(self.device) 
            # Get and apply the transformation matrix 
            grid_coord = torch.matmul(grid_coord, rot[i].T) + trans[i]
            grid_coord = torch.round(grid_coord).long()

            grid_coord[:, 2] -= self.min_height


            # For actual coordiantes, apply the same transformation but scaled by grid_size 
            coord = coord.to(torch.float32)
            coord[:, :2] -= torch.tensor([local_w // 2. * grid_size, local_h //2. * grid_size]).to(self.device) 
            coord = torch.matmul(coord, rot[i].T) + trans[i] * grid_size
            
            coord[:, 2] -= self.min_height * grid_size



            # Get and update local map
            local_map = get_local_map(grid_coord, (4, local_w, local_h), min_z, max_z).to(self.device)
            local_map = torch.cat((map_last[i].unsqueeze(1), local_map.unsqueeze(1)), 1)
            local_map, _ = torch.max(local_map, 1)

            # Attach batch index to the coordinates 
            grid_coord = torch.cat([torch.full((grid_coord.shape[0], 1), i).to(self.device), grid_coord], dim = 1)

            extras = torch.zeros((coord.shape[0], extra_dim)).to(self.device)
            feat = torch.cat([coord, feat, extras], dim = 1)


            # Merge the current observed points with previously observed points 
            # Here the notation 'coord' refer to grid coord, and the actual coord is merged into feat 
            coord = torch.cat([coord_last, grid_coord], dim = 0)
            feat = torch.cat([feat_last, feat], dim = 0)


            sampled_coords.append(coord)
            sampled_feats.append(feat)
            local_maps.append(local_map)
        
        # coords: N x 4, feats: N x nF, local_maps: bs x 4 x local_w x local_h 
        sampled_coords = torch.cat(sampled_coords, dim = 0)
        sampled_feats = torch.cat(sampled_feats, dim = 0)
        local_maps = torch.stack(local_maps)




        return sampled_coords, sampled_feats, local_maps, current_poses
    
def get_new_pose_batch(pose, rel_pose_change):
    pose[:, 1] += rel_pose_change[:, 0] * \
        torch.sin(pose[:, 2] / 57.29577951308232) \
        + rel_pose_change[:, 1] * \
        torch.cos(pose[:, 2] / 57.29577951308232)
    pose[:, 0] += rel_pose_change[:, 0] * \
        torch.cos(pose[:, 2] / 57.29577951308232) \
        - rel_pose_change[:, 1] * \
        torch.sin(pose[:, 2] / 57.29577951308232)
    pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

    pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
    pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

    return pose

class Final_Projector(BackProjector):
    @classmethod
    def from_config(cls, cfg, device):
        args = Namespace()
        args.device = device
        args.frame_height = cfg.GLOBAL_AGENT.frame_height
        args.frame_width = cfg.GLOBAL_AGENT.frame_width
        args.map_resolution = cfg.GLOBAL_AGENT.map_resolution
        args.map_size_cm = cfg.GLOBAL_AGENT.map_size_cm
        args.camera_height = cfg.GLOBAL_AGENT.camera_height
        args.global_downscaling = cfg.GLOBAL_AGENT.global_downscaling
        args.vision_range = cfg.PROJECTOR.vision_range
        args.hfov = float(cfg.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV)
        args.du_scale = cfg.PROJECTOR.du_scale
        args.cat_pred_threshold = cfg.PROJECTOR.cat_pred_threshold
        args.exp_pred_threshold = cfg.PROJECTOR.exp_pred_threshold
        args.map_pred_threshold = cfg.PROJECTOR.map_pred_threshold
        args.num_sem_categories = cfg.GLOBAL_AGENT.num_sem_categories
        args.num_processes = cfg.NUM_ENVIRONMENTS

        return cls(args)