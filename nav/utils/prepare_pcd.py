import spconv.pytorch as spconv
import torch 
import numpy as np 
import random 
class Prepare_data(object):
    def __init__(self):
        self.normalize_color = NormalizeColor()
        self.center_shift = CenterShift(apply_z = False)


        self.rotate1 = RandomRotateTargetAngle(angle = [0], axis = "z", 
                                               center = [0, 0, 0], p = 1)
        self.rotate2 = RandomRotateTargetAngle(angle = [1 / 2], axis = "z", 
                                               center = [0, 0, 0], p = 1)
        self.rotate3 = RandomRotateTargetAngle(angle = [1], axis = "z", 
                                               center = [0, 0, 0], p = 1)
        self.rotate4 = RandomRotateTargetAngle(angle = [3 / 2], axis = "z", 
                                               center = [0, 0, 0], p = 1)

        self.aug_transform = [self.rotate1, self.rotate2, self.rotate3, self.rotate4]


    def __call__(self, grid_coords, feats, batch_size, device = None, spatial_shape = None):

        if isinstance(grid_coords, np.ndarray):
            assert device is not None, "When passing in numpy array, device must be specified"
        else:
            device = grid_coords.device
        transformed_coords = []
        transformed_feats = []
        for i in range(batch_size):
            batch_mask = grid_coords[:, 0] == i
            if(torch.sum(batch_mask) <= 0.):
                continue 
            grid_coord = grid_coords[batch_mask]
            for i in range(1, 4):
                grid_coord[:, i] -= grid_coord[:, i].min()
            color = self.normalize_color(feats[batch_mask][:, 3:6])
            coord = self.center_shift(feats[batch_mask][:, :3], device)
            transformed_coords.append(grid_coord.int())
            feat = torch.cat([coord, color], dim = 1)
            transformed_feats.append(feat)
        
        transformed_feats = [feat for feat in transformed_feats if feat.numel() > 0]
        transformed_coords = [coord for coord in transformed_coords if coord.numel() > 0]
        if len(transformed_feats) == 0:
            return 0 
        feats = torch.cat(transformed_feats, dim = 0)
        index = torch.cat(transformed_coords, dim = 0)
        x = spconv.SparseConvTensor(
            feats, index, spatial_shape, batch_size
        )

        return x
        
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, coord, device = None):

        if isinstance(coord, np.ndarray):
            assert device is not None, "When passing in numpy array, device must be specified"
            coord = torch.from_numpy(coord).to(device)
        else:
            device = coord.device
        
        x_min, y_min, z_min = coord.min(axis=0)[0]
        x_max, y_max, _ = coord.max(axis=0)[0]
        if self.apply_z:
            shift = torch.tensor([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min], device = device)
        else:
            shift = torch.tensor([(x_min + x_max) / 2, (y_min + y_max) / 2, 0], device = device)
        coord -= shift
        return coord

class NormalizeColor(object):
    def __call__(self, color):
        color = color / 127.5 - 1
        return color

class RandomRotateTargetAngle(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, coord):
        if random.random() > self.p:
            return coord
        angle = random.choice(self.angle) * torch.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = torch.tensor([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]]).to(coord.device)
        elif self.axis == "y":
            rot_t = torch.tensor([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]]).to(coord.device)
        elif self.axis == "z":
            rot_t = torch.tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]).to(coord.device)
        else:
            raise NotImplementedError
        
        if self.center is None:
            x_min, y_min, z_min = coord.min(axis=0)[0]
            x_max, y_max, z_max = coord.max(axis=0)[0]
            center = torch.tensor([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]).to(coord.device)
        else:
            center = torch.tensor(self.center).to(coord.device)
        coord -= center 
        torch.matmul(coord.to(torch.float64), rot_t.T)
        coord += center
        
        return coord