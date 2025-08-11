import torch 
def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    index = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, index 

def voxel2points(volume):
    """The last element of feature dimension indicates whether the voxel is empty"""
    nF, h, w, d = volume.shape
    # volume = torch.where(occupancy > 0, volume, 0)
    device = volume.device

    # Create grid of voxel coordinates (x, y, z)
    x, y, z = torch.meshgrid(torch.arange(h), torch.arange(w), torch.arange(d))
    xyz = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1).to(device)  # Shape: [N, 3]

    # Flatten the volume to extract feat values
    feat = volume.permute(1, 2, 3, 0).reshape(-1, nF).to(torch.float32)

    valid_mask = (volume[-1, :, :, :].flatten() > 0.).to(device)
    valid_xyz = xyz[valid_mask]
    valid_feat = feat[valid_mask, :-1]

    return valid_xyz, valid_feat

def voxel_downsample(coords, grid_size = 0.05, return_grid_coord = False):
    """Downsampling points clouds according to specific grid size
    Parameters:
        coords (torch.Tensor): Coordianates of the points, of shape N x 3
        grid_size (float): the size of a single grid, only one of the points lying in a same grid will be left
        return_grid_coord (bool): Whether to returned the coords in grid. If not, only return the indices of sampled points
    Returns:
        index (torch.Tensor): The indices of sampled points, of shape N', N' <= N 
        gird_coord (torch.Tensor)"""
    
    
    sampled_coord = coords / torch.tensor(grid_size)
    grid_coord = torch.floor(sampled_coord).to(torch.int64)
    
    grid_coord, indices = unique(grid_coord, dim = 0)

    if return_grid_coord:
        return indices, grid_coord 

    return indices 

def points2voxels(coords, feat, grid_shape, addi_occupancy = True, normalized = False, vr = None, max_h = None, 
                  min_h = None):
    """
    Downsampling point clouds to voxels. 
    Params: 
        coords (torch.Tensor): N x 3. Coordinates could be either normalized to (-1, 1) or non-normalized
        feat (torch.Tensor): N x nF. 
        grid_shape (tuple): (nF, W, D, H)
        normalized (bool): Whether the coordinates are normalized, if True, pass in vr and min_h & max_h to denormalize
        min_h & max_h: used to denormalize the points
    
    Returns:
        voxels (torch.Tensor): (nF + 1) x W x D x H. Each grid contain features. The additional feature 
                                indicate the voxel is empty.
    """

    # Check the shape consistency
    assert coords.shape[0] == feat.shape[0], "Coordinate and Features should have the same num" 
    assert coords.shape[-1] == 3, "Coordinate must has 3 dimensions"

    device = coords.device
    
    # Denormalize 
    if normalized:
        coords[..., :2] = (coords[..., :2] * vr / 2.) + (vr // 2.)
        coords[..., 2] = (coords[..., 2] * (max_h - min_h) / 2.) + (max_h + min_h) // 2.


    coords = torch.round(coords).to(torch.int32)

    
    nF, w, d, h = grid_shape
    batch_coords, index = unique(coords, dim=0)


    # Directly index and assign feature values to voxels and occupancy 
    if addi_occupancy:
        voxels = torch.zeros((nF + 1, w, d, h)).to(device)
        voxels[:-1, batch_coords[:, 0], batch_coords[:, 1], batch_coords[:, 2]] = feat[index].T
        voxels[-1, batch_coords[:, 0], batch_coords[:, 1], batch_coords[:, 2]] = 1.
    
    else:
        voxels = torch.zeros(grid_shape).to(device)
        voxels[:, batch_coords[:, 0], batch_coords[:, 1], batch_coords[:, 2]] = feat[index].T


    
    return voxels 

def batch_points2voxels(points_coords, points_feats, batch_size, grid_shape, addi_occupancy = True,
                        normalized = False, vr = None, max_h = None, min_h = None):
    """Convert a batch of points to voxels
    Parameters:
        points_coords (torch.Tensor): coordinates of the points, shape (N x 4).
                                    The additional feature represents the batch index
                                    Or of shape (bs x N x 3) in some case
        points_feats (torch.Tensor): features of the points, shape N x nF
        grid_shape (tuple): nF x h x w x d 
        addi_occupancy (bool): True by default, add an extra dimension to indicate whether the voxel is occupied
    Returns:
        voxels (torch.Tensor): of shape bs x (nF + 1) x h x w x d, additional value in feature dimension indicates empty voxel"""
    device = points_coords.device
    if len(points_coords.shape) == 3:
        assert points_coords.shape[-1] == 3, "points should be in shape (bs x N x 3) or (N x 4) to be indexed within batch"
        batch_voxels = []
        for b in range(batch_size):
            voxels = points2voxels(points_coords[b], points_feats[b], grid_shape, addi_occupancy, 
                                   normalized, vr, max_h, min_h)
            batch_voxels.append(voxels)
        voxels = torch.stack(batch_voxels).to(device)
    else:
        assert points_coords.shape[-1] == 4, "points should be in shape (bs x N x 3) or (N x 4) to be indexed within batch"
        if not addi_occupancy:
            voxels = torch.zeros(batch_size, *grid_shape).to(device)
        else:
            nF, h, w, d = grid_shape
            voxels = torch.zeros(batch_size, nF + 1, h, w, d)
        for b in range(batch_size):
            mask = points_coords[:, 0] == b
            coords = points_coords[mask, 1:]
            feats = points_feats[mask]
            voxels[b] = points2voxels(coords, feats, grid_shape, addi_occupancy, normalized, vr, max_h, min_h)
    
    return voxels

def get_safe_points(points, colors, grid_shape = None):
    """
    Some points is out of our vision range (local rectangle of vr x vr ).
    They are indicated by coordinates beyond range [-1, 1] or out of given voxel shape
    Params:
        points : bs x N x 3 (or N x 4)
        colors : bs x N x 3 (or N x nF)
        grid_shape: h x w x d, Note that you only pass it when coordinates are not normalized
        min_h & max_h: used to normalize and denormalize vertical coordinates
    Returns:
        safe_points: bs x N' x 3 
        safe_colors: bs x N' x 3
    """


    if len(points.shape) == 3:
        safe_points = []
        safe_colors = []
        for i in range(points.shape[0]):
            if grid_shape is not None:
                h, w, d = grid_shape
                safe_mask = (points[:, 0] <= h) & (points[:, 1] <= w) & (points[:, 2] <= d) & (torch.all(points >= 0, dim = 1))
            # Since coords larger than 0.95 will be rounded to 1 when voxelizing, deem it unsafe either.
            else:
                safe_mask = torch.all((points[i] > -.95) & (points[i] < .95), dim = -1)

            coords = points[i][safe_mask]
            coords = torch.cat([torch.full((coords.shape[0], 1), i).to(coords.device), coords], dim = 1)
            safe_points.append(coords) 
            safe_colors.append(colors[i][safe_mask])
        
        safe_points = torch.cat(safe_points, dim=0)
        safe_colors = torch.cat(safe_colors, dim=0)


        return safe_points, safe_colors
        
    else:
        if grid_shape is not None:
            h, w, d = grid_shape
            safe_mask = (points[:, 1] <= h) & (points[:, 2] <= w) & (points[:, 3] <= d) & (torch.all(points[:, 1:] >= 0, dim = 1))
        else:
            safe_mask = torch.all((points[:, 1:] > -.95) & (points[:, 1:] < .95), dim = 1)
        return points[safe_mask], colors[safe_mask]

def get_local_points(global_points_coords, global_points_feats, e, x1, x2, y1, y2):
    """"Get local points from global points according to given boundaries
    Parameters:
        global_points_coords (torch.Tensor): Coordinates of global points, shape N x 3
        global_points_feats (torch.Tensor): Features of global points, shape N x (nF + 1)
                                            The additional feature dimension is the batch indices
        x1, x2, y1, y2 (int): boundaries of local points, the points with x, y coords within the range are extracted
    
    Returns:
        local_points_coords (torch.Tensor): shape N' x 3 (N' <= N)
        local_points_feats (torch.Tensor): shape N' x (nF+1)
        updated_global_points_coords (torch.Tensor): The extracted points are removed, shape (N - N') x 3
        updated_global_points_feats (torch.Tensor): shape (N - N') x (nF+1)
        num (torch.Tensor): The number of extracted points. Can be useful for computing intrinsic reward"""
    
    local_mask = (global_points_coords[:, 0] == e) & \
                (global_points_coords[:, 1] >= x1) & (global_points_coords[:, 1] < x2) & \
                (global_points_coords[:, 2] >= y1) & (global_points_coords[:, 2] < y2)
    
    local_points_coords = global_points_coords[local_mask]
    local_points_feats = global_points_feats[local_mask]

    updated_global_points_coords = global_points_coords[~local_mask]
    updated_global_points_feats = global_points_feats[~local_mask]

    return local_points_coords, local_points_feats, updated_global_points_coords, \
            updated_global_points_feats, torch.sum(local_mask)

def get_points_num(points_coords, b):
    """Get the number of points with specific batch index"""
    mask = points_coords[:, -1] == b 
    return torch.sum(mask)
