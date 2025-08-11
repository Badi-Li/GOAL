import torch 
def get_semantic_map(sem_coords, sem_pred, nc, map_size, num_scenes, min_z, max_z, cat_pred_threshold):
    # Exclude out of bound points
    valid_mask = (sem_coords[:, 1] < map_size[0]) & (sem_coords[:, 1] >= 0) & \
                (sem_coords[:, 2] < map_size[1]) & (sem_coords[:, 2] >= 0) & \
                (sem_coords[:, 3] <= max_z)

    sem_coords = sem_coords[valid_mask].long()
    sem_pred = sem_pred[valid_mask]
    device = sem_coords.device
    semantic_maps = torch.zeros((num_scenes, nc, *map_size)).to(device)
    for b in range(num_scenes):
        batch_mask = sem_coords[:, 0] == b 
        batch_coords = sem_coords[batch_mask]
        batch_pred = sem_pred[batch_mask]
        semantic_maps[b, batch_pred, batch_coords[:, 1], batch_coords[:, 2]] += 1
        semantic_maps[b] = torch.clamp(semantic_maps[b] / cat_pred_threshold, min = 0.0, max = 1.0)
    return semantic_maps