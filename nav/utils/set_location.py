import torch 
import numpy as np 
def set_target_loc(t_pfs, t_area_pfs, cn, thr, mask = None): # generated + potential function
    t_pfs = torch.from_numpy(t_pfs).to(t_area_pfs.device)
    
    cat_semantic_map = t_pfs[cn].to(t_area_pfs.device)
    cat_semantic_map = torch.where(cat_semantic_map<thr, torch.tensor(0.0).to(cat_semantic_map.device), cat_semantic_map)
    
    t_area = t_area_pfs[0]

    new_map = cat_semantic_map+t_area
    if mask is not None:
        new_map = new_map * torch.from_numpy(1 - mask).to(new_map.device)
    
    tar_loc = torch.where(new_map == torch.max(new_map))
    target = np.zeros((2))
    target[0] = tar_loc[0][0]
    target[1] = tar_loc[1][0]

    return target

def set_target_loc_base(t_area_pfs, mask = None):
    t_area = t_area_pfs[0]
    if mask is not None:
        t_area = t_area * torch.from_numpy(1 - mask).to(t_area.device)
    area_loc = torch.where(t_area == torch.max(t_area))
    target = np.zeros((2))
    target[0] = area_loc[0][0]
    target[1] = area_loc[1][0]
    return target 