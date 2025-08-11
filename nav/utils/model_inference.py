from flow_matching.solver import ODESolver 
import torch 
from torch.nn import functional as nnf
import torchvision.transforms.functional as F
import numpy as np 
def max1(a,b):
    if a>b:
        return a
    return b

def min1(a,b):
    if a<b:
        return a
    return b

def model_inference(x, model, cfg):
    solver = ODESolver(velocity_model = model)
    free_mask = (torch.sum(x, dim = 0, keepdim = True) == 0).repeat(x.shape[0], 1, 1).unsqueeze(0)
    if cfg.condition is None:
        x0 = x.unsqueeze(0)
        x1 = solver.sample(x_init = x0, 
                           step_size = 1 / cfg.num_steps,
                           mask = free_mask)
    elif cfg.condition == 'CA':
        x0 = cfg.std * torch.randn_like(x).unsqueeze(0)
        x1 = solver.sample(x_init = x0,
                           step_size = 1 / cfg.num_steps,
                           mask = free_mask,
                           model_extras = {
                               'c': x.unsqueeze(0)
                           }
                        )

    else:
        raise ValueError(f'No condition type {cfg.condition}')

    return x1.squeeze(0)

def data_precompute(data, type):
    nz_c = torch.nonzero(torch.sum(data, dim=0))
    # data: c x h x w (No batch dimension)
    if (not torch.is_tensor(data)):
        data = torch.from_numpy(data)
    if nz_c.shape[0]==0:
        return np.zeros_like(data.cpu()), (0, 0, 0)
    # x_min, x_max, y_min, y_max = min(nz_c[:,0]), max(nz_c[:,0]), min(nz_c[:,1]), max(nz_c[:,1])
    x_min, x_max, y_min, y_max = nz_c[:,0].min(), nz_c[:,0].max(), nz_c[:,1].min(), nz_c[:,1].max()
    wh = min1(x_max-x_min,y_max-y_min)
    # Only a row or column has value is also not acceptable 
    if wh == 0:
        return np.zeros_like(data.cpu()), (0, 0, 0)
    if type == 0:
        input_map = F.resize(F.crop(data, x_min,y_min,wh,wh),256)
    elif type>0:
        dwh = type
        x_min1 = int(max1(0, (x_min - dwh)))
        x_max1 = int(min1((x_max + dwh), 479))
        y_min1 = int(max1(0, (y_min - dwh)))
        y_max1 = int(min1((y_max + dwh), 479))
        wh1 = min1(x_max1-x_min1,y_max1-y_min1)
        input_map = F.resize(F.crop(data, x_min1,y_min1,wh1,wh1),256)
        x_min=x_min1
        y_min=y_min1
        wh=wh1
    else:
        dwh = type
        x_min=int(x_min-dwh)
        y_min=int(y_min-dwh)
        wh=int(wh+2*dwh)
        input_map = F.resize(F.crop(data, x_min, y_min, wh, wh),256)

    return input_map, (x_min,y_min,wh)

def merge_map(org_map, gen_map, loc):
    org_map = org_map.detach()
    gen_map = gen_map.detach()
    x_min,y_min,wh = loc
    gen_map = nnf.interpolate(gen_map.unsqueeze(dim=0), size=[wh, wh])
    gen_map = gen_map.squeeze(dim=0)
    if (x_min+wh<=480 and y_min+wh<=480):
        org_map[:,x_min:x_min+wh,y_min:y_min+wh] = gen_map
    elif (x_min+wh>480):
        gen_map=gen_map[:, 0:480-x_min, :]
        org_map[:,x_min:480,y_min:y_min+wh] = gen_map
    elif (y_min+wh>480):
        gen_map=gen_map[:,:, 0:480-y_min]
        org_map[:,x_min:x_min+wh,y_min:480] = gen_map
    else:
        gen_map=gen_map[:, 0:480-x_min, 0:480-y_min]
        org_map[:,x_min:480,y_min:480] = gen_map
    return org_map.cpu().numpy()

def cal_res(input, model, wh, cfg, return_intermediates = False):
    ratio = cfg.expand_ratio
    model_input, loc_s = data_precompute(input, wh * ratio)
    if loc_s[2]==0:
        return np.zeros_like(input.cpu()), 1
    if return_intermediates:
        result_map, intermediates = model_inference(model_input, model, cfg)
        result_map = merge_map(input, result_map, loc_s)
        result_intermediates = []
        for intermediate in intermediates:
            result_intermediates.append(merge_map(input, intermediate.squeeze(0), loc_s))
        return result_map, result_intermediates, 0
    result_map = model_inference(model_input, model, cfg) 
    result_map = merge_map(input, result_map, loc_s)
    return result_map, 0