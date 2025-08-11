from models.sparse_unet import SpUNetBase
from models.DiT import DiT
import torch 
from collections import OrderedDict
def get_seg_model(cfg):
    model = SpUNetBase(cfg.in_channels, 
                       cfg.out_channels, 
                       cfg.base_channels, 
                       cfg.channels,
                       cfg.layers
                    )
    
    model = load_ckpt(model, cfg.sem_pred_weights)
    return model

def get_fm_model(cfg):
    model =  DiT(
        input_size = cfg.input_size,
        patch_size = cfg.patch_size, 
        in_channels = cfg.in_channels,
        hidden_size = cfg.hidden_size, 
        num_heads = cfg.num_heads,
        depth = cfg.depth, 
        condition = cfg.condition
    )
    
    model = load_ckpt(model, cfg.fm_weights)
    return model

def load_ckpt(model, ckpt):
    ckpt = torch.load(ckpt, map_location='cpu')
    if 'state_dict' in ckpt.keys():
        ckpt = ckpt['state_dict']
    weight = OrderedDict()
    for key, value in ckpt.items():
        if key.startswith('module.backbone'):
            key = key[16:] # module.backbone.xxxx -> xxxx 
        elif key.startswith('module'):
            key = key[7:]
        weight[key] = value
    model.load_state_dict(weight)
    return model 