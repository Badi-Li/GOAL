"""
This is for loading utils like model, dataset, optimizer etc.
"""
from models.DiT import DiT
import torch 
from torch.utils.data import DataLoader
import os 
import logging
from datasets.SemanticMapDataset import SemanticMapDataset
from collections import OrderedDict
def load_model(args):
    return DiT(
        input_size = args.input_size,
        patch_size = args.patch_size, 
        in_channels = args.in_channels,
        hidden_size = args.hidden_size, 
        depth = args.depth,
        num_heads = args.num_heads,
        condition = args.condition
    )

def load_optimizer(args, model):
    return torch.optim.AdamW(
        model.parameters(),
        lr = args.base_lr,
        weight_decay = args.weight_decay
    )

def load_scheduler(args, optimizer, loader_length):
    total_steps = args.epochs * loader_length 
    warmup_steps = int(args.warmup_steps * total_steps) if args.warmup_steps < 1 else int(args.warmup_steps)
    plateau_steps = int(args.plateau_steps * total_steps) if args.plateau_steps < 1 else int(args.plateau_steps)
    

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = args.final_lr / args.base_lr,
        total_iters = warmup_steps
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = total_steps - warmup_steps - plateau_steps,
        eta_min = args.final_lr
    )

    plateau_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=args.final_lr / args.base_lr,
        total_iters=plateau_steps
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler, plateau_scheduler],
        milestones=[warmup_steps, total_steps - plateau_steps]
    )

    return scheduler

def load_dataset(args):
    return SemanticMapDataset(args, split = 'train')

def load_loader(args, rank, dataset):
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas = args.world_size,
            rank = rank
        )

    else: 
        sampler = None 
    loader = DataLoader(dataset, 
                        batch_size = args.batch_size, 
                        sampler = sampler, 
                        shuffle = False,
                        num_workers = args.num_workers,
                        drop_last = False)
    
    return loader 

def load_logger(args):
    log_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        filename = log_file,
        level = logging.INFO, 
        format='[%(asctime)s %(filename)s line %(lineno)d] %(message)s'
    )
    return logging.getLogger(args.exp_name)

def load_ckpt(model, ckpt):
    state = OrderedDict()
    for key, value in ckpt.items():
        if key.startswith('module'):
            state[key[7:]] = value
        else:
            state[key] = value
    model.load_state_dict(state)
    return model
