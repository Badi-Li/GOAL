from trainer import Trainer
import torch   
from utils.misc import set_all_seed, find_free_port
import torch.multiprocessing as mp 
from arguments import get_args
import os 
def train(rank, args):
    trainer = Trainer(args, rank)
    trainer.train()

if __name__ == '__main__':
    args = get_args()
    set_all_seed(args.seed)
    nprocs = torch.cuda.device_count()
    args.world_size = nprocs

    if args.distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    mp.set_start_method('spawn', force=True)
    mp.spawn(train, nprocs=nprocs, args = (args, ))