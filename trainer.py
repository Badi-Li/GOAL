from utils import load 
import torch 
import torch.distributed as dist 
from torch.utils.tensorboard import SummaryWriter
import os 
from utils.misc import AverageMeter
from copy import deepcopy
from torch import nn
from collections import OrderedDict
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
import time 
import torch.nn.functional as F 
class Trainer():
    def __init__(self, args, rank):
        self.args = args
        self.unpack_args(rank)
        self.prepare()

        torch.cuda.set_device(rank)
        torch.backends.cudnn.enabled = True

        self.kernel = torch.ones(1, 1, 3, 3, device = rank)
    
    def train(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.epochs):
            for i, data in enumerate(self.data_loader):
                step = epoch * self.epoch_len + (i + 1)
                end = time.time()
                loss = self.forward(data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
                self.scheduler.step() 
                update_ema(self.ema, self.model.module, self.ema_decay)

                self.time_meter.update(time.time() - end)
                self.loss_meter.update(loss.item(), self.bs)
                end = time.time()

                self.log(step) 
            
            if (epoch + 1) % self.save_freq == 0:
                self.save(epoch) 
            self.log(step, epoch)
            self.loss_meter.reset()
              
    def log(self, step, epoch = None):
        if (step + 1) % self.log_freq == 0 and self.main:
            remain_time = self.calculate_remain_time(step)
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                    'Steps [{step}/{total_iter}] '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Average epoch Loss {loss_meter.avg:.7f} '
                    'Remain {remain_time} '
                    'lr {lr: .7f}'.format(
                        step = step, total_iter = self.total_len,
                        loss_meter = self.loss_meter, 
                        batch_time = self.time_meter,
                        remain_time = remain_time,
                        lr = lr),
                    )
            self.writer.add_scalar('train/lr', lr, step)
            self.writer.add_scalar('train/step_loss', self.loss_meter.val, step)
        if epoch is not None and self.main:
            self.logger.info(
                    'Epochs [{epoch}/{epochs}]'
                    'Average loss {loss: .7f}'.format(
                        epoch = epoch + 1, epochs = self.epochs,
                        loss = self.loss_meter.avg),
                    )
            self.writer.add_scalar('train/epoch_loss', self.loss_meter.avg, epoch)

    def calculate_remain_time(self, step):
        remain_len = self.total_len - step 
        remain_time = remain_len * self.time_meter.avg 
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        return remain_time
    
    def forward(self, data):
        if self.args.prior:
            partial_sem, complete_sem, prior = data 
            prior = prior.float().to(self.device)
        else:
            partial_sem, complete_sem = data 
            prior = None 

        partial_sem = partial_sem.float().to(self.device)
        complete_sem = complete_sem.float().to(self.device)
        self.bs = partial_sem.shape[0]
        if self.args.condition is None:
            return self.model_forward(partial_sem, complete_sem, prior)
        elif self.args.condition == 'CA':
            return self.model_forward_CA(partial_sem, complete_sem, prior)
        elif self.args.condition == 'PA':
            return self.model_forward_PA(partial_sem, complete_sem, prior)
        else:
            raise ValueError(f'Invalid condition type {self.args.condition}!')
    
    def unpack_args(self, rank):
        args = self.args
        if args.resume is not None:
            self.ckpt = torch.load(args.resume, map_location = 'cpu')
            self.start_epoch = self.ckpt['epoch']
        else:
            self.ckpt = None 
            self.start_epoch = args.start_epoch
        
        self.epochs = self.args.epochs
        self.device = self.rank = rank 
        self.condition = args.condition 
        self.log_freq = args.log_freq
        self.std = args.std
        self.ema_decay = args.ema_decay
        self.save_freq = args.save_freq
        self.bs = args.batch_size

        self.main = rank == 0

    def preprocess_data(self, partial_sem, complete_sem, prior, t):
        free_mask = self.compute_free_mask(partial_sem)

        noise = self.args.std * torch.randn_like(partial_sem)
        x0 = free_mask * noise + ~free_mask * partial_sem 
        x1 = self.augment(complete_sem, free_mask, prior)
        
        sample = self.path.sample(t = t, x_0 = x0, x_1 = x1)

        return sample.x_t, sample.dx_t

    def model_forward(self, partial_sem, complete_sem, prior):
        t = torch.randn(self.bs).to(self.device)
        input, target = self.preprocess_data(partial_sem, complete_sem, prior, t)
        u_t = self.model(input, t)
        loss = self.criterion(u_t, target)
        return loss 
    
    def preprocess_data_CA(self, partial_sem, complete_sem, prior, t):
        x0 = self.args.std * torch.randn_like(partial_sem)
        free_mask = self.compute_free_mask(partial_sem)
        x1 = self.augment(complete_sem, free_mask, prior)

        sample = self.path.sample(t = t, x_0 = x0, x_1 = x1)

        return sample.x_t, partial_sem, sample.dx_t
    
    def model_forward_CA(self, partial_sem, complete_sem, prior):
        t = torch.randn(self.bs).to(self.device)
        input, condition, target = self.preprocess_data_CA(partial_sem, complete_sem, prior, t)
        u_t  = self.model(input, condition, t)
        loss = self.criterion(u_t, target)
        return loss 

    def preprocess_data_PA(self, partial_sem, complete_sem, prior, t):
        free_mask = self.compute_free_mask(partial_sem)

        x0 = x0 = self.args.std * torch.randn_like(partial_sem)
        x1 = self.augment(complete_sem, free_mask, prior)

        sample = self.path.sample(t = t, x_0 = x0, x_1 = x1)

        return sample.x_t, partial_sem, ~free_mask.float(), sample.dx_t

    def model_forward_PA(self, partial_sem, complete_sem, prior):
        t = torch.randn(self.bs).to(self.device)
        input, condition, mask, target = self.preprocess_data_PA(partial_sem, complete_sem, prior, t)
        u_t = self.model(input, condition, mask, t)
        loss = self.criterion(u_t, target)
        
        return loss 
    
    def augment(self, complete_sem, free_mask, prior):
        if prior is not None:
            x1 = complete_sem
            x1[:, 2:, :, :][free_mask[:, 2:, :, :]] += self.args.prior_coeff * prior[free_mask[:, 2:, :, :]]
        else:
            x1 = complete_sem 
        
        return x1

    def compute_free_mask(self, partial_sem):
        obstacle_mask = torch.zeros(self.bs, 1, *partial_sem.shape[2:]).bool().to(self.device)
        for b in range(self.bs):
            obstacle_mask[b, 0] = torch.sum(partial_sem[b], dim = 0) > 0
            obstacle_mask[b, 0] = self.dilation(obstacle_mask[b, 0])
        free_mask = (~obstacle_mask).expand_as(partial_sem)

        return free_mask.bool()
    
    def dilation(self, mask):
        dilated_mask = F.conv2d(mask.unsqueeze(0).float(), self.kernel, padding = 1)
        return dilated_mask.squeeze(0).ge(1).float()
    
    def prepare(self):
        self.model = load.load_model(self.args).to(self.device)
        if self.args.distributed:
            self.setupDDP()
        self.setupEMA()

        self.optimizer = load.load_optimizer(self.args, self.model)
        self.dataset = load.load_dataset(self.args)
        self.data_loader = load.load_loader(self.args, self.rank, self.dataset)
        self.scheduler = load.load_scheduler(self.args, self.optimizer, len(self.data_loader))
        
        if self.ckpt is not None:
            self.model.load_state_dict(self.ckpt['model'])
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            self.scheduler.load_state_dict(self.ckpt['scheduler'])

        flow_scheduler = CondOTScheduler()
        self.path = AffineProbPath(scheduler = flow_scheduler)

        self.criterion = torch.nn.MSELoss(reduction = 'mean')

        self.loss_meter = AverageMeter()
        self.time_meter = AverageMeter()

        self.epoch_len = len(self.data_loader)
        self.total_len = (self.args.epochs - self.args.start_epoch) * self.epoch_len

        if self.main:
            self.logger = load.load_logger(self.args)
            self.writer = SummaryWriter(log_dir=os.path.join(self.args.log_dir, self.args.exp_name))
            self.logger.info(self.args)
            
    def setupDDP(self):
        dist.init_process_group(backend = 'nccl', world_size = self.args.world_size, 
                                        rank = self.rank)
        os.environ['RANK'] = str(self.rank)
        os.environ['WORLD_SIZE'] = str(self.args.world_size)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank], output_device=self.rank)

    def setupEMA(self):
        if self.args.distributed:
            self.ema = deepcopy(self.model.module).to(self.device)
        else:
            self.ema = deepcopy(self.model).to(self.device)
        requires_grad(self.ema, False)

    def save(self, epoch):
        state = {
            'epoch': epoch + 1,
            'model': self.model.state_dict(),
            'ema_model': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(), 
            'scheduler': self.scheduler.state_dict() 
        }
        save_path = os.path.join(self.args.log_dir, self.args.exp_name, f'goal-epoch{epoch+1}.pth')
        torch.save(state, save_path)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

