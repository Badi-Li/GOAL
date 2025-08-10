import argparse 

def get_args():
    parser = argparse.ArgumentParser()

    # Training utils 
    parser.add_argument('--start_epoch', type = int, default = 0)
    parser.add_argument('--epochs', type = int, default = 25)
    parser.add_argument('--ema_decay', type = float, default = 0.999)
    parser.add_argument('--resume', type = str, default = None, 
                        help = 'load models from path to resume training')
    parser.add_argument('--distributed', action = 'store_false',
                        help = 'Enable DDP Training')
    parser.add_argument('--seed', type = int, default = 3407)

    # Data
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = 'Batch size per gpu')
    parser.add_argument('--num_workers', type = int, default = 8)
    parser.add_argument('--expand_ratio', type = float, default = 1.4, 
                        help = 'How much to expand the minimum bounding box while training')
    parser.add_argument('--prior', action = 'store_false',
                        help = 'whether to use llm prior')
    parser.add_argument('--llm', type = str, default = 'chatgpt', choices = ['chatglm', 'chatgpt', 'deepseek', 'llama'])
    parser.add_argument('--min_std', type = float, default = 20.)
    parser.add_argument('--max_std', type = float, default = 50.)
    parser.add_argument('--k', type = int, default = 5, 
                        help = 'Some LLMs output small distances for all other objects for specific central objects, select the top k')
    parser.add_argument('--dist_thr', type = float, default = 2.5)
    parser.add_argument('--conf_thr', type = float, default = 0.85)
    parser.add_argument('--prior_coeff', type = float, default = 1500., 
                        help = 'Weight for LLM prior')
    parser.add_argument('--std', type = float, default = 0.01, 
                        help = 'Standard deviation of gaussian noise added to free space of partial map')
    parser.add_argument('--data_root', type = str, default = './data/semantic_maps')
    parser.add_argument('--dataset', type = str, default = 'mp3d', choices = ['mp3d', 'hm3d', 'joint'])
    
    # Model (DiT-B/16)
    parser.add_argument('--input_size', type = int, default = 256)
    parser.add_argument('--in_channels', type = int, default = 23)
    parser.add_argument('--patch_size', type = int, default = 16)
    parser.add_argument('--depth', type = int, default =12)
    parser.add_argument('--num_heads', type = int, default = 12)
    parser.add_argument('--hidden_size', type = int, default = 768)
    parser.add_argument('--condition', type = str, default = None, choices = ['CA', 'PA'],
                        help = 'CA for cross-attention and PA for DiTPainter-like. The latter is not actually condition.')

    # Optimizer & Scheduler
    parser.add_argument('--base_lr', type = float, default = 0.00015)
    parser.add_argument('--weight_decay', type = float, default = 0.01)
    parser.add_argument('--final_lr', type = float, default = 0.00001)
    parser.add_argument('--warmup_steps', type =float, default = 0.075, 
                        help = '>1 for actual steps, <1 for a certain proportion of steps')
    parser.add_argument('--plateau_steps', type = float, default = 0.075,
                        help = '>1 for actual steps, <1 for a certain proportion of steps')



    # logging
    parser.add_argument('--log_dir', type = str, default = './logs')
    parser.add_argument('--exp_name', type = str, default = 'GOAL')
    parser.add_argument('--log_freq', type = int, default = 100,
                        help = 'Frequency of logging')
    parser.add_argument('--save_freq', type = int, default = 2)
    
    args = parser.parse_args()
    return args
