import os
import torch
import random
import numpy as np
from munch import Munch

def seed_everything(seed: int):
    """Seed all RNGs

    Args:
        seed (int): seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_optimizer(optimizer):
    return getattr(torch.optim, optimizer)

def parse_args(args, **kwargs):
    args = Munch({'epoch': 0}, **args)
    kwargs = Munch({'no_cuda': False, 'debug': False}, **kwargs)
    args.wandb = not kwargs.debug and not args.debug
    args.device = 'cuda' if torch.cuda.is_available() and not kwargs.no_cuda else 'cpu'
    args.max_dimensions = [args.max_width, args.max_height]
    args.min_dimensions = [args.get('min_width', 32), args.get('min_height', 32)]
    if 'decoder_args' not in args or args.decoder_args is None:
        args.decoder_args = {}
    if 'model_path' in args:
        args.out_path = os.path.join(args.model_path, args.name)
        os.makedirs(args.out_path, exist_ok=True)
    return args

