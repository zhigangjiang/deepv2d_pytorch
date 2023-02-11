import os
import torch
import logging
from datetime import datetime, timezone, timedelta

from yacs.config import CfgNode as CN

_C = CN()
_C.DEBUG = False
# Run mode train/val
_C.MODE = 'train'
_C.TAG = 'default'
_C.COMMENT = 'add some comments to help you understand'
_C.SHOW_BAR = True
_C.MODEL = CN()
_C.MODEL.NAME = 'model_name'
_C.MODEL.ARGS = []
_C.MODEL.FINE_TUNE = []

# Motion modules
_C.MODEL.MOTION = CN()
# [0, 255] -> [-1, 1]
_C.MODEL.MOTION.RESCALE_IMAGES = False

# -----------------------------------------------------------------------------
# Output settings
# -----------------------------------------------------------------------------
_C.CKPT = CN()
_C.CKPT.PYTORCH = './ckpts'
_C.CKPT.ROOT = './ckpts'
_C.CKPT.DIR = os.path.join(_C.CKPT.ROOT, _C.MODEL.NAME, _C.TAG)
_C.CKPT.RESULT_DIR = os.path.join(_C.CKPT.DIR, 'results', _C.MODE)

_C.LOGGER = CN()
_C.LOGGER.DIR = os.path.join(_C.CKPT.DIR, "logs")
_C.LOGGER.LEVEL = logging.DEBUG

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = 'Dataset'
_C.DATASET.ARGS = []
_C.DATASET.NUM_WORKERS = 0
_C.DATASET.BATCH_SIZE = 2
_C.DATASET.PIN_MEMORY = True
# -----------------------------------------------------------------------------
# Running settings
# -----------------------------------------------------------------------------
_C.RUN = CN()
# GPU count
_C.RUN.WORLD_SIZE = 1
# Set cuda or cuda:1 or cpu
_C.RUN.DEVICE = 'cuda'
_C.RUN.DETERMINISTIC = True
_C.RUN.SEED = 123
_C.RUN.LOCAL_RANK = 0

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.SCRATCH = False
_C.TRAIN.BASE_LR = 1e-4
_C.TRAIN.WEIGHT_DECAY = 0
_C.TRAIN.RESUME_LAST = False
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 100
_C.TRAIN.SAVE_FREQ = 1
_C.TRAIN.SAVE_BEST = True

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Loss
_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.NAME = 'loss'
_C.TRAIN.LOSS.ARGS = [{}]

# Criterion
_C.TRAIN.CRITERION = CN()
_C.TRAIN.CRITERION.NAME = 'acc'
_C.TRAIN.CRITERION.ARGS = [{}]

# Visualization
_C.TRAIN.VIS = CN()
_C.TRAIN.VIS.NAME = 'vis'
_C.TRAIN.VIS.ARGS = [{}]


def set_dir(config, ckpt_dir=None):
    ckpt_dir_ = datetime.now().astimezone(timezone(timedelta(hours=8))).strftime('%y%m%d_%H%M%S') if ckpt_dir is None else ckpt_dir
    config.CKPT.DIR = os.path.join(config.CKPT.ROOT, config.MODEL.NAME, config.TAG + ('_debug' if (config.DEBUG and ckpt_dir is None) else ''), ckpt_dir_)
    config.CKPT.RESULT_DIR = os.path.join(config.CKPT.DIR, 'results', config.MODE)
    config.LOGGER.DIR = os.path.join(config.CKPT.DIR, "logs")

    if config.TRAIN.SCRATCH and os.path.exists(config.CKPT.DIR) and config.MODE == 'train':
        print(f"Train from scratch, delete checkpoint dir: {config.CKPT.DIR}")
        f = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(config.CKPT.DIR) if 'pkl' in f]
        if len(f) > 0:
            last_epoch = max(f)
            if last_epoch > 10:
                c = input(f"delete it (last_epoch: {last_epoch})?(Y/N)\n")
                if c != 'y' and c != 'Y':
                    exit(0)

        shutil.rmtree(config.CKPT.DIR, ignore_errors=True)

    os.makedirs(config.CKPT.DIR, exist_ok=True)
    os.makedirs(config.CKPT.RESULT_DIR, exist_ok=True)
    os.makedirs(config.LOGGER.DIR, exist_ok=True)


def set_device(config):
    if 'cuda' in config.RUN.DEVICE:
        if torch.cuda.is_available():
            if ':' not in config.RUN.DEVICE:
                c = torch.cuda.device_count()
                ids = ",".join([str(i) for i in range(c)]) if c > 1 else '0'
                config.RUN.DEVICE = f'cuda:{ids}'
            config.RUN.WORLD_SIZE = len(config.RUN.DEVICE.split(':')[-1].split(','))
        else:
            print(f"Cuda is not available(config is: {config.RUN.DEVICE}), will use cpu ...")
            config.RUN.DEVICE = "cpu"
            config.RUN.WORLD_SIZE = 1


def get_config(args=None):
    config = _C.clone()
    ckpt_dir = None
    if args:
        if 'cfg' in args and args.cfg:
            config.merge_from_file(args.cfg)

        if 'mode' in args and args.mode:
            config.MODE = args.mode

        if 'debug' in args:
            config.DEBUG = args.debug

        if 'bs' in args and args.bs is not None:
            config.DATASET.BATCH_SIZE = args.bs

        if 'device' in args and args.device:
            config.RUN.DEVICE = args.device

        if 'ckpt_dir' in args and args.ckpt_dir:
            ckpt_dir = args.ckpt_dir

    config.TAG = os.path.basename(args.cfg).split('.')[0] if config.TAG == 'default' else config.TAG
    set_dir(config, ckpt_dir)
    set_device(config)

    if config.MODE == 'train':
        with open(os.path.join(config.CKPT.DIR, "config.yaml"), "w") as f:
            f.write(config.dump(allow_unicode=True))
    config.freeze()

    return config


def get_rank_config(cfg, local_rank):
    local_rank = 0 if local_rank is None else local_rank
    config = cfg.clone()
    config.defrost()
    if config.RUN.WORLD_SIZE > 1:
        ids = config.RUN.DEVICE.split(':')[-1].split(',') if ':' in config.RUN.DEVICE else range(config.RUN.WORLD_SIZE)
        config.RUN.DEVICE = f'cuda:{ids[local_rank]}'

        if 'cuda' in config.RUN.DEVICE:
            torch.cuda.set_device(config.RUN.DEVICE)

    config.RUN.LOCAL_RANK = local_rank
    config.RUN.SEED = config.RUN.SEED + local_rank
    config.freeze()
    torch.hub._hub_dir = config.CKPT.PYTORCH
    return config
