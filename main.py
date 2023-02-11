import sys
import argparse
import os
import numpy as np
import shutil
import torch
import torch.multiprocessing as mp

from utils.config import get_config, get_rank_config
from utils.init_env import init_env
from utils.logger import build_logger
from utils.misc import print_args
from pipline.train import Trainer


def parse_option():
    parser = argparse.ArgumentParser(description='Pytorch implementation of DeepV2D')
    parser.add_argument('--cfg',
                        type=str,
                        default='src/config/depth/kitti_10.yaml',
                        metavar='FILE',
                        help='path to config file')

    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'val', 'test'],
                        help='train/val/test mode')

    parser.add_argument('--bs',
                        type=int,
                        help='batch size')

    parser.add_argument('--device',
                        type=str,
                        help='device')

    parser.add_argument('--ckpt_dir',
                        type=str,
                        help='ckpt dir')

    args = parser.parse_args()
    args.debug = True if sys.gettrace() else False
    print_args(args, parser.description)
    return args


def main():
    args = parse_option()
    config = get_config(args)

    if config.RUN.DEVICE == 'cpu' or config.RUN.WORLD_SIZE < 2:
        print(f"Use single process, device: {config.RUN.DEVICE}")
        main_worker(0, config)
    else:
        print(f"Use {config.RUN.WORLD_SIZE} processes ...")
        mp.spawn(main_worker, nprocs=config.RUN.WORLD_SIZE, args=(config,), join=True)


def main_worker(local_rank, cfg):
    config = get_rank_config(cfg, local_rank)
    logger = build_logger(config)
    logger.info(f"Checkpoint dir: {config.CKPT.DIR}")
    logger.info(f"Comment: {config.COMMENT}")
    cur_pid = os.getpid()
    logger.info(f"Current process id: {cur_pid}")
    init_env(config)

    trainer = Trainer(config, logger, run_mode=config.MODE)

    if config.MODE == 'train':
        trainer.train()
    else:
        trainer.val()


if __name__ == '__main__':
    main()
