"""
@Date: 2022/9/23
@Description:
"""
import argparse
import sys
import os
import time

import cv2
import numpy as np
import torch

from tqdm import tqdm
from utils.misc import print_args
from utils.config import get_config
from utils.logger import get_logger, build_logger
from models.build import build_model
from dataset.build import build_loader
from pipline.misc import data_to_device
from visualization.show_depth import get_img_depth_rgb, get_depth_rgb
from utils.misc import tensor2np_d
from accuracy.build import build_acc
from visualization.show_pt import show_pt
from geometry.uv_pt import depth2pt
from visualization.viz import SLAMFrontend


def parse_option():
    parser = argparse.ArgumentParser(description='Pytorch implementation of DeepV2D')
    parser.add_argument('--cfg',
                        type=str,
                        default='src/config/kitti_0.yaml',
                        metavar='FILE',
                        help='path to config file')

    parser.add_argument('--bs',
                        type=int,
                        help='batch size')

    parser.add_argument('--device',
                        type=str,
                        help='device')

    parser.add_argument('--show2d',
                        action='store_true',
                        help='device')

    parser.add_argument('--show3d',
                        action='store_true',
                        help='device')

    parser.add_argument('--save_result',
                        action='store_true',
                        help='device')

    parser.add_argument('--ckpt_dir',
                        type=str,
                        help='ckpt dir')
    args = parser.parse_args()
    args.mode = 'test'
    args.debug = True if sys.gettrace() else False
    # args.debug = True
    print_args(args, parser.description)
    return args


def inference_depth(model, data_loader, save_dir, acc, save_result=False, show2d=False, show3d=False):
    bar = tqdm(enumerate(data_loader), ncols=100, total=len(data_loader))
    os.makedirs(save_dir, exist_ok=True)

    epoch_acc_d = {}

    # frontend = SLAMFrontend().start()

    for i, gt in bar:
        data_to_device(gt, model.device)
        dt = model(gt)
        acc_d = acc(gt, dt)

        for acc_k, acc_v in acc_d.items():
            if acc_k not in epoch_acc_d:
                epoch_acc_d[acc_k] = []
            epoch_acc_d[acc_k].append(acc_v)

        gt = tensor2np_d(gt)
        dt = tensor2np_d(dt)
        depths_gt = gt['depth'][:, 0]
        images_gt = gt['images'][:, 0]
        poses_gt = gt['poses'][:, 0]
        intrinsics_gt = gt['intrinsics']
        depths_dt = dt['depths'][-1]


        if show2d or show3d or save_result:
            for j, (depth_dt, image_gt, depth_gt, intrinsic_gt, pose_gt) in enumerate(
                    zip(depths_dt, images_gt, depths_gt, intrinsics_gt, poses_gt)):
                img_depth_rgb_gt = get_img_depth_rgb(image_gt, depth_gt, mode='all', show=False)
                depth_rgb_dt = get_depth_rgb(depth_dt, fill=False)
                img_depth_rgb = np.concatenate([img_depth_rgb_gt, depth_rgb_dt], axis=0)  # in height
                if save_result:
                    cv2.imwrite(os.path.join(save_dir, f'{i}_dt.jpg'), img_depth_rgb[..., ::-1])
                if show2d:
                    cv2.imshow(f'vis.jpg', img_depth_rgb[..., ::-1])
                    cv2.waitKey()
                if show3d:
                    key_pts = depth2pt(torch.from_numpy(depth_dt[None, None]), torch.from_numpy(intrinsic_gt)).numpy()[0, 0]
                    key_pts = key_pts[10:, ...].reshape(-1, 3)
                    color_data = image_gt.transpose(1, 2, 0)[10:, ...].reshape(-1, 3)
                    show_pt(key_pts, color_data)
                    # frontend.update_pose(i*data_loader.batch_size+j, pose_gt)
                    # frontend.update_points(i*data_loader.batch_size+j, key_pts, color_data)
                    # time.sleep(2)

    epoch_acc_d = dict(zip(epoch_acc_d.keys(), [np.array(epoch_acc_d[k]).mean() for k in epoch_acc_d.keys()]))
    print(epoch_acc_d)

def main():

    args = parse_option()
    config = get_config(args)
    logger = get_logger()

    torch.hub._hub_dir = 'ckpts'

    _, data_loader = build_loader(config, logger)
    model = build_model(config, logger)[0]
    acc = build_acc(config, logger)


    model.eval()

    save_dir = os.path.join(config.CKPT.RESULT_DIR, 'vis')
    logger.info(f'vis dir: {save_dir}')
    inference_depth(model, data_loader, save_dir, acc, args.save_result, args.show2d, args.show3d)


if __name__ == '__main__':
    main()
