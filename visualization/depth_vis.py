"""
@time: 2022/10/05
@description:
"""
import numpy as np
from visualization.show_depth import get_img_depth_rgb, get_depth_rgb
from utils.misc import tensor2np_d


class DepthVIS:
    def __init__(self, show_indexes=None, max_b=1):
        if show_indexes is None:
            show_indexes = [10, 40]
        self.show_indexes = show_indexes
        self.max_b = max_b

    def __call__(self, gt, dt):

        gt = tensor2np_d(gt)
        dt = tensor2np_d(dt)

        images_gt = gt['images'][:, 0][:self.max_b]
        depths_gt = gt['depth'][:, 0][:self.max_b]
        depths_dt = dt['depths'][-1][:self.max_b]
        imgs = []
        for depth_dt, image_gt, depth_gt in zip(depths_dt, images_gt, depths_gt):
            img_depth_rgb_gt = get_img_depth_rgb(image_gt, depth_gt, mode='all', show=False)
            depth_rgb_dt = get_depth_rgb(depth_dt, fill=False)
            img_depth_rgb = np.concatenate([img_depth_rgb_gt, depth_rgb_dt], axis=0)  # in height
            imgs.append(img_depth_rgb.transpose(2, 0, 1))

        imgs = np.array(imgs)
        return {
            'imgs': imgs,
        }
