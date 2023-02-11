import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy import interpolate


def fill_depth(depth):
    x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                       np.arange(depth.shape[0]).astype("float32"))
    xx = x[depth > 0]
    yy = y[depth > 0]
    zz = depth[depth > 0]

    grid = interpolate.griddata((xx, yy), zz.ravel(),
                                (x, y), method='nearest')
    return grid


def get_img_depth_rgb(img, depth, max_depth=12, mode='all', show=False):
    # img: bgr
    if mode is None:
        depth_rgb = get_depth_rgb(depth, max_depth, fill=False)
    elif mode == 'fill':
        depth_rgb = get_depth_rgb(depth, max_depth, fill=True)
    elif mode == 'all':
        depth_rgb = np.concatenate([
            get_depth_rgb(depth, max_depth, fill=False),
            get_depth_rgb(depth, max_depth, fill=True)
        ], axis=0)
    else:
        raise NotImplemented

    img_depth_rgb = np.concatenate([img.transpose(1, 2, 0)[..., ::-1], depth_rgb], axis=0)  # in height
    if show:
        plt.imshow(img_depth_rgb)
        plt.show()
    return img_depth_rgb


def get_depth_rgb(depth, max_depth=12, fill=True, show=False):
    depth = fill_depth(depth) if fill else depth
    max_depth = depth.max() if max_depth is None else max(depth.max(), max_depth)
    cmap = plt.get_cmap('plasma')  # plasma, gray
    depth_rgb = (cmap((depth/max_depth).astype(np.float32))[..., :3] * 255).astype(np.uint8)
    if show:
        plt.imshow(depth_rgb)
        plt.show()
    return depth_rgb


def test():
    from dataset.kitti.kitti_dataset import KittiDataset
    from utils.misc import tensor2np_d
    from tqdm import tqdm

    split_path = {'test': 'src/dataset/kitti/debug_scenes_eigen.txt'}
    dataset = KittiDataset(data_dir='src/dataset/kitti', split_path=split_path, mode='test')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    bar = tqdm(data_loader)

    save_dir = os.path.join(dataset.data_dir, 'vis')
    os.makedirs(save_dir, exist_ok=True)
    for i, gt in enumerate(bar):
        gt = tensor2np_d(gt)
        depths_gt = gt['depth'][:, 0]
        images_gt = gt['images'][:, 0]
        for image_gt, depth_gt in zip(images_gt, depths_gt):
            img_depth_rgb = get_img_depth_rgb(image_gt, depth_gt, max_depth=None, mode='all', show=False)
            cv2.imwrite(os.path.join(save_dir, f'{i}.jpg'), img_depth_rgb[..., ::-1])


if __name__ == '__main__':
    test()
