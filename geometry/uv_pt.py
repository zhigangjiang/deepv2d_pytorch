import torch


def get_uv_grid(shape, device):
    b, f, h, w = shape
    v, u = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    u = u[None, None].repeat(b, f, 1, 1)
    v = v[None, None].repeat(b, f, 1, 1)
    return u.to(device), v.to(device)


def get_f_c(intrinsics):
    intrinsics = intrinsics.reshape(-1, 1, 1, 1, 4)
    fx, fy, cx, cy = intrinsics[..., 0], intrinsics[..., 1], intrinsics[..., 2], intrinsics[..., 3]
    return fx, fy, cx, cy


def depth2pt(depths, intrinsics):
    """

    :param depths: [b, f(1), h, w]
    :param intrinsics: [b, fx(4, fx, fy, cx, cy)]
    :return: 3d point: [b, f(1), h, w, 3]
    """
    u, v = get_uv_grid(depths.shape, depths.device)
    fx, fy, cx, cy = get_f_c(intrinsics)
    z = depths
    x = z * (u - cx) / fx
    y = z * (v - cy) / fy
    pc = torch.stack([x, y, z], dim=-1)
    return pc


def pt2uv(pt, intrinsics, invalid_mask=None, min_depth=0.1):
    """
    :param pt: [b, f, h, w, 3]
    :param intrinsics: [b, 4(fx, fy, cx, cy)]
    :param invalid_mask: [b, f, h, w]
    :param min_depth: 0.1
    :return: 2d coord: [b, f, h, w, 2]
    """
    x, y, z = pt[..., 0], pt[..., 1], pt[..., 2]
    fx, fy, cx, cy = get_f_c(intrinsics)

    z = torch.maximum(z, torch.tensor([min_depth], device=z.device))  # fill min_depth to temp=min_depth calculate
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    if invalid_mask is not None and invalid_mask.sum() != 0:
        u_, v_ = get_uv_grid(pt.shape[:-1], pt.device)
        u[invalid_mask] = u_[invalid_mask].float()
        v[invalid_mask] = v_[invalid_mask].float()

    coords = torch.stack([u, v], dim=-1)
    return coords


def test():
    from dataset.kitti.kitti_dataset import KittiDataset
    from visualization.show_pt import show_pt

    split_path = {'test': 'src/dataset/kitti/test_scenes_eigen_local.txt'}
    dataset = KittiDataset(data_dir='src/dataset/kitti', split_path=split_path, mode='test')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for data in data_loader:
        images = data['images']
        depths = data['depth']
        intrinsics = data['intrinsics']
        pt = depth2pt(depths, intrinsics)
        coords = pt2uv(pt, intrinsics)
        show_pt(pt[0])


if __name__ == '__main__':
    test()
