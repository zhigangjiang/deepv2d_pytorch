import torch
import numpy as np

from lietorch import SE3
from geometry.uv_pt import depth2pt, pt2uv, get_f_c, get_uv_grid
from einops import rearrange
from visualization.show_pt import show_pt


def se3_transform_depth(T: SE3, depths, intrinsics, min_depth=0.1, project_mask=True):
    """
    :param T: [b, f]
    :param depths: [b, f, h, w], key depths
    :param intrinsics: [b, 4(fx, fy, cx, cy)]
    :param min_depth: if < min_depth, it is invalid depth, will use self coords
    :param project_mask:
    :return:
    key_pts: [b, f, h, w, 2]
    project_pts： [b, f, h, w, 3]
    project_coors: [b, f, h, w, 2]
    mask： [b, f, h, w]
    """
    key_pts = depth2pt(depths, intrinsics)
    mask = key_pts[..., -1] >= min_depth
    project_pts = (T[:, :, None, None] if len(T.shape) > 1 else T[:, None, None]).act(key_pts)
    if project_mask:
        mask = mask & (project_pts[..., -1] > min_depth)  # z > min_depth [b, f, h, w]

    project_coors = pt2uv(project_pts, intrinsics, ~mask, min_depth)
    return key_pts, project_pts, project_coors, mask


def jac_local_pose(project_pts):
    """
    for least squares optimization, apply pose derivative
    :param project_pts: [b, f, h, w, 3]
    :return: jac_pose: [b, f, h, w, 3, 6]
    """
    x, y, z = project_pts[..., 0], project_pts[..., 1], project_pts[..., 2]
    o, i = torch.zeros_like(x), torch.ones_like(x)
    j1 = torch.stack([i, o, o], dim=-1)
    j2 = torch.stack([o, i, o], dim=-1)
    j3 = torch.stack([o, o, i], dim=-1)
    j4 = torch.stack([o, -z, y], dim=-1)
    j5 = torch.stack([z, o, -x], dim=-1)
    j6 = torch.stack([-y, x, o], dim=-1)
    jac_pose = torch.stack([j1, j2, j3, j4, j5, j6], dim=-1)
    return jac_pose


def jac_local_pt(project_pts, intrinsics, min_depth=0.1):
    """

    :param project_pts:  [b, f, h, w, 3]
    :param intrinsics:  [b, 4(fx, fy, cx, cy)]
    :param min_depth: not /0
    :return: jac_pt: [b, f, h, w, 2, 3]
    """
    fx, fy, cx, cy = get_f_c(intrinsics)
    x, y, z = project_pts[..., 0], project_pts[..., 1], project_pts[..., 2]

    o = torch.zeros_like(x)  # used to fill in zeros
    z_inv1 = torch.where(z < min_depth, torch.zeros_like(x), 1.0 / z)
    z_inv2 = torch.where(z < min_depth, torch.zeros_like(x), 1.0 / z ** 2)

    # jac_pt w.r.t (X, Y, Z)
    jac_pt = torch.stack([
        torch.stack([fx * z_inv1, o, -fx * x * z_inv2], dim=-1),
        torch.stack([o, fy * z_inv1, -fy * y * z_inv2], dim=-1)
    ], dim=-2)

    return jac_pt


def se3_transform_pt(T: SE3, key_pts, intrinsics, min_depth=0.1, project_mask=True):
    """

    :param T: [b, f]
    :param key_pts: [b, f, h, w, 3]
    :param intrinsics: [b, 4(fx, fy, cx, cy)]
    :param min_depth: if < min_depth, it is invalid depth, will use self coords
    :param project_mask:
    :return:
    project_pts： [b, f, h, w, 3]
    project_coors: [b, f, h, w, 2]
    jac: [b, f, h, w, 2, 6]
    mask： [b, f, h, w]
    """

    mask = key_pts[..., -1] > min_depth
    project_pts = T[:, :, None, None].act(key_pts)
    if project_mask:
        mask = mask & (project_pts[..., -1] > min_depth)  # z > min_depth [b, f, h, w]

    jac_pose = jac_local_pose(project_pts)
    project_coors = pt2uv(project_pts, intrinsics, ~mask, min_depth)
    jac_pt = jac_local_pt(project_pts, intrinsics, min_depth)
    jac = torch.einsum('...ij,...jk->...ik', jac_pt, jac_pose)

    return project_pts, project_coors, jac, mask


def induced_flow(T: SE3, depths, intrinsics, min_depth=0.1, project_mask=True, show=False):
    """
    :param T: [b, f]
    :param depths: [b, f, h, w], key depths
    :param intrinsics: [b, 4(fx, fy, cx, cy)]
    :param min_depth: if < min_depth, it is invalid depth, will use self coords
    :param project_mask:
    :param show:
    :return:
    flow: [b, f, h, w, 2]
    mask： [b, f, h, w] valid
    """
    u, v = get_uv_grid(depths.shape, device=depths.device)
    coords = torch.stack([u, v], dim=-1)

    key_pts = depth2pt(depths, intrinsics)
    mask = key_pts[..., -1] > min_depth
    if show:
        show_pt(key_pts[0, 0][mask[0, 0]].cpu().numpy())

    project_pts = T[:, :, None, None].act(key_pts)
    if project_mask:
        mask = mask & (project_pts[..., -1] > min_depth)  # z > min_depth [b, f, h, w]

    if show:
        show_pt(project_pts[0, 0][mask[0, 0]].cpu().numpy())

    project_coors = pt2uv(project_pts, intrinsics, ~mask, min_depth)
    flow = project_coors - coords
    return flow, mask


def test():
    import torch
    from dataset.kitti.kitti_dataset import KittiDataset

    split_path = {'test': 'src/dataset/kitti/debug_scenes_eigen.txt'}
    dataset = KittiDataset(data_dir='src/dataset/kitti', split_path=split_path, mode='test')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for data in data_loader:
        images = data['images']
        depths = data['depth']
        intrinsics = data['intrinsics']
        T = SE3.exp(torch.Tensor([[[0, 0, 0, 0, 0, 0]]]))
        pt = depth2pt(depths, intrinsics)
        # coors = pt2uv(pt, intrinsics)
        key_pts, project_pts, project_coors, mask = se3_transform_depth(T, depths, intrinsics)
        # show_pt(pt[0], images[0, 0])
        show_pt(project_pts[0, 0][mask[0, 0]], colors=images[0, 0].permute(1, 2, 0)[mask[0, 0]])


if __name__ == '__main__':
    test()
