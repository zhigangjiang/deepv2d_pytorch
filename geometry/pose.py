import numpy
import torch


def pose_vec2mat(vec, use_filler=True):
    """
    :param vec: 6DoF [b, 6(tx, ty, tz, rx, ry, rz)]
    :param use_filler: add vec[1, 4](0, 0, 0, 1) to bottom of transform_mat[3, 4]
    :return: transformation matrix [b, 4, 4] if use_filler else [b, 3, 4]
    """
    b = vec.shape[0]
    transl = vec[:, :3]
    rot = vec[:, 3:]
    rot_mat = euler2mat(rot)
    rot_mat = torch.squeeze(rot_mat, dim=1)
    transform_mat = torch.cat([rot_mat, transl[..., None]], dim=-1)
    if use_filler:
        filler_vec = torch.tensor([0, 0, 0, 1])[None, None].repeat(b, 1, 1)
        transform_mat = torch.cat([transform_mat, filler_vec], dim=1)
    return transform_mat


def euler2mat(rot):
    """
    https://zhuanlan.zhihu.com/p/144032401
    :param rot: [b, 3(rx, ry, rz)]
    :return:
    """
    b = rot.shape[0]
    rot = torch.clamp(rot, -numpy.pi, numpy.pi)
    x, y, z = rot[:, [0]], rot[:, [1]], rot[:, [2]]

    zeros = torch.zeros([b, 1])
    ones = torch.ones([b, 1])

    cos_z = torch.cos(z)
    sin_z = torch.sin(z)
    z_mat = torch.stack([
        torch.cat([cos_z, -sin_z, zeros], dim=-1),
        torch.cat([sin_z, cos_z, zeros], dim=-1),
        torch.cat([zeros, zeros, ones], dim=-1)
    ], dim=1)

    cos_y = torch.cos(y)
    sin_y = torch.sin(y)
    y_mat = torch.stack([
        torch.cat([cos_y, zeros, sin_y], dim=-1),
        torch.cat([zeros, ones, zeros], dim=-1),
        torch.cat([-sin_y, zeros, cos_y], dim=-1)
    ], dim=1)

    cos_x = torch.cos(x)
    sin_x = torch.sin(x)
    x_mat = torch.stack([
        torch.cat([ones, zeros, zeros], dim=-1),
        torch.cat([zeros, cos_x, -sin_x], dim=-1),
        torch.cat([zeros, sin_x, cos_x], dim=-1)
    ], dim=1)

    rotMat = x_mat.matmul(y_mat).matmul(z_mat)
    return rotMat
