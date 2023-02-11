import numpy as np
import sys
import torch
import torchvision

from einops import rearrange


def intrinsics_vec_to_matrix(k_vec):
    fx, fy, cx, cy = torch.unbind(k_vec, dim=-1)
    z = torch.zeros_like(fx)
    o = torch.ones_like(fx)

    K = torch.stack([fx, z, cx, z, fy, cy, z, z, o], dim=-1)
    K = torch.reshape(K, list(k_vec.shape)[:-1] + [3, 3])
    return K


def intrinsics_matrix_to_vec(kmat):
    fx = kmat[..., 0, 0]
    fy = kmat[..., 1, 1]
    cx = kmat[..., 0, 2]
    cy = kmat[..., 1, 2]
    return torch.stack([fx, fy, cx, cy], dim=-1)


def update_intrinsics(intrinsics, delta_focal):
    kvec = intrinsics_matrix_to_vec(intrinsics)
    fx, fy, cx, cy = torch.unbind(kvec, dim=-1)
    df = torch.squeeze(delta_focal, -1)

    # update the focal lengths
    fx = torch.exp(df) * fx
    fy = torch.exp(df) * fy

    kvec = torch.stack([fx, fy, cx, cy], dim=-1)
    kmat = intrinsics_vec_to_matrix(kvec)
    return kmat



