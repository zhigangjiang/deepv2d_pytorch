"""
@Date: 2022/9/8
@Description:
e.g. point feat1(u, v), it's corresponding point is feat2(u+x1, v+y1), then feat1(u, v)=feat2(u+x1, v+y1)
the point projecting is (u+x2, v+y2) by predicting pose (above coors), take feat2(u+x2, v+y2)
the flow is (x2-x2, y2-y1) between feat1(u, v)=feat2(u+x1, v+y1) and (u+x2, v+y2)
"""
import torch

from lietorch import SE3
from geometry.transform import se3_transform_pt
from einops import rearrange


class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        """
        :param ctx:
        :param H: [b, f, 6, 6]
        :param b: [b, f, 6]
        :return:
        """
        chol = torch.linalg.cholesky(H)
        xx = torch.cholesky_solve(b, chol)
        ctx.save_for_backward(chol, xx)
        return xx

    @staticmethod
    def backward(ctx, grad_output):
        chol, xx = ctx.saved_tensors
        dz = torch.cholesky_solve(chol, grad_output)
        xs = torch.squeeze(xx, -1)
        zs = torch.squeeze(dz, -1)
        dH = -torch.einsum('...i,...j->...ij', xs, zs)
        return dH, dz


def solve_pose(H, b, max_update=1.0):
    u = torch.linalg.cholesky(H)
    pose_vec = torch.cholesky_solve(b, u)
    pose_vec = torch.where(torch.isnan(pose_vec), torch.zeros_like(pose_vec), pose_vec)
    pose_vec = pose_vec[..., 0]
    return pose_vec


def keyframe_optim(Tij: SE3, flow, weight, project_coors, key_pts, intrinsics, num_iters=2):
    """
    optimize function: f(*Tij) = flow - (*Tij(depth) - Tij(depth))
    flow=(x2-x2, y2-y1)| Tij(depth)=(u+x2, v+y2)=coors | *Tij(depth)=(key_pts, pose)
    optimize pose to *Tij(depth)=(u+x1, v+y1) by Gauss-Newton iterations
    :param Tij: [b, f]
    :param flow: [b, f, h, w, 2]
    :param weight: [b, f, h, w, 2] in range(0, 1)
    :param project_coors: [b, f, h, w, 2]
    :param key_pts: [b, f, h, w, 3]
    :param intrinsics: [b, 4(fx, fy, cx, cy)]
    :param num_iters:
    :return:
    """
    lm_lambda = 0.0001
    ep_lambda = 100.0
    max_update = 1.0
    # optimize function: f(pose) = flow - (project_coors(key_pts, pose) - project_coors)
    for k in range(num_iters):
        #  Tij -> pose_k
        project_pts_k, project_coors_k, jac_k, mask_k = se3_transform_pt(Tij, key_pts, intrinsics)
        J_k = (mask_k[..., None, None].float() * weight[..., None] * jac_k).to(torch.float64)

        # Approximation of Hessian matrix H using Jacobian matrix J: J(pose_k)J(pose_k)^T=H(pose_k)
        H_k = torch.einsum('bf...j,bf...k->bfjk', J_k, J_k)
        # optimize function
        f_k = flow - (project_coors_k - project_coors).to(torch.float64)
        b_k = torch.einsum('bf...j,bf...->bfj', J_k, f_k)[..., None]
        H_k = H_k + ep_lambda * torch.eye(6, device=H_k.device) + lm_lambda * torch.eye(6, device=H_k.device)

        upsilon_k = solve_pose(H_k, b_k, max_update).to(torch.float32)

        T_upsilon_k = SE3.exp(upsilon_k)
        Tij = T_upsilon_k * Tij

    return Tij
