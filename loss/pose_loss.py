"""
@Date: 2022/9/9
@Description:
"""
import torch.nn as nn
import torch

from geometry.transform import induced_flow
from einops import rearrange
from lietorch import SE3


class PoseLoss(nn.Module):
    def __init__(self, residual_w=0, weight_w=0, dt_project_mask=True):
        super().__init__()
        self.max_error = 100
        self.residual_w = residual_w
        self.weight_w = weight_w
        self.dt_project_mask = dt_project_mask
        pass

    def forward(self, gt, dt):
        Ts_gt, intrinsics = SE3.InitFromVec(gt['poses']), gt['intrinsics']
        transforms_dt, residuals_dt, weights_dt = dt['transforms'], dt['residuals'], dt['weights']
        depths = dt['depths']
        return self.__forward__(Ts_gt, depths, intrinsics, transforms_dt, residuals_dt, weights_dt)

    def __forward__(self, Ts_gt, depths, intrinsics, transforms_dt, residuals_dt, weights_dt):
        b, f = Ts_gt.shape

        ii = torch.tensor(([0] * (f - 1)), dtype=torch.int64)
        jj = torch.arange(1, f)

        Tij_gt = Ts_gt[:, jj] * Ts_gt[:, ii].inv()
        # pt.z+ --> forward, gt pose.z maybe retreat, so project_mask=True
        flow_gt, mask_gt = induced_flow(Tij_gt, depths, intrinsics, project_mask=True)

        loss = 0
        for i in range(len(transforms_dt)):
            Ts_dt = transforms_dt[i]
            Tij_dt = Ts_dt[:, jj] * Ts_dt[:, ii].inv()  # 0(j) 1(j) 2(ii) 3(j)...
            flow_dt, mask_dt = induced_flow(Tij_dt, depths, intrinsics, project_mask=self.dt_project_mask)
            if self.dt_project_mask:
                # dt pose.z maybe always retreat, mask all elements are False, then loss always is 0!
                mask = (mask_gt & mask_dt)[..., None].float()
            else:
                # pt.z+ --> forward, prevent dt pose.z always retreat, so dt_project_mask=False
                # gt mask is authentic
                mask = mask_gt[..., None].float()
            re_proj_diff = mask * torch.clip(torch.abs(flow_dt - flow_gt), -self.max_error, self.max_error)
            re_proj_loss = re_proj_diff.mean()

            if i > 0 and self.residual_w != 0:
                res_dt = residuals_dt[i - 1]
                weight_dt = weights_dt[i - 1]
                res_dt = torch.clip(res_dt, -self.max_error, self.max_error)
                res_dt = mask * weight_dt * res_dt ** 2
                res_loss = self.residual_w * res_dt.mean()
            else:
                res_loss = 0

            loss += re_proj_loss + res_loss

        if self.weight_w != 0:
            # encourage larger weights
            weights_dt = torch.stack(weights_dt, dim=0)
            weights_dt = rearrange(weights_dt, 'i b f h w d -> (i b f d) (h w)')
            top_v = torch.topk(weights_dt, k=2048).values
            target = torch.ones_like(top_v)
            weights_loss = torch.nn.functional.binary_cross_entropy_with_logits(top_v, target, reduction='mean')
            loss += self.weight_w * weights_loss

        return loss, {'loss': loss.item()}
