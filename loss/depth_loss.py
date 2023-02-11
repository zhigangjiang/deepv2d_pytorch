"""
@Date: 2022/9/23
@Description:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import ACCValue


class DepthLoss(nn.Module):
    def __init__(self, smooth_w=0.02):
        super().__init__()
        self.smooth_w = smooth_w
        self.min_depth = 0

    def forward(self, gt, dt):
        depth_gt = torch.squeeze(gt['depth'], dim=1)  # b h w
        depths_dt = dt['depths']  # [[b h w] ...]
        return self.__forward__(depth_gt, depths_dt)

    def __forward__(self, depth_gt, depths_dt):
        mask = depth_gt > self.min_depth
        valid = mask.to(torch.float)
        s = 1.0 / (torch.mean(valid) + 1e-8)
        total_loss = 0.0

        for i, depth_dt in enumerate(depths_dt):
            loss = 0
            if self.smooth_w != 0:
                gx = depth_dt[:, :, 1:] - depth_dt[:, :, :-1]
                gy = depth_dt[:, 1:, :] - depth_dt[:, :-1, :]
                vx = valid[:, :, 1:] * valid[:, :, :-1]
                vy = valid[:, 1:, :] * valid[:, :-1, :]
                loss += self.smooth_w * (
                        torch.mean((1 - vx) * torch.abs(gx)) +
                        torch.mean((1 - vy) * torch.abs(gy))
                )
            loss_depth = s * torch.mean(torch.abs(depth_gt[mask] - depth_dt[mask]))

            loss += loss_depth
            w = .5 ** (len(depths_dt) - i - 1)
            total_loss += w * loss

        return total_loss, {'loss': ACCValue(value=total_loss.item(), lager_better=False)}
