"""
@date: 2022/10/6
@description:
"""
import torch
from utils.misc import ACCValue

class DepthACC:
    def __init__(self, min_depth=0, scale=0.1):
        self.min_depth = min_depth
        self.scale = scale

    def __call__(self, gt, dt):
        depths_gt = gt['depth'][:, 0] / self.scale
        depths_dt = dt['depths'][-1] / self.scale

        rmse = self.rmse(depths_gt, depths_dt)
        delta_1 = self.delta(depths_gt, depths_dt, k=1)
        return {
            'rmse': ACCValue(value=rmse.item(), lager_better=False, key_acc=True),
            'delta_1': ACCValue(value=delta_1.item(), lager_better=True),
        }

    def rmse(self, depths_gt, depths_dt):
        rmse_s = []
        for depth_dt, depth_gt in zip(depths_dt, depths_gt):
            mask = depth_gt > self.min_depth
            rmse = ((depth_dt[mask] - depth_gt[mask]) ** 2).mean() ** 0.5
            rmse_s.append(rmse)
        rmse_m = torch.tensor(rmse_s).mean()
        return rmse_m

    def delta(self, depths_gt, depths_dt, k):
        delta_s = []
        for depth_dt, depth_gt in zip(depths_dt, depths_gt):
            threshold = torch.max(depth_gt / depth_dt, depth_dt / depth_gt)
            delta = (threshold < 1.25 ** k).to(torch.float).mean()
            delta_s.append(delta)
        delta_m = torch.tensor(delta_s).mean()
        return delta_m