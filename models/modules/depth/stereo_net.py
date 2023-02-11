import torch
import torch.nn as nn
import torch.nn.functional as F

from lietorch import SE3
from geometry.transform import se3_transform_depth
from einops import repeat, reduce, rearrange
from models.modules.hourglass import Hourglass3D
from models.modules.conv import ResConv3D
from utils.misc import coords_normal


def project_depth_volume(T, feats, depths_vec, intrinsics):
    """
    :param T: [b f]
    :param feats: [b, f, 32(c), h, w]
    :param depths_vec: [b, f, 32(d), h, w] 32(d) = cost_volume
    :param intrinsics: [b, 4(fx, fy, cx, cy)]
    :return: volumes [b f c d h w]
    """
    (b, f, c, h, w), d = feats.shape, depths_vec.shape[0]
    depths = repeat(depths_vec.to(feats.device), 'd -> b f d h w', b=b, f=f, h=h, w=w)  # [b, f, 32(d), h, w]
    project_coors = se3_transform_depth(
        SE3.InitFromVec(repeat(T.data, 'b f q -> (b d) f q', d=d).contiguous()),  # q = 7 (tx ty tz 4quat)
        rearrange(depths, 'b f d h w -> (b d) f h w'),
        repeat(intrinsics, 'b i -> (b d) i', d=d),  # i = 4
        min_depth=0.1, project_mask=False)[2]


    volumes = rearrange(F.grid_sample(
        repeat(feats, 'b f c h w -> (b f d) c h w', d=d),  # d = cost_volume c = feat_dim
        rearrange(coords_normal(project_coors), '(b d) f h w u -> (b f d) h w u', b=b),  # d = cost_volume u = 2(u, v)
        mode='bilinear', align_corners=True), '(b f d) c h w -> b f c d h w', b=b, f=f)
    return volumes


class ConcatBlock(nn.Module):
    def __init__(self, depths_vec, in_dim, dim):
        super().__init__()
        self.depths_vec = depths_vec
        self.conv = torch.nn.Conv3d(in_dim, dim, kernel_size=3, stride=1, padding=1)
        self.res_conv = ResConv3D(dim, dim)

    def forward(self, Ts, feats, intrinsics):
        """
        :param Ts:
        :param feats: [b, f, 32(c), h, w]
        :param intrinsics: [b, 4(fx, fy, cx, cy)]
        :return:
        """
        b, f = feats.shape[:2]
        ii = torch.tensor(([0] * f), dtype=torch.int64)
        jj = torch.arange(0, f)
        Tij = Ts[:, jj] * Ts[:, ii].inv()  # [b f]
        x = rearrange(project_depth_volume(Tij, feats, self.depths_vec, intrinsics), 'b f c d h w -> b (f c) d h w')
        x = self.conv(x)
        x = self.res_conv(x)
        return x


class AvgBlock(nn.Module):
    def __init__(self, depths_vec, dim=32):
        super().__init__()
        self.depths_vec = depths_vec
        self.conv = torch.nn.Conv3d(dim * 2, dim, kernel_size=(1, 1, 1), stride=1)
        self.res_conv = ResConv3D(dim, dim)

    def forward(self, Ts, feats, intrinsics):
        """
        :param Ts:
        :param feats: [b, f, 32(c), h, w]
        :param intrinsics: [b, 4(fx, fy, cx, cy)]
        :return: [b c d h w]
        """
        b, f = feats.shape[:2]

        ii = torch.tensor(([0] * (f - 1)), dtype=torch.int64)
        jj = torch.arange(1, f)

        Tii = Ts[:, ii] * Ts[:, ii].inv()  # b f-1
        Tij = Ts[:, jj] * Ts[:, ii].inv()  # b f-1
        x = rearrange(torch.cat([
            project_depth_volume(Tii, feats[:, ii], self.depths_vec, intrinsics),  # b f-1 c d h w
            project_depth_volume(Tij, feats[:, jj], self.depths_vec, intrinsics)  # b f-1 c d h w
        ], dim=-4), 'b f c d h w -> (b f) c d h w')  # b*f-1 2c d h w
        x = self.conv(x)
        x = self.res_conv(x)
        x = reduce(x, '(b f) c d h w -> b c d h w', b=b, reduction='mean')  # view pooling: avg
        return x


class HGHead(nn.Module):
    def __init__(self, image_size, dim, hg_expand=32):
        super().__init__()
        self.image_size = image_size
        self.hg = Hourglass3D(dim, 4, expand=hg_expand)
        self.stereo_head = nn.Sequential(  # stereo_head
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, 1, kernel_size=(1, 1, 1), stride=1, padding=0),
        )
        self.up_sample = nn.Upsample(size=list(self.image_size), mode='bilinear', align_corners=False)

    def forward(self, x):
        """
        :param x: [b c d h w]
        :return: [b d oh ow]
        """
        x = self.hg(x)
        y = self.stereo_head(x)  # b c(1) d h w
        y = torch.squeeze(y, 1)  # b d h w
        y = self.up_sample(y)
        return x, y


class StereoNet(nn.Module):
    def __init__(self, image_size, mode, dim, hg_count=2, seq_len=3):
        super().__init__()
        self.min_depth = 0.1
        self.max_depth = 8.0
        self.cost_volume = 32
        self.depths_vec = torch.linspace(self.min_depth, self.max_depth, self.cost_volume)

        self.image_size = image_size
        self.mode = mode
        if self.mode == 'concat':
            self.project_block = ConcatBlock(self.depths_vec, in_dim=dim * seq_len, dim=dim)
        elif self.mode == 'avg':
            self.project_block = AvgBlock(self.depths_vec, dim=dim)
        else:
            raise NotImplementedError(self.mode)
        self.hg_heads = nn.Sequential(*[HGHead(self.image_size, dim) for _ in range(hg_count)])
        self.softmax = nn.Softmax(dim=1)  # DiffArgmax

    def forward(self, Ts, feats, intrinsics):
        """
        :param Ts:
        :param feats: [b, f, 32(c), h, w]
        :param intrinsics: [b, 4(fx, fy, cx, cy)]
        :return: depths: [[b h w], ...]
        """
        x = self.project_block(Ts, feats, intrinsics)  # [b c d h w]

        depths = []
        for hg_head in self.hg_heads:
            x, pred_logit = hg_head(x)
            prob_volume: torch.Tensor = self.depths_vec[None, :, None, None].to(pred_logit.device) * self.softmax(
                pred_logit)
            depth = torch.sum(prob_volume, axis=1)  # b h w
            depths.append(depth)

        depths = torch.stack(depths, dim=0)
        return depths
