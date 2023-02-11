import torch.nn as nn
import torch
from models.modules.conv import ResConv2D, Conv2D, Conv3D, ResConv3D


def upnn3d(x, y, sc=2):
    dim = list(x.shape)[1]
    bx, _, hx, wx, dx = list(x.shape)
    by, _, hy, wy, dy = list(y.shape)

    x1 = torch.reshape(torch.tile(x, [1, 1, sc, sc, sc]), [bx, dim, sc * hx, sc * wx, sc * dx])
    if not (sc * hx == hy and sc * wx == wy):
        x1 = x1[:, :hy, :wy]

    return x1


class Hourglass2D(nn.Module):
    def __init__(self, dim, n, stride=1, expand=64):
        super().__init__()
        self.stride = stride
        self.n = n
        self.expand = expand
        self.dim = dim
        self.dim_expand = self.dim + self.expand

        self.res_conv = ResConv2D(self.dim, self.dim)
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            Conv2D(self.dim, self.dim_expand)
        )
        if n > 1:
            self.hourglass_next = Hourglass2D(self.dim_expand, n - 1)
        else:
            self.conv_last = ResConv2D(self.dim_expand, self.dim_expand)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = Conv2D(self.dim_expand, self.dim)

    def forward(self, x):
        x = self.res_conv(x)
        y = self.pool(x)
        if self.n > 1:
            y = self.hourglass_next(y)
        else:
            y = self.conv_last(y)
        y = self.conv(y)
        y = self.up_sample(y)
        y = x + y
        return y


class Hourglass3D(torch.nn.Module):
    def __init__(self, dim, n, stride=1, expand=32):
        super().__init__()
        self.stride = stride
        self.n = n
        self.expand = expand
        self.dim = dim
        self.dim_expand = self.dim + self.expand

        self.res_conv = ResConv3D(self.dim, self.dim)
        self.pool = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, padding=1, stride=2),
            Conv3D(self.dim, self.dim_expand)
        )
        if n > 1:
            self.hourglass_next = Hourglass3D(self.dim_expand, n - 1, expand)
        else:
            self.conv_last = ResConv3D(self.dim_expand, self.dim_expand)

        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = Conv3D(self.dim_expand, self.dim)

    def forward(self, x):
        x = self.res_conv(x)
        y = self.pool(x)
        if self.n > 1:
            y = self.hourglass_next(y)
        else:
            y = self.conv_last(y)
        y = self.conv(y)
        y = self.up_sample(y)
        y = x + y
        return y
