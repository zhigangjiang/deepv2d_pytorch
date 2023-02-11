import torch.nn as nn
import torch
import torchvision.models as models

from models.modules.conv import ResConv2D
from models.modules.hourglass import Hourglass2D
from einops import rearrange


class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        self.out_dim = 32
        self.downscale = 4

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        self.conv2 = nn.Conv2d(self.encoder.layer3[-1].conv1.in_channels, self.out_dim, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.conv2(x)
        return x


class DownSample(nn.Module):
    def __init__(self, downscale):
        super().__init__()
        self.out_dim = 3
        self.downscale = downscale
        self.down_sample = nn.Upsample(scale_factor=1/self.downscale, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.down_sample(x)
        return x


class ResHGNet(nn.Module):
    def __init__(self, downscale=4):
        super().__init__()

        self.downscale = downscale
        self.out_dim = 32
        self.layer1 = torch.nn.Conv2d(3, 32, (7, 7), stride=int(downscale/2), padding=3)
        self.layer2 = torch.nn.Sequential(
            ResConv2D(32, 32, 1),
            ResConv2D(32, 32, 1),
            ResConv2D(32, 32, 1),
            ResConv2D(32, 64, 2),
            ResConv2D(64, 64, 1),
            ResConv2D(64, 64, 1),
            ResConv2D(64, 64, 1)
        )
        self.layer3 = torch.nn.Sequential(
            Hourglass2D(64, 4),
            Hourglass2D(64, 4)
        )
        self.layer4 = torch.nn.Conv2d(64, self.out_dim, (1, 1), stride=1)

    def forward(self, x):
        """
        :param x: [b*f, c(3), h, w]
        :return:
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, backbone='res_hg', downscale=4):
        super().__init__()
        if backbone == 'res_hg':
            self.extractor = ResHGNet(downscale)
        elif backbone == 'down_sample':
            self.extractor = DownSample(downscale)
        else:
            self.extractor = ResNet(backbone, pretrained=True)
        self.downscale = self.extractor.downscale
        self.out_dim = self.extractor.out_dim

    def forward(self, x):
        """
        :param x: images [b, f, c(3), h, w]
        :return:
        """
        b = x.shape[0]
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.extractor(x)
        x = rearrange(x, '(b f) c h w -> b f c h w', b=b)
        return x
