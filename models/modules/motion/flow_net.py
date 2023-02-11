"""
@Date: 2022/9/7
@Description:
"""
import numpy as np
import torch
import torch.nn as nn

from einops import rearrange


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 32, (7, 7), stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, (7, 7), stride=1, padding=3),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (5, 5), stride=1, padding=2),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        :param x: [b*f, c(128), h, w]
        :return: x1:c=32 x2:c=64 x3:c=128 x4:c=256 x5:c=512
        """
        x1 = self.layer1(x)  # c=32
        x2 = self.layer2(x1)  # c=64
        x3 = self.layer3(x2)  # c=128
        x4 = self.layer4(x3)  # c=256
        x5 = self.layer5(x4)  # c=512
        return [x1, x2, x3, x4, x5]


class Decoder(nn.Module):
    def __init__(self, layer5_output_padding=(0, 0)):
        super().__init__()

        self.up_layer5 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, (3, 3), stride=2, padding=1, output_padding=layer5_output_padding),
            nn.ReLU(inplace=True),
        )
        # concat with layer4
        self.up_layer4 = nn.Sequential(
            nn.Conv2d(512, 256, (3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(256, 128, (3, 3), stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )

        # concat with layer3
        self.up_layer3 = nn.Sequential(
            nn.Conv2d(256, 128, (3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 64, (3, 3), stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )

        # concat with layer2
        self.up_layer2 = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )

        # concat with layer1
        self.up_layer1 = nn.Sequential(
            nn.Conv2d(64, 32, (3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, 32, (3, 3), stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.flow_layer = nn.Conv2d(32, 2, (3, 3), stride=1, padding=1)
        self.weight_layer = nn.Sequential(
            nn.Conv2d(32, 2, (3, 3), stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, fs):
        """
        :param fs: x1:c=32 x2:c=64 x3:c=128 x4:c=256 x5:c=512
        :return:
        """
        x1, x2, x3, x4, x5 = fs

        x = self.up_layer5(x5)  # c=256
        x = torch.cat([x, x4], dim=-3)  # c=512
        x = self.up_layer4(x)  # c=128
        x = torch.cat([x, x3], dim=-3)  # c=256
        x = self.up_layer3(x)  # c=64
        x = torch.cat([x, x2], dim=-3)  # c=128
        x = self.up_layer2(x)  # c=32
        x = torch.cat([x, x1], dim=-3)  # c=64
        x = self.up_layer1(x)  # c=32
        x = self.output_layer(x)  # c=32
        flow = self.flow_layer(x)  # c=2
        weight = self.weight_layer(x)  # c=2
        return flow, weight


class FlowNet(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(layer5_output_padding=((feat_size % (2 ** 5)) == 0).astype(int))

    def forward(self, f1, f2):
        """
        :param f1: [b, f, c, h, w], key_feats
        :param f2: [b, f, c, h, w], other_feats project on key_feats
        :return:
        """
        x = torch.cat((f1, f2), dim=-3)  # concat at c dim, then c=2c
        b = x.shape[0]
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.encoder(x)
        flow, weight = self.decoder(x)
        flow = rearrange(flow, '(b f) c h w -> b f h w c', b=b)
        weight = rearrange(weight, '(b f) c h w -> b f h w c', b=b)
        return flow, weight


def test():
    b, f, c, h, w = 2, 3, 64, 48, 272
    f1 = torch.rand((b, f, c, h, w), dtype=torch.float)
    f2 = torch.rand((b, f, c, h, w), dtype=torch.float)
    flow_net = FlowNet(feat_size=np.array([h, w]))
    flow, weight = flow_net(f1, f2)


if __name__ == '__main__':
    test()
