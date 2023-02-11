import torch
import torch.nn as nn

from models.modules.conv import ResConv2D
from einops import rearrange


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 32, (7, 7), stride=2, padding=3)

        self.layer2 = nn.Sequential(
            ResConv2D(32, 32, 1),
            ResConv2D(32, 32, 1),
            ResConv2D(32, 32, 1),
            ResConv2D(32, 64, 2)
        )

        self.layer3 = nn.Sequential(
            ResConv2D(64, 64, 1),
            ResConv2D(64, 64, 1),
            ResConv2D(64, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (1, 1), stride=1)
        )
        self.downscale = 4

    def forward(self, x):
        """
        :param x: [b, f, c(3), h, w]
        :return:
        """
        b = x.shape[0]
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = rearrange(x, '(b f) c h w -> b f c h w', b=b)
        return x


def test():
    b, f, c, h, w = 2, 8, 3, 512, 1024
    images = torch.rand((b, f, c, h, w), dtype=torch.float)
    feature_extractor = FeatureExtractor()
    x = feature_extractor(images)
    print(x.shape)


if __name__ == '__main__':
    test()
