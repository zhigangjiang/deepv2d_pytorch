import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

from einops import rearrange, repeat
from lietorch import SE3


class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, pose_len=6):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        self.encoder.fc = nn.Linear(2048, pose_len)
        del self.encoder.conv1

    def forward(self, x):
        x = self.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.encoder.fc(x)
        return x


class CNN(nn.Module):
    def __init__(self, image_size, downscale=2, pose_len=6):
        super().__init__()
        self.downscale = downscale
        self.layer1 = nn.Conv2d(6, 32, (7, 7), stride=2, padding=3)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, (7, 1), stride=1, padding=(3, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 5), stride=1, padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (5, 1), stride=1, padding=(2, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, (3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, (3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        full_conv_size = np.ceil(image_size / self.downscale / 2 ** 6).astype(np.int64)
        self.full_layer = nn.Sequential(
            nn.Conv2d(256, 512, full_conv_size, stride=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(512, pose_len)

    def forward(self, x: torch.Tensor):
        # in original code: interpolation=AREA
        _, _, h, w = x.shape
        x = torchvision.transforms.Resize(size=(h // self.downscale, w // self.downscale))(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.full_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class PoseRegressor(nn.Module):
    def __init__(self, image_size, backbone='resnet50', pose_len=6):
        """

        :param image_size:
        :param backbone:
        :param pose_len: 6--6Dof or 7--[trans3, quat4]
        """
        super().__init__()
        self.pose_len = pose_len
        if backbone == 'cnn':
            self.regressor = CNN(image_size, downscale=2, pose_len=self.pose_len)
        else:
            self.regressor = ResNet(backbone, pretrained=True, pose_len=self.pose_len)

    def forward(self, x):
        """
        :param x: images [b, f, c(3), h, w]
        :return: pose_vec [b, f-1, 6or7]
        """
        b, f, _, h, w = x.shape
        # concat in c dim
        x = torch.cat([
            repeat(x[:, 0], 'b c h w -> (b f) c h w', f=f - 1),  # key_images
            rearrange(x[:, 1:], 'b f c h w -> (b f) c h w')  # other_images
        ], dim=1)
        x = self.regressor(x)
        pose_vec = x.reshape(b, f - 1, self.pose_len)
        return pose_vec


def test():
    b, f, c, h, w = 2, 8, 3, 512, 1024
    images = torch.rand((b, f, c, h, w), dtype=torch.float)
    pose_regressor = PoseRegressor(image_size=np.array([h, w]), backbone='resnet50', pose_len=7)
    pose_vec = pose_regressor(images)


if __name__ == '__main__':
    test()
