import torch


class Conv2D(torch.nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, bn=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_dim, out_dim, (3, 3), stride=stride, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn = bn
        self.batch_norm = torch.nn.BatchNorm2d(in_dim)

    def forward(self, x):
        if self.bn:
            return self.conv(self.relu(self.batch_norm(x)))
        else:
            return self.conv(self.relu(x))


class ResConv2D(torch.nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()
        self.stride = stride
        self.conv1 = Conv2D(in_dim, out_dim, stride)
        self.conv1_for_stride2 = Conv2D(in_dim, out_dim, 1)
        self.conv2 = Conv2D(out_dim, out_dim, stride)
        if stride == 2:
            self.relu = torch.nn.ReLU(inplace=False)
            self.conv = torch.nn.Conv2d(in_dim, out_dim, (1, 1), stride=stride)

    def forward(self, x):
        if self.stride == 1:
            y = self.conv2(self.conv1(x))
        else:
            y = self.conv2(self.conv1_for_stride2(x))
            x = self.conv(self.relu(x))
        return x + y


class Conv3D(torch.nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, bn=True):
        super().__init__()
        self.conv = torch.nn.Conv3d(in_dim, out_dim, (3, 3, 3), stride=stride, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn = bn
        self.batch_norm = torch.nn.BatchNorm3d(in_dim)

    def forward(self, x):
        if self.bn:
            return self.conv(self.relu(self.batch_norm(x)))
        else:
            return self.conv(self.relu(x))


class ResConv3D(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = Conv3D(in_dim, out_dim)
        self.conv2 = Conv3D(out_dim, out_dim)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y

