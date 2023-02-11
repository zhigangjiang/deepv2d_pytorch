from lietorch import SE3
import torch
import numpy as np

from models.modules.depth.feature_extractor import FeatureExtractor
from models.modules.base import BaseModule
from models.modules.depth.stereo_net import StereoNet


class DepthNet(BaseModule):
    def __init__(self, ckpt_dir, device, image_size, backbone, mode='avg', downscale=4, hg_count=2, seq_len=3):
        super().__init__(ckpt_dir, device)
        self.image_size = np.array(image_size)
        self.backbone = backbone
        self.mode = mode
        self.feature_extractor = FeatureExtractor(self.backbone['extractor'], downscale)
        self.stereo_net = StereoNet(self.image_size, self.mode, dim=self.feature_extractor.out_dim,
                                    hg_count=hg_count, seq_len=seq_len)
        self.depths = torch.linspace(0.1, 8.0, 32)
        self.to(self.device)
        self.pred_logits = []

    def forward(self, data):
        poses, images, intrinsics = data['poses'], data['images'], data['intrinsics']
        Ts = SE3.InitFromVec(poses)
        return self.__forward__(Ts, images, intrinsics)

    def __forward__(self, Ts, images, intrinsics):
        """
        :param Ts:
        :param images: [b, f, 3(c), h, w] uint8
        :param intrinsics: [b, 4(fx, fy, cx, cy)]
        :return:
        """
        images = 2 * (images / 255.0) - 1.0

        feats = self.feature_extractor(images)  # [b, f, 32(c), h/4, w/4] out_dim=32(c)
        ds = self.feature_extractor.downscale
        intrinsics = intrinsics / ds

        depths = self.stereo_net(Ts, feats, intrinsics)
        return {'depths': depths}


def test():
    from pipline.misc import data_to_device
    from dataset.kitti.kitti_dataset import KittiDataset
    from utils.init_env import init_env
    from loss.depth_loss import DepthLoss

    init_env()
    torch.hub._hub_dir = 'ckpts'

    split_path = {'test': 'src/dataset/kitti/debug_scenes_eigen.txt'}
    dataset = KittiDataset(data_dir='src/dataset/kitti', split_path=split_path, mode='test', debug_len=100, seq_len=5)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    device = 'cuda:0'
    net = DepthNet(ckpt_dir='', device=device, image_size=dataset.image_size, backbone={'extractor': 'down_sample'},
                   mode='avg', seq_len=dataset.seq_len, downscale=1)
    optim = torch.optim.SGD(net.parameters(), lr=1e-3)
    depth_loss = DepthLoss()

    for gt in data_loader:
        data_to_device(gt, device)
        dt = net(gt)
        loss, _ = depth_loss(gt, dt)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f'loss={loss.item()}')


if __name__ == '__main__':
    test()
