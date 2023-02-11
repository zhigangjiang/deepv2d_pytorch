import numpy as np
import torch
import torchvision
import torch.nn.functional as F

from lietorch import SE3
from einops import rearrange
from geometry.transform import se3_transform_depth
from geometry.keyframe_optim import keyframe_optim
from models.modules.motion.pose_regressor import PoseRegressor
from models.modules.motion.feature_extractor import FeatureExtractor
from models.modules.motion.flow_net import FlowNet
from models.modules.base import BaseModule
from utils.misc import coords_normal


class MotionNet(BaseModule):
    def __init__(self, ckpt_dir, device, image_size, backbone, mode='keyframe', pose_iter=0, pose_len=6):

        super().__init__(ckpt_dir, device)
        self.image_size = np.array(image_size)
        self.backbone = backbone
        self.mode = mode
        self.pose_iter = pose_iter
        self.delta_upsilon = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        self.is_training = True
        self.use_regressor = True  # initial pose by pose_regressor when inference
        self.is_calibrated = True
        self.pose_regressor = PoseRegressor(self.image_size, backbone=self.backbone['pose'], pose_len=pose_len)
        self.feature_extractor = FeatureExtractor()
        self.downscale = self.feature_extractor.downscale
        self.flow_net = FlowNet(feat_size=self.image_size // self.downscale)
        self.to(self.device)

    def forward(self, data):
        images, depths, intrinsics = data['images'], data['depth'], data['intrinsics']
        return self.__forward__(None, images, depths, intrinsics)

    def __forward__(self, Ts, images, depths, intrinsics, init: bool = False):
        """
        :param Ts:
        :param images: [b, f, 3(c), h, w] uint8
        :param depths: [b, 1(f), h, w]
        :param intrinsics: [b, 4(fx, fy, cx, cy)]
        :param init: bool, inference use
        :return:
        """

        transforms = []
        residuals = []
        weights = []

        # motion network performs projection operations in features space
        b, f, _, h, w = images.shape

        images = images / 255.0
        if Ts is None or (self.use_regressor and init):
            pose_vec = self.pose_regressor(images)
            # pose_vec = 0.01 * pose_vec
            # add self pose: identity
            pose_vec = torch.cat([torch.tensor([0] * 6, device=self.device)[None, None].repeat(b, 1, 1), pose_vec], dim=1)
            Ts_init = SE3.exp(pose_vec)
            Ts = Ts_init
        else:
            Ts_init = Ts

        # add init pose
        transforms.append(Ts_init)

        if self.pose_iter == 0:
            if self.is_training:
                # add upsilon
                vec_upsilon = torch.normal(mean=0., std=1., size=[b, f, 6]) * torch.Tensor(self.delta_upsilon)
                T_upsilon = SE3.exp(vec_upsilon).to(self.device)
                Ts_init = T_upsilon * Ts_init

            ii = torch.tensor(([0] * (f - 1)), dtype=torch.int64)
            jj = torch.arange(1, f)
            feats = self.feature_extractor(images)  # [b, f, c, h//4, w//4]

            ds = self.feature_extractor.downscale
            intrinsics = intrinsics / ds
            depths = torchvision.transforms.Resize(size=(h // ds, w // ds))(depths)
            depths = depths[:, ii, ...]

            # key_feats
            feats1 = feats[:, ii, ...]
            # other_feats
            feats2 = feats[:, jj, ...]

            Tij_init = Ts_init[:, ii] * Ts_init[:, jj].inv()

            for i in range(self.pose_iter):
                # detach to stop gradient
                Tij_init = SE3.InitFromVec(Tij_init.data.detach())

                key_pts_init, _, project_coors_init, mask_init = se3_transform_depth(Tij_init, depths, intrinsics)
                # e.g. point feat1(u, v), it's corresponding point is feat2(u+x1, v+y1), then feat1(u, v)=feat2(u+x1, v+y1)
                # the point projecting is (u+x2, v+y2) by predicting pose (above project_coors), take feat2(u+x2, v+y2)
                # the flow is (x2-x2, y2-y1) between feat1(u, v)=feat2(u+x1, v+y1) and (u+x2, v+y2)
                featsw = rearrange(F.grid_sample(
                    rearrange(feats2, 'b f c h w -> (b f) c h w'),
                    rearrange(coords_normal(project_coors_init), 'b f h w u -> (b f) h w u'),
                    mode='bilinear', align_corners=True), '(b f) c h w -> b f c h w', b=b)
                featsw = mask_init[:, :, None].float() * featsw
                flow, weight = self.flow_net(feats1, featsw)
                weight = mask_init[..., None].float() * weight
                weights.append(weight)

                if (self.mode == 'keyframe') and self.is_calibrated:
                    Tij_opti = keyframe_optim(Tij_init, flow, weight, project_coors_init, key_pts_init, intrinsics)

                    # add self(keyframe) pose: identity
                    self_pose_vec = torch.tensor([0.] * 6, device=self.device)[None, None].repeat(b, 1, 1)
                    Ts_opti = SE3.InitFromVec(torch.cat([SE3.exp(self_pose_vec).data, Tij_opti.data], dim=1))
                else:
                    raise NotImplementedError

                _, _, project_coors_opti, mask_opti = se3_transform_depth(Tij_opti, depths, intrinsics)
                mask_all = (mask_init & mask_opti)[..., None].float()
                f_opti = mask_all * (flow - (project_coors_opti - project_coors_init))
                residuals.append(f_opti)
                transforms.append(Ts_opti)
                Tij_init = Tij_opti
                Ts = Ts_opti

        return {'Ts': Ts, 'transforms': transforms, 'residuals': residuals, 'weights': weights, 'depths': depths}


def test():
    from pipline.misc import data_to_device
    from dataset.kitti.kitti_dataset import KittiDataset
    from utils.init_env import init_env
    from loss.pose_loss import PoseLoss
    from tqdm import tqdm

    init_env()

    split_path = {'test': 'src/dataset/kitti/debug_test_scenes_eigen.txt'}
    dataset = KittiDataset(data_dir='src/dataset/kitti', split_path=split_path, mode='test')

    data_loader = tqdm(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))
    device = 'cuda'
    net = MotionNet(ckpt_dir='', device=device, image_size=dataset.image_size, )
    pose_loss = PoseLoss()

    for gt in data_loader:
        data_to_device(gt, device)
        dt = net(gt)
        loss, _ = pose_loss(gt, dt)
        loss.backward()


if __name__ == '__main__':
    test()
