import os
import csv
import numpy as np
import torch

from dataset.kitti.kitti import KittiRaw
from scipy.spatial.transform import Rotation
from einops import rearrange
from utils.logger import get_logger
from tqdm import tqdm


def pose_matrix_to_quaternion(pose):
    """ convert 4x4 pose matrix to (t, q) """
    q = Rotation.from_matrix(pose[..., :3, :3]).as_quat().astype(np.float32)
    return np.concatenate([pose[..., :3, 3], q], axis=-1)


def process_raw_data(data):
    return {
        'poses': pose_matrix_to_quaternion(data['poses']),
        'depth': data['depth'][None],
        'images': rearrange(data['images'], 'f h w c -> f c h w'),
        'intrinsics': data['intrinsics']
    }


def data_augment(img):
    random_gamma = np.random.uniform(0.9, 1.1)
    random_brightness = np.random.uniform(0.8, 1.2)
    random_colors = np.random.uniform(0.8, 1.2, [3])

    img = 255.0 * ((img / 255.0) ** random_gamma)
    img *= random_brightness
    img *= np.reshape(random_colors, [1, 3, 1, 1])
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img


class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, split_path=None, mode='train', logger=None,
                 debug_len=None, load_cache=False, seq_len=3, crops=108, use_aug=False):
        self.split_path = {
            'train': 'data/kitti/train_scenes_eigen.txt',
            'test': 'data/kitti/test_scenes_eigen.txt'
        } if split_path is None else split_path
        self.data_dir = data_dir
        self.crop_size = crops
        self.use_aug = use_aug
        self.logger = logger if logger else get_logger()
        self.args = {'frames': seq_len, 'height': 300, 'width': 1088, 'crop': self.crop_size, 'scale': 0.1, }
        self.debug_len = debug_len
        self.mode = mode
        self.image_size = np.array([self.args['height'] - self.args['crop'], self.args['width']])
        self.seq_len = self.args['frames']
        self.sequences = self.load_sequences()
        self.cache_dir = os.path.join(self.data_dir, f'pkl_cache_{self.seq_len}_{self.crop_size}')
        self.load_cache = load_cache

        self.data = self.load_pkl_cache() if self.load_cache else None
        if self.data is None:
            self.data = KittiRaw(self.data_dir, self.sequences, self.mode, self.args)
            if self.load_cache:
                self.save_pkl_cache()

        self.logger.info(f"Build dataset mode: {self.mode}  image_size: {self.image_size} "
                         f"seq_len: {self.seq_len}  debug_len: {self.debug_len}  data_len: {len(self)}")

    def __getitem__(self, idx):
        data = self.data[idx].copy()
        if not self.load_cache:
            data = process_raw_data(data)

        if self.use_aug and self.mode == 'train':
            data['images'] = data_augment(data['images'])

        return data

    def __len__(self):
        if self.debug_len is None:
            return len(self.data)
        else:
            return min(len(self.data), self.debug_len)

    def load_sequences(self):
        with open(self.split_path[self.mode]) as f:
            reader = csv.reader(f)
            sequences = [x[0] for x in reader]
        return sequences

    def load_pkl_cache(self):
        self.logger.info(f'Loading {self.mode} pkl cache to memory')
        data = []
        for drive in self.sequences:
            pkl_path = os.path.join(self.cache_dir, f'{drive}.pkl')
            if os.path.exists(pkl_path):
                data.extend(torch.load(pkl_path))
            else:
                self.logger.error(f'{pkl_path} not exists')
                return None
            if self.debug_len is not None and len(data) > self.debug_len:
                return data
        return data

    def save_pkl_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f'Saving pkl cache to dir: {self.cache_dir}')
        drive_training_set_index = {}
        for sequence in self.data.training_set_index:
            drive = sequence[0]['drive']
            if drive not in drive_training_set_index:
                drive_training_set_index[drive] = []
            drive_training_set_index[drive].append(sequence)

        for drive, training_set_index in drive_training_set_index.items():
            pkl_path = os.path.join(self.cache_dir, f'{drive}.pkl')
            pkl_data = []
            for i in tqdm(range(len(training_set_index))):
                pkl_data.append(process_raw_data(self.data.load_example(training_set_index[i])))
            torch.save(pkl_data, pkl_path)
            self.logger.info(f'Saved at {pkl_path}')
        self.logger.info('Saved all pkl cache')


def test():
    from visualization.show_pt import show_pt
    split_path = {'test': 'src/dataset/kitti/debug_scenes_eigen.txt'}
    dataset = KittiDataset(data_dir='src/dataset/kitti', split_path=split_path, mode='test', load_cache=True)
    data_loader = tqdm(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))
    for data in data_loader:
        pass


if __name__ == '__main__':
    test()
