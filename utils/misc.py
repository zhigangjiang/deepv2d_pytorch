import numpy as np
import torch


def print_args(args, description=None):
    print("-" * 50)
    if description:
        print(f"description: {description}")
    print("arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("-" * 10)


def tensor2np(t: torch.Tensor) -> np.array:
    if isinstance(t, torch.Tensor):
        if t.device == 'cpu':
            return t.detach().numpy()
        else:
            return t.detach().cpu().numpy()
    else:
        return t


def tensor2np_d(d: dict) -> dict:
    output = {}
    for k in d.keys():
        output[k] = tensor2np(d[k])
    return output


def coords_normal(coords):
    """
    using F.grid_sample, it should have most values in the range of [-1, 1].
    :param coords: [..., h, w, u]  u = (u, v) or (x, y) => (w, h)
    :return: normalized coords [..., h, w, u]
    """
    w, h = coords.shape[-2], coords.shape[-3]
    coords_normalized = (coords / (torch.tensor([w, h], device=coords.device)[None, None, None])) * 2 - 1
    return coords_normalized


class ACCValue:
    def __init__(self, value, lager_better, key_acc=False):
        self.value = value
        self.lager_better = lager_better
        self.key_acc = key_acc

    def better(self, other):
        return (self.value < other) if self.lager_better else (self.value > other)

    @staticmethod
    def mean(values):
        mean_value = np.array([v.value for v in values]).mean()
        lager_better = values[0].lager_better
        key_acc = values[0].key_acc
        return ACCValue(mean_value, lager_better, key_acc)

    @staticmethod
    def to_float(acc_d):
        res = {}
        for k, v in acc_d.items():
            res[k] = v.value
        return res