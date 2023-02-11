import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import cv2


def init_env(config=None):
    seed = config.RUN.SEED if config else 0
    # Fix seed
    # Python & NumPy
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    #  PyTorch
    torch.manual_seed(seed)  # For cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # For current cpu
        torch.cuda.manual_seed_all(seed)  # For all cpu

    # cuDNN
    if config is None or config.RUN.DETERMINISTIC:
        # 复现
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # 将这个 flag 置为 True 的话，每次返回的卷积算法将是确定的，即默认算法
    else:
        cudnn.benchmark = True  # 如果网络的输入数据维度或类型上变化不大，设置true
        torch.backends.cudnn.deterministic = False

    # Using multiple threads in Opencv can cause deadlocks
    if config is None or config.DATASET.NUM_WORKERS != 0:
        cv2.setNumThreads(0)
