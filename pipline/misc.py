"""
@Date: 2022/9/14
@Description:
"""


def data_to_device(data, device):
    for data_k in data:
        data[data_k] = data[data_k].to(device, non_blocking=True)
