"""
@Date: 2022/9/12
@Description:
"""
import models
from utils.time_watch import TimeWatch
from models.misc.optimizer import build_optimizer


def build_model(config, logger):
    name = config.MODEL.NAME
    w = TimeWatch(f"Build model: {name}", logger)

    device = config.RUN.DEVICE
    logger.info(f"Creating model: {name} to device:{device}, args:{config.MODEL.ARGS[0]}")
    net = getattr(models, name)
    if len(config.MODEL.ARGS) != 0:
        model: models.BaseModule = net(ckpt_dir=config.CKPT.DIR, device=device, **config.MODEL.ARGS[0])
    else:
        model: models.BaseModule = net(ckpt_dir=config.CKPT.DIR, device=device)

    optimizer = None
    scheduler = None

    if config.MODE == 'train':
        optimizer = build_optimizer(config, model, logger)

    config.defrost()
    best = config.MODE != 'train' or not config.TRAIN.RESUME_LAST
    config.TRAIN.START_EPOCH = model.load(device, logger, optimizer, best)
    config.freeze()

    return model, optimizer, scheduler
