"""
@date: 2021/7/19
@description:
"""
import loss


def build_loss(config, logger):
    name = config.TRAIN.LOSS.NAME
    args = config.TRAIN.LOSS.ARGS[0]

    loss_ = getattr(loss, name)(**args)
    loss_ = loss_.to(config.RUN.DEVICE)
    logger.info(f"Build loss: {name}, args:{args}")
    return loss_
