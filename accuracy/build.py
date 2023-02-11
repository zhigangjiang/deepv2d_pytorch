"""
@time: 2022/10/05
@description:
"""

import accuracy


def build_acc(config, logger):
    name = config.TRAIN.CRITERION.NAME
    args = config.TRAIN.CRITERION.ARGS[0]

    acc = getattr(accuracy, name)(**args)
    logger.info(f"Build acc: {name}, args:{args}")
    return acc
