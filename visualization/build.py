"""
@time: 2022/10/05
@description:
"""

import visualization


def build_vis(config, logger):
    name = config.TRAIN.VIS.NAME
    if hasattr(visualization, name):
        args = config.TRAIN.VIS.ARGS[0]
        acc = getattr(visualization, name)(**args)
        logger.info(f"Build visualization: {name}, args:{args}")
        return acc
    else:
        return None

