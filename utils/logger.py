import os
import sys
import datetime

import logging
import functools
from termcolor import colored


def build_logger(config):
    output_dir = config.LOGGER.DIR
    local_rank = config.RUN.LOCAL_RANK
    name = config.MODEL.NAME
    logger = get_logger(output_dir, local_rank, name)
    return logger


def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


@functools.lru_cache()
def get_logger(output_dir=None, local_rank=None, name='Logger'):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.Formatter.converter = beijing

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = f'[%(asctime)s %(name)s][%(levelname)1.1s](%(filename)s %(lineno)d): %(message)s'
    color_fmt = colored(f'[%(asctime)s %(name)s][%(levelname)1.1s][{local_rank}]', 'green') + colored(
        f'(%(filename)s %(lineno)d)',
        'yellow') + ': %(message)s'
    if local_rank in [0] or local_rank is None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    if output_dir is not None:
        # create file handlers
        file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{local_rank}.log'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    return logger
