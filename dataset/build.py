import dataset
import torch
import numpy as np


def build_loader(config, logger):
    train_dataset = None
    train_data_loader = None
    train_sampler = None
    test_sampler = None

    if config.MODE == 'train':
        train_dataset, _ = build_dataset(mode='train', config=config, logger=logger)

    val_dataset, load_cache = build_dataset(mode='test', config=config, logger=logger)

    batch_size = config.DATASET.BATCH_SIZE
    num_workers = 0 if config.DEBUG else config.DATASET.NUM_WORKERS
    pin_memory = config.DATASET.PIN_MEMORY

    if train_dataset:
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=test_sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    logger.info(f'Build dataset loader: batch size: {batch_size} num_workers: {num_workers} pin_memory: {pin_memory}')
    return train_data_loader, val_data_loader


def build_dataset(mode, config, logger):
    name = config.DATASET.NAME
    args = config.DATASET.ARGS[0]
    if config.DEBUG:
        args['debug_len'] = 100
    dataset_ = getattr(dataset, name)(mode=mode, logger=logger, **args)

    return dataset_, args['load_cache'] if 'load_cache' in args else False
