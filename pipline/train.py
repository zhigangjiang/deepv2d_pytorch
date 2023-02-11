"""
@Date: 2022/9/12
@Description:
"""
import torch
import numpy as np

from models.build import build_model
from loss.build import build_loss
from accuracy.build import build_acc
from visualization.build import build_vis
from dataset.build import build_loader
from pipline.misc import data_to_device
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.misc import ACCValue

class Trainer:
    def __init__(self, config, logger, run_mode):
        self.run_mode = run_mode
        self.config = config
        self.logger = logger
        self.writer = SummaryWriter(config.CKPT.DIR) if self.run_mode == 'train' else None
        self.train_data_loader, self.val_data_loader = build_loader(self.config, self.logger)
        self.model, self.optimizer, self.scheduler = build_model(self.config, self.logger)
        self.loss = build_loss(self.config, self.logger)
        self.acc = build_acc(self.config, self.logger)
        self.vis = build_vis(self.config, self.logger)

    def train(self):
        for epoch in range(self.config.TRAIN.START_EPOCH, self.config.TRAIN.EPOCHS):
            self.logger.info("=" * 100)
            self.train_an_epoch(epoch)
            acc_d = self.val_an_epoch(epoch)

            self.model.save(self.optimizer, epoch, self.logger, replace=True, acc_d=acc_d, config=self.config)
            for k in self.model.acc_d:
                self.writer.add_scalar(f"BestACC/{k}", self.model.acc_d[k]['acc'], epoch)

            if self.scheduler is not None:
                if self.scheduler.min_lr is not None and self.optimizer.param_groups[0]['lr'] <= self.scheduler.min_lr:
                    continue
                self.scheduler.step()
        self.writer.close()

    def val(self):
        acc_d = self.val_an_epoch(0)
        return acc_d

    def train_an_epoch(self, epoch):
        mode = 'train'
        self.logger.info(f'Start Train Epoch {epoch}/{self.config.TRAIN.EPOCHS - 1}')
        self.model.train()
        self.optimizer.zero_grad()

        data_len = len(self.train_data_loader)
        start_i = data_len * epoch * self.config.RUN.WORLD_SIZE
        bar = enumerate(self.train_data_loader)
        if self.config.RUN.LOCAL_RANK == 0 and self.config.SHOW_BAR:
            bar = tqdm(bar, total=data_len, ncols=100)

        torch.autograd.set_detect_anomaly(True)
        epoch_acc_d = {}
        for i, gt in bar:
            global_step = start_i + i * self.config.RUN.WORLD_SIZE + self.config.RUN.LOCAL_RANK
            data_to_device(gt, self.config.RUN.DEVICE)
            dt = self.model(gt)
            loss = self.calc_criterion(mode, gt, dt, epoch_acc_d, global_step, bar, train=True, index=i)
            self.optimizer.zero_grad()
            # self.logger.info(f'{loss.item()}')
            loss.backward()
            self.optimizer.step()
        self.epoch_info(mode, epoch, epoch_acc_d)

    @torch.no_grad()
    def val_an_epoch(self, epoch):
        mode = 'val'
        self.logger.info(f'Start Validate Epoch {epoch}/{self.config.TRAIN.EPOCHS - 1}')
        self.model.eval()

        data_len = len(self.val_data_loader)
        start_i = data_len * epoch * self.config.RUN.WORLD_SIZE
        bar = enumerate(self.val_data_loader)
        if self.config.RUN.LOCAL_RANK == 0 and self.config.SHOW_BAR:
            bar = tqdm(bar, total=data_len, ncols=100)

        epoch_acc_d = {}
        for i, gt in bar:
            global_step = start_i + i * self.config.RUN.WORLD_SIZE + self.config.RUN.LOCAL_RANK
            data_to_device(gt, self.config.RUN.DEVICE)
            dt = self.model(gt)
            self.calc_criterion(mode, gt, dt, epoch_acc_d, global_step, bar, index=i)

        acc_d = self.epoch_info(mode, epoch, epoch_acc_d)
        return acc_d

    def epoch_info(self, mode, epoch, epoch_acc_d):
        if self.config.RUN.LOCAL_RANK != 0:
            return

        epoch_acc_d = dict(zip(epoch_acc_d.keys(), [ACCValue.mean(epoch_acc_d[k]) for k in epoch_acc_d.keys()]))
        s = f'{mode}Epoch: '
        for key, val in epoch_acc_d.items():
            if self.run_mode == 'train':
                self.writer.add_scalar(f'{mode}Epoch/{key}', val.value, epoch)
            s += f" {key}={val.value}"
        self.logger.info(s)

        if mode == 'train':
            if self.run_mode == 'train':
                self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            self.logger.info(f"LearningRate: {self.optimizer.param_groups[0]['lr']}")

        return epoch_acc_d

    def calc_criterion(self, mode, gt, dt, epoch_acc_d, global_step, bar, train=False, index=None):
        loss, acc_d = self.loss(gt, dt)

        if not train:
            acc_d.update(self.acc(gt, dt))

        if self.config.RUN.LOCAL_RANK == 0 and self.config.SHOW_BAR:
            bar.set_postfix(ACCValue.to_float(acc_d))

        for acc_k, acc_v in acc_d.items():
            if acc_k not in epoch_acc_d:
                epoch_acc_d[acc_k] = []
            epoch_acc_d[acc_k].append(acc_v)

        if self.run_mode == 'train':
            for key, val in acc_d.items():
                self.writer.add_scalar(f'{mode}Batch/{key}', val.value, global_step)

            if self.vis and index in self.vis.show_indexes:
                vis_d = self.vis(gt, dt)
                for key, val in vis_d.items():
                    self.writer.add_images(f'{mode}Batch/{key}', val, global_step)

        return loss
