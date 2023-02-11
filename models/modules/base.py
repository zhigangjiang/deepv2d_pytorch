""" 
@Date: 2021/07/17
@description:
"""
import os
import torch
import torch.nn as nn

from datetime import datetime, timezone, timedelta
from utils.misc import ACCValue


class BaseModule(nn.Module):
    def __init__(self, ckpt_dir=None, device='cpu'):
        super().__init__()

        self.ckpt_dir = ckpt_dir
        self.device = device
        if ckpt_dir:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            else:
                self.model_lst = [x for x in sorted(os.listdir(self.ckpt_dir)) if x.endswith('.pkl')]

        self.last_model_path = None
        self.best_model_path = None
        self.best_accuracy = None
        self.acc_d = {}

    def show_parameter_number(self, logger):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info('{} parameter total:{:,}, trainable:{:,}'.format(self._get_name(), total, trainable))

    def load(self, device, logger, optimizer=None, best=False):
        if len(self.model_lst) == 0:
            logger.info('*' * 50)
            logger.info("Empty model folder! Using initial weights")
            logger.info('*' * 50)
            return 0

        last_model_lst = list(filter(lambda n: '_last_' in n, self.model_lst))
        best_model_lst = list(filter(lambda n: '_best_' in n, self.model_lst))

        if len(last_model_lst) == 0 and len(best_model_lst) == 0:
            logger.info('*' * 50)
            ckpt_path = os.path.join(self.ckpt_dir, self.model_lst[0])
            logger.info(f"Load: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
            self.load_state_dict(checkpoint, strict=False)
            logger.info('*' * 50)
            return 0

        checkpoint = None
        if len(last_model_lst) > 0:
            self.last_model_path = os.path.join(self.ckpt_dir, last_model_lst[-1])
            checkpoint = torch.load(self.last_model_path, map_location=torch.device(device))
            self.best_accuracy = checkpoint['accuracy']
            self.acc_d = checkpoint['acc_d']

        if len(best_model_lst) > 0:
            self.best_model_path = os.path.join(self.ckpt_dir, best_model_lst[-1])
            best_checkpoint = torch.load(self.best_model_path, map_location=torch.device(device))
            self.best_accuracy = best_checkpoint['accuracy']
            self.acc_d = best_checkpoint['acc_d']
            if best:
                checkpoint = best_checkpoint

        for k in self.acc_d:
            if isinstance(self.acc_d[k], float):
                self.acc_d[k] = {
                    'acc': self.acc_d[k],
                    'epoch': checkpoint['epoch']
                }

        if checkpoint is None:
            logger.error("Invalid checkpoint")
            return

        self.load_state_dict(checkpoint['net'], strict=False)
        if optimizer and not best:  # best的时候使用新的优化器比如从adam->sgd
            logger.info('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

        logger.info('*' * 50)
        if best:
            logger.info(f"Lode best: {self.best_model_path}")
        else:
            logger.info(f"Lode last: {self.last_model_path}")

        logger.info(f"Best accuracy: {self.best_accuracy}")
        logger.info(f"Last epoch: {checkpoint['epoch'] + 1}")
        logger.info('*' * 50)
        return checkpoint['epoch'] + 1

    def update_acc(self, acc_d, epoch, logger):
        logger.info("-" * 100)
        for k in acc_d:
            if k not in self.acc_d.keys() or acc_d[k].better(self.acc_d[k]['acc']):
                self.acc_d[k] = {
                    'acc': acc_d[k].value,
                    'epoch': epoch
                }
            logger.info(f"Update {k}: {self.acc_d[k]['acc']:.4f}({self.acc_d[k]['epoch']}-{epoch}) "
                        f"lager_better: {acc_d[k].lager_better} key_acc: {acc_d[k].key_acc}")

        logger.info("-" * 100)

    def save(self, optim, epoch, logger, replace=True, acc_d=None, config=None):
        key_acc = [k for k in acc_d if acc_d[k].key_acc][0]
        acc: ACCValue = acc_d[key_acc]

        self.update_acc(acc_d, epoch, logger)
        name = datetime.now().astimezone(timezone(timedelta(hours=8))).strftime(
            '%Y-%m-%d-%H-%M-%S_last_{:.4f}_{}'.format(acc.value, epoch))
        name = f"model_{name}.pkl"
        checkpoint = {
            'net': self.state_dict(),
            'optimizer': optim.state_dict(),
            'epoch': epoch,
            'accuracy': acc.value,
            'acc_d': ACCValue.to_float(acc_d)
        }

        if config.TRAIN.SAVE_FREQ != 0 and epoch % config.TRAIN.SAVE_FREQ == 0:
            if replace and self.last_model_path and os.path.exists(self.last_model_path):
                os.remove(self.last_model_path)
            self.last_model_path = os.path.join(self.ckpt_dir, name)
            torch.save(checkpoint, self.last_model_path)
            logger.info(f"Saved last model: {self.last_model_path}")

        if self.best_accuracy is None or acc.better(self.best_accuracy):
            self.best_accuracy = acc.value
            if config.TRAIN.SAVE_BEST:
                if replace and self.best_model_path and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                self.best_model_path = os.path.join(self.ckpt_dir, name.replace('last', 'best'))
                torch.save(checkpoint, self.best_model_path)
                logger.info(f"Saved best model: {self.best_model_path}")
