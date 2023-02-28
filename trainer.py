#!/usr/bin/env python3

from __future__ import annotations
import logging
import pickle
import torch


__all__ = ('device', 'Trainer', 'get_logger')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    """Computes and stores the current and weighted average value."""

    def __init__(self):
        self.average = 0.
        self.value = 0.
        self.sum = 0.
        self.count = 0

    def update(self, value: float, n: int = 1) -> AverageMeter:
        self.value = value
        self.sum += value * n
        self.count += n
        self.avrage = self.sum / self.count
        return self


class Trainer():
    """Class for training a model."""
    def __init__(self, model: torch.nn.Module,
                 *,
                 epochs: int = 200,
                 start_epoch: int = 0,
                 filename: str | None = None,
                 logger: logging.Logger | str | None = None,
                 lr: float = 0.1,
                 milestones: list[int] = [100, 150],
                 gamma: float = 0.1,
                 weight_decay: float = 1e-4,
                 momentum: float = 0.9
                 ):
        self.model = model
        self.epochs = epochs
        self.epoch = start_epoch
        self.filename = 'trainer.trainer' if filename is None else filename
        if isinstance(logger, logging.Logger):
            self.logger = logger
        elif isinstance(logger, str):
            self.logger = get_logger(logger)
        else:
            self.logger = get_logger('trainer')

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr, weight_decay=weight_decay, momentum=momentum)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, gamma=gamma, milestones=milestones,
            last_epoch=self.epoch-1)

        self.history: dict[str, list[float]] = {'train_loss': [],
                                                'train_error': [],
                                                'test_loss': [],
                                                'test_error': []}

    @property
    def lr(self) -> list[float]:
        return [pg['lr'] for pg in self.optimizer.param_groups]

    @lr.setter
    def lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self):
        data = self.__dict__.copy()
        data['model'] = self.model.cpu()
        with open(self.filename, 'wb') as f:
            f.write(pickle.dumps((data, self.model.device)))

    @staticmethod
    def load(filename: str, device: torch.device | int | str | None = None
             ) -> Trainer:
        with open(filename, 'rb') as f:
            data, device = pickle.loads(f.read())
        if device is not None:
            data['model'] = data['model'].to(device)
        res = object.__new__(Trainer)
        res.__dict__.update(data)
        return res

    def train(self,
              train_dataloader: torch.utils.data.DataLoader,
              log_every: int = 50
              ) -> tuple[AverageMeter, AverageMeter]:
        "Train the model by given dataloader."
        self.model.train()
        loss_meter = AverageMeter()
        error_meter = AverageMeter()
        size = len(train_dataloader.dataset)
        trained_samples = 0
        for i, (x, y) in enumerate(train_dataloader):
            current_batch_size = x.shape[0]
            # compute prediction error
            y_pred = self.model(x)
            loss = self.loss_function(y_pred, y)
            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # record result
            trained_samples += current_batch_size
            loss_meter.update(loss.item(), current_batch_size)
            with torch.no_grad():
                error_meter.update(
                    1 - (y_pred.argmax(1) == y).sum().item(),
                    current_batch_size)
            if (i + 1) % log_every == 0:
                self.logger.info(f'loss = {loss_meter.value:.7f} '
                                 f'error = {error_meter.value*100:>0.1f} '
                                 f'[{trained_samples:>5d}/{size:>5d}]')
        self.logger.info(f'train result: '
                         f'loss = {loss_meter.average:.7f} '
                         f'error = {error_meter.average*100:>.1f}')
        self.scheduler.step()
        # Save information for this epoch.
        self.epoch += 1
        self.history['train_loss'].append(loss_meter.average)
        self.history['train_error'].append(error_meter.average)
        return loss_meter, error_meter

    def test(self, test_dataloader: torch.utils.data.DataLoader
             ) -> tuple[AverageMeter, AverageMeter]:
        "Test the model by given dataloader."
        self.model.eval()
        loss_meter = AverageMeter()
        error_meter = AverageMeter()
        with torch.no_grad():
            for i, (x, y) in enumerate(test_dataloader):
                current_batch_size = x.shape[0]
                y_pred = self.model(x)
                loss_meter.update(self.loss_function(y_pred, y).item(),
                                  current_batch_size)
                error_meter.update(1 - (y_pred.argmax(1) == y).sum().item(),
                                   current_batch_size)
        self.logger.info(f'test result: '
                         f'loss = {loss_meter.average:.7f} '
                         f'error = {error_meter.average*100:>.1f}')
        # Save test results only the fisrt run.
        if len(self.history['test_loss']) < self.epoch:
            self.history['test_loss'].append(loss_meter.average)
            self.history['test_error'].append(error_meter.average)
        return loss_meter, error_meter


def get_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    # formatter = logging.Formatter('%(asctime)s %(message)s')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    return logger
