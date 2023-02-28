#!/usr/bin/env python3

import logging
import pickle
import torch
from torch.utils.data import DataLoader
import torchvision


class Shortcut(torch.nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers

    def forward(self, x):
        y = torch.nn.functional.pad(x[:, :, ::2, ::2],
            pad=[0, 0, 0, 0, self.n_layers//4, self.n_layers//4],
            mode='constant', value=0.)
        return y


class BasicBlock(torch.nn.Module):
    def __init__(self, n_layers, subsampling):
        super().__init__()
        if subsampling:
            path1 = torch.nn.Sequential(
                torch.nn.Conv2d(n_layers // 2, n_layers, [3, 3], [2, 2],
                                padding=[1, 1], bias=False),
                torch.nn.BatchNorm2d(n_layers),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_layers, n_layers, [3, 3], padding='same', bias=False),
                torch.nn.BatchNorm2d(n_layers)
            )
            path2 = Shortcut(n_layers)
        else:
            path1 = torch.nn.Sequential(
                torch.nn.Conv2d(n_layers, n_layers, [3, 3], padding='same', bias=False),
                torch.nn.BatchNorm2d(n_layers),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_layers, n_layers, [3, 3], padding='same', bias=False),
                torch.nn.BatchNorm2d(n_layers)
            )
            path2 = torch.nn.Identity()
        self.path1 = path1
        self.path2 = path2
        self.output = torch.nn.ReLU()

    def forward(self, x):
        y1 = self.path1(x)
        y2 = self.path2(x)
        return self.output(y1 + y2)


class SimpleResNet(torch.nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.n = n
        self.input = torch.nn.Sequential(torch.nn.Conv2d(3, 16, [3, 3],
                                                         padding='same'),
                                         torch.nn.BatchNorm2d(16),
                                         torch.nn.ReLU()
                                         )
        self.layer1 = torch.nn.Sequential(
            *[BasicBlock(16, False) for i in range(n)])
        self.layer2 = torch.nn.Sequential(
            BasicBlock(32, True),
            *[BasicBlock(32, False) for i in range(n - 1)])
        self.layer3 = torch.nn.Sequential(
            BasicBlock(64, True),
            *[BasicBlock(64, False) for i in range(n - 1)])
        self.pool = torch.nn.AdaptiveAvgPool2d([1, 1])
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(64, 10)

        # Use He Kaiming's normal initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                for name, parameter in m.named_parameters():
                    if name == 'weight':
                        torch.nn.init.kaiming_normal_(parameter)
                    elif name == 'bias':
                        torch.nn.init.zeros_(parameter)
                    else:
                        assert "impossible!"

    def forward(self, x):
        x = self.input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        features = self.flatten(x)
        y = self.fc(features)
        return y


def upgrade(model: SimpleResNet):
    """Add one more layer to model, with trained paremeters reserved."""
    n = model.n
    model2 = SimpleResNet(n + 1)
    model2.input.load_state_dict(model.input.state_dict())
    model2.layer1[:2 * n].load_state_dict(model.layer1.state_dict())
    model2.layer2[:2 * n].load_state_dict(model.layer2.state_dict())
    model2.layer3[:2 * n].load_state_dict(model.layer3.state_dict())
    model2.fc.load_state_dict(model.fc.state_dict())
    return model2


class Trainer():
    def __init__(self, net, filename, logger=logging):
        self.model = net
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                         weight_decay=1e-4, momentum=0.9)

        self.logger = logger
        self.filename = filename

        self.tune_state = {'epoch': 0,
                           'batch_size': [],
                           'lr_history': [],
                           'train_loss_history': [],
                           'train_error_history': [],
                           'test_loss_history': [],
                           'test_error_history': []}

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    @lr.setter
    def lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr

    def save(self):
        with open(self.filename, 'wb') as f:
            f.write(pickle.dumps(self))

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            res = pickle.loads(f.read())
        return res

    def train(self, train_dataloader):
        self.model.train()
        batch_size = train_dataloader.batch_size
        loss_history = []
        correct = 0
        size = len(train_dataloader) * batch_size
        for i, (x, y) in enumerate(train_dataloader):
            # compute prediction error
            y_pred = self.model(x)
            loss = self.loss_function(y_pred, y)
            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # record result
            loss_history.append(loss.item())
            with torch.no_grad():
                correct += (y_pred.argmax(1) == y).sum().item()
            if (int((i * batch_size) / size * 10) !=
                    int(((i + 1) * batch_size) / size * 10)):
                self.logger.info(f'loss =  {loss:.7f} '
                                 f'[{i * batch_size:>5d}/'
                                 f'{size:>5d}]')
        error = 1 - correct / size
        self.logger.info(f'train result: error = {(100*error):>0.1f}%')
        return loss_history, error

    def test(self, test_dataloader):
        size = len(test_dataloader) * test_dataloader.batch_size
        self.model.eval()
        loss, correct = 0, 0
        with torch.no_grad():
            for x, y in test_dataloader:
                y_pred = self.model(x)
                loss += self.loss_function(y_pred, y).item()
                correct += (y_pred.argmax(1) == y).sum().item()
        loss /= size
        error = 1 - correct / size
        self.logger.info(f'test result: error = {(100*error):>0.1f}%, '
                         f'avg loss = {loss:>8f}')
        return loss, error

    def tune(self, get_dataloader_func, nepochs=10,
             lrs=[0.1, 0.1, 0.01, 0.001]):
        """Tune the model.

        The model is tuned with different `lrs`, and with each
        `lr`, the model is trained `nepochs` epochs. The training
        and testing dataloaders are provided by `get_dataloader_func`,
        which accepts one argument `is_train`. If `is_train==True`,
        this function returns the training dataloader, else it returns
        test dataloader. The trainer is saved every epoch along
        with its model by calling `self.save()`.

        We use get_dataloader_func here so we could get a different
        dataloader in each epoch for data augmentation.

        """
        for lr in lrs:
            self.logger.info(f'\n------ Set lr to {lr} ------')
            self.lr = lr
            for i in range(nepochs):
                train_dataloader = get_dataloader_func(True)
                test_dataloader = get_dataloader_func(False)
                self.logger.info(f'\n------ EPOCH {i} (lr={lr}, '
                                 f'batch_size={train_dataloader.batch_size}) '
                                 f'------')
                train_loss, train_error = trainer.train(train_dataloader)
                test_loss, test_error = trainer.test(test_dataloader)
                # Save state after every epoch.
                self.tune_state['epoch'] += 1
                self.tune_state['batch_size'].append(train_dataloader.
                                                     batch_size)
                self.tune_state['lr_history'].append(lr)
                self.tune_state['train_loss_history'].append(train_loss)
                self.tune_state['train_error_history'].append(train_error)
                self.tune_state['test_loss_history'].append(test_loss)
                self.tune_state['test_error_history'].append(test_error)
                self.save()
        return self


def get_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    # formatter = logging.Formatter('%(asctime)s %(message)s')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    return logger


def get_dataloader(is_train, batch_size=128):
    if is_train:
        dataset = torchvision.datasets.CIFAR10(
            root='datasets', train=True, download=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, 4),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                ]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = torchvision.datasets.CIFAR10(
            root='datasets', train=False, download=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                ]))
        return DataLoader(dataset, batch_size=batch_size)


if __name__ == '__main__':
    logger = get_logger('my logger')

    train_dataloader = get_dataloader(True)
    test_dataloader = get_dataloader(False)

    x, y = next(iter(test_dataloader))

    trainers = []

    for n in [1, 3, 5, 7, 9]:
        model = SimpleResNet(n)
        print(f'\n------ Model n={model.n} ------')
        trainer = Trainer(model, f'trainer-n{model.n}.pickle', logger)
        trainer.tune(get_dataloader,
                     nepochs=50,
                     lrs=[0.1, 0.1, 0.01, 0.001])
        trainers.append(trainer)
