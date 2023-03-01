#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
import torchvision


__all__ = ('SimpleResNet', 'get_dataloader',
           'resnet8', 'resnet20', 'resnet32', 'resnet44', 'resnet56',
           'resnet110')


class Shortcut1(torch.nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.pad(
            x[:, :, ::2, ::2],
            pad=[0, 0, 0, 0, 0, self.n_layers - self.n_layers//2],
            mode='constant', value=0.)
        return y


class Shortcut2(torch.nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        y = torch.nn.functional.pad(
            x, pad=[0, 0, 0, 0, 0, self.n_layers - self.n_layers//2],
            mode='constant', value=0.)
        return y
    

class BasicBlock(torch.nn.Module):
    def __init__(self, n_layers: int, subsampling: bool, option='A1'):
        "Option is only necessary when subsampling is True"
        path2: torch.nn.Module
        super().__init__()
        if subsampling:
            path1 = torch.nn.Sequential(
                torch.nn.Conv2d(n_layers // 2, n_layers, (3, 3), (2, 2),
                                padding=(1, 1), bias=False),
                torch.nn.BatchNorm2d(n_layers),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_layers, n_layers, (3, 3),
                                padding='same', bias=False),
                torch.nn.BatchNorm2d(n_layers)
            )
            if option == 'A1':
                path2 = Shortcut1(n_layers)
            elif option == 'A2':
                path2 = Shortcut2(n_layers)
            elif option == 'B':
                path2 = torch.nn.Conv2d(n_layers // 2, n_layers,
                    (3, 3), (2, 2), (1, 1),
                    bias=False)
            else:
                    raise ValueError("option must be 'A1', 'A2' or 'B'")
            # match option:
            #     case 'A':
            #         path2 = Shortcut(n_layers)
            #     case 'B':
            #         path2 = torch.nn.Conv2d(n_layers // 2, n_layers,
            #                                 (3, 3), (2, 2), (1, 1),
            #                                 bias=False)
            #     case _:
            #         raise ValueError("option must be 'A' or 'B'")

        else:
            path1 = torch.nn.Sequential(
                torch.nn.Conv2d(n_layers, n_layers, (3, 3),
                                padding='same', bias=False),
                torch.nn.BatchNorm2d(n_layers),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_layers, n_layers, (3, 3),
                                padding='same', bias=False),
                torch.nn.BatchNorm2d(n_layers)
            )
            path2 = torch.nn.Identity()
        self.path1 = path1
        self.path2 = path2
        self.output = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.path1(x)
        y2 = self.path2(x)
        return self.output(y1 + y2)


class SimpleResNet(torch.nn.Module):
    def __init__(self, n: int = 2, option: str = 'A1'):
        super().__init__()
        self.n = n
        self.input = torch.nn.Sequential(torch.nn.Conv2d(3, 16, (3, 3),
                                                         padding='same'),
                                         torch.nn.BatchNorm2d(16),
                                         torch.nn.ReLU()
                                         )
        self.layer1 = torch.nn.Sequential(
            *[BasicBlock(16, False) for i in range(n)])
        self.layer2 = torch.nn.Sequential(
            BasicBlock(32, True, option),
            *[BasicBlock(32, False) for i in range(n - 1)])
        self.layer3 = torch.nn.Sequential(
            BasicBlock(64, True, option),
            *[BasicBlock(64, False) for i in range(n - 1)])
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        features = self.flatten(x)
        y = self.fc(features)
        return y


def resnet8(device: torch.device | str | int | None = None
            ) -> SimpleResNet:
    return SimpleResNet(1).to(device)


def resnet20(device: torch.device | str | int | None = None
             ) -> SimpleResNet:
    return SimpleResNet(3).to(device)


def resnet32(device: torch.device | str | int | None = None
             ) -> SimpleResNet:
    return SimpleResNet(5).to(device)


def resnet44(device: torch.device | str | int | None = None
             ) -> SimpleResNet:
    return SimpleResNet(7).to(device)


def resnet56(device: torch.device | str | int | None = None
             ) -> SimpleResNet:
    return SimpleResNet(9).to(device)


def resnet110(device: torch.device | str | int | None = None
              ) -> SimpleResNet:
    return SimpleResNet(18).to(device)
    

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


def get_dataloader(is_train: bool,
                   download: bool = False,
                   batch_size: int = 128) -> torch.utils.data.DataLoader:
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
    if is_train:
        dataset = torchvision.datasets.CIFAR10(
            root='datasets', train=True, download=download,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, 4),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                normalize,
                ]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = torchvision.datasets.CIFAR10(
            root='datasets', train=False, download=download,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                normalize
                ]))
        return DataLoader(dataset, batch_size=batch_size)
