#!/usr/bin/env python3

"""Same as main.py but normalization of input dataset is not
performed."""


import torch
import torchvision
from torch.utils.data import DataLoader
from resnet import SimpleResNet
from trainer import Trainer, device


def get_dataloader(is_train: bool,
                   download: bool = False,
                   batch_size: int = 128
                   ) -> torch.utils.data.DataLoader:
    if is_train:
        dataset = torchvision.datasets.CIFAR10(
            root='datasets', train=True, download=download,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, 4),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                ]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = torchvision.datasets.CIFAR10(
            root='datasets', train=False, download=download,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                ]))
        return DataLoader(dataset, batch_size=batch_size)

    
if __name__ == '__main__':
    epochs = 200

    train_dataloader = get_dataloader(True)
    test_dataloader = get_dataloader(False)

    ns = 1, 3, 5, 7, 9, 18
    names = ('resnet8', 'resnet20', 'resnet32', 'resnet44',
             'resnet56', 'resnet110')

    print(f"using device: {device.type}")

    for i, name in enumerate(names):
        name += '-4'
        filename = f'{name}.trainer'
        print(f' ###### Start to process {name} ###### ')
        model = SimpleResNet(ns[i], 'A1').to(device)
        try:
            trainer = Trainer.load(filename, device=device)
            print("Use stored trainer, continue training.")
        except FileNotFoundError:
            trainer = Trainer(model, filename=filename)
            print("New trainer generated.")
        while trainer.epoch < epochs:
            trainer.train(train_dataloader)
            trainer.test(test_dataloader)
            trainer.save(device='cpu')
