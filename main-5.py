#!/usr/bin/env python3

"""Same as main.py but uses ResNet Option B."""


import torch
import torchvision
from torch.utils.data import DataLoader
from resnet import get_dataloader, SimpleResNet
from trainer import Trainer, device

    
if __name__ == '__main__':
    epochs = 200

    train_dataloader = get_dataloader(True)
    test_dataloader = get_dataloader(False)

    ns = 1, 3, 5, 7, 9, 18
    names = ('resnet8', 'resnet20', 'resnet32', 'resnet44',
             'resnet56', 'resnet110')

    print(f"using device: {device.type}")

    for i, name in enumerate(names):
        name += '-5'
        filename = f'{name}.trainer'
        print(f' ###### Start to process {name} ###### ')
        model = SimpleResNet(ns[i], 'B').to(device)
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
