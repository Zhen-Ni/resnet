#!/usr/bin/env python3

"""Same as main.py but uses ResNet option A2"""

import torch
from resnet import get_dataloader
from resnet import SimpleResNet
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
        name += '-2'
        filename = f'{names[i]}.trainer'
        print(f' ###### Start to process {name} ###### ')
        model = SimpleResNet(ns[i], 'A2').to(device)
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
