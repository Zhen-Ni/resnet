#!/usr/bin/env python3

import torch
from resnet import get_dataloader
from resnet import resnet8, resnet20, resnet32, resnet44, resnet56, resnet110
from trainer import Trainer, device


if __name__ == '__main__':
    epochs = 200

    train_dataloader = get_dataloader(True)
    test_dataloader = get_dataloader(False)

    names = ('resnet8', 'resnet20', 'resnet32', 'resnet44',
             'resnet56', 'resnet110')

    print(f"using device: {device.type}")

    for i, name in enumerate(names):
        filename = f'{names[i]}.trainer'
        print(f' ###### Start to process {name} ###### ')
        model = eval(name)(device)
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
