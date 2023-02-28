#!/usr/bin/env python3

import torch
from resnet import get_dataloader
from resnet import resnet8, resnet20, resnet32
from trainer import Trainer, device


if __name__ == '__main__':
    epochs = 200

    train_dataloader = get_dataloader(True)
    test_dataloader = get_dataloader(False)

    resnet8 = resnet8.to(device)
    resnet20 = resnet20.to(device)
    resnet32 = resnet32.to(device)
    names = 'resnet8', 'resnet20', 'resnet32'

    print(f"using device: {device.type}")

    for i, model in enumerate([resnet8, resnet20, resnet32]):
        filename = f'{names[i]}.trainer'
        print(f' ###### Start to process {names[i]} ###### ')
        try:
            trainer = Trainer.load(filename, device=device)
            print("Use stored trainer, continue training.")
        except FileNotFoundError:
            trainer = Trainer(model, filename=filename)
            print("New trainer generated.")
        while trainer.epoch < epochs:
            trainer.train(train_dataloader)
            trainer.test(test_dataloader)
            trainer.save()
