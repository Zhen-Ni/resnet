#!/usr/bin/env python3

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
plt.rc('font', family='STIXGeneral', weight='normal', size=10)
plt.rc('mathtext', fontset='stix')

from trainer import Trainer

names = []
trainers = []

for filename in os.listdir():
    if filename.endswith('.trainer'):
        names.append(filename[:-8])
        trainers.append(Trainer.load(filename, device='cpu'))

fig = plt.figure(figsize=(6, 4.5))
ax = fig.add_subplot(111)

colors = 'k', 'r', 'b', 'g', 'c', 'm', 'y'

for i, (name, trainer) in enumerate(zip(names, trainers)):
    ax.plot(trainer.history['train_error'], ':', color=colors[i])
    ax.plot(trainer.history['test_error'], '-', color=colors[i], label=name)

ax.grid()
ax.set_ylim(0, .4)
ax.set_xlim(0, 200)
ax.legend()

plt.show()
plt.savefig('error.svg')



