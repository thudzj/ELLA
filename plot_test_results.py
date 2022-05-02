import os
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})

import pandas as pd
import numpy.random as rnd
import seaborn as sns

data = []
archs = ['resnet20', 'resnet32', 'resnet44', 'resnet56']
for arch in archs:
    file = './logs/cifar10/cifar10_{}/default/test_results.npy'.format(arch)
    data.append(np.load(file))

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
for item, arch in zip(data, archs):
    ax.plot(item[:, 0], item[:, 1], label=arch)
# ax.set_xticklabels([int(float(t.get_text()))  for t in ax.get_xticklabels()])
ax.spines['bottom'].set_color('gray')
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('$N$', fontsize=14)
ax.set_ylabel('Test NLL', fontsize=14)
ax.set_axisbelow(True)
ax.grid(axis='y', color='lightgray', linestyle='--')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig('test_curve2.pdf', format='pdf', dpi=1000, bbox_inches='tight')

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(121)
for item, arch in zip(data, archs):
    ax.plot(item[:, 0], item[:, 2], label=arch)
# ax.set_xticklabels([int(float(t.get_text()))  for t in ax.get_xticklabels()])
ax.spines['bottom'].set_color('gray')
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('$N$', fontsize=14)
ax.set_ylabel('Test accuracy', fontsize=14)
ax.set_axisbelow(True)
ax.grid(axis='y', color='lightgray', linestyle='--')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

ax = fig.add_subplot(122)
for item, arch in zip(data, archs):
    ax.plot(item[:, 0], item[:, 3], label=arch)
# ax.set_xticklabels([int(float(t.get_text()))  for t in ax.get_xticklabels()])
ax.spines['bottom'].set_color('gray')
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('$N$', fontsize=14)
ax.set_ylabel('Test ECE', fontsize=14)
ax.set_axisbelow(True)
ax.grid(axis='y', color='lightgray', linestyle='--')

plt.savefig('test_curve.pdf', format='pdf', dpi=1000, bbox_inches='tight')
