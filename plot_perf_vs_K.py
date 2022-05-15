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

r1 = '''K=1  Average loss: 0.2774, Accuracy: 0.9254, ECE: 0.0371
K=2  Average loss: 0.2734, Accuracy: 0.9257, ECE: 0.0356
K=5  Average loss: 0.2637, Accuracy: 0.9265, ECE: 0.0325
K=10 Average loss: 0.2499, Accuracy: 0.9260, ECE: 0.0231
K=15 Average loss: 0.2397, Accuracy: 0.9266, ECE: 0.0152
K=20 Average loss: 0.2331, Accuracy: 0.9258, ECE: 0.0092
K=25 Average loss: 0.2304, Accuracy: 0.9258, ECE: 0.0079
K=30 Average loss: 0.2303, Accuracy: 0.9255, ECE: 0.0112'''
Ks1, nlls1, eces1 = [], [], []
for line in r1.split("\n"):
    Ks1.append(int(line.split()[0].split('=')[1]))
    nlls1.append(float(line.split()[3].replace(',', '')))
    eces1.append(float(line.split()[7]))

r2 = '''K=1  Average loss: 0.2785, Accuracy: 0.9350, ECE: 0.0367
K=2  Average loss: 0.2628, Accuracy: 0.9351, ECE: 0.0324
K=5  Average loss: 0.2360, Accuracy: 0.9348, ECE: 0.0203
K=10 Average loss: 0.2175, Accuracy: 0.9349, ECE: 0.0066
K=15 Average loss: 0.2170, Accuracy: 0.9350, ECE: 0.0202
K=20 Average loss: 0.2161, Accuracy: 0.9349, ECE: 0.0082
K=25 Average loss: 0.2153, Accuracy: 0.9352, ECE: 0.0108
K=30 Average loss: 0.2237, Accuracy: 0.9350, ECE: 0.0262'''
Ks2, nlls2, eces2 = [], [], []
for line in r2.split("\n"):
    Ks2.append(int(line.split()[0].split('=')[1]))
    nlls2.append(float(line.split()[3].replace(',', '')))
    eces2.append(float(line.split()[7]))

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
ax.plot(Ks1, nlls1, label='ResNet20')
# ax.plot(Ks2, nlls2, label='ResNet32')
# ax.set_xticklabels([int(float(t.get_text()))  for t in ax.get_xticklabels()])
ax.spines['bottom'].set_color('gray')
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('$K$', fontsize=14)
ax.set_ylabel('Test NLL', fontsize=14)
ax.set_axisbelow(True)
ax.grid(axis='y', color='lightgray', linestyle='--')
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=labels)
plt.savefig('perf_vs_K.pdf', format='pdf', dpi=1000, bbox_inches='tight')
