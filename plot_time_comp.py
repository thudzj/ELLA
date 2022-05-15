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

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '{:.2f}'.format(height),
                ha='center', va='bottom', fontsize=12)

a = [14.65, 1.21, 8.15, 1.25, 1.22, 49.09, 52.04]
labels = []

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
# ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=10)

columns = ['ELLA', 'MAP', 'MFVI-BF', 'LLA$^*$', 'LLA$^*$-KFAC', 'LLA-Diag', 'LLA-KFAC']
rects1 = ax.bar(columns, a, color='sienna', width = 0.6)

ax.set_yscale('log')
ax.set_ylabel("Time spent predicting all test data (s)")
ax.set_ylim([0.5, 100])
# ax.set_yticks([0, 2, 4, 8, 16, 32, 100])

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=[], labels=[])
# ax.set_xticklabels([int(float(t.get_text()))  for t in ax.get_xticklabels()])
ax.spines['bottom'].set_color('gray')
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['left'].set_color('gray')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Skew intensity', fontsize=20)
ax.set_axisbelow(True)
# ax.grid(axis='y', color='lightgray', linestyle='--')

autolabel(rects1)

locs, labels = plt.xticks()
plt.setp(labels, rotation=30)
# plt.plot(x, delay

plt.tight_layout()
plt.savefig('time_comp.pdf', format='pdf', dpi=1000, bbox_inches='tight')
