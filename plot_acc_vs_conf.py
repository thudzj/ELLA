import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})

import os
import math
from tqdm import tqdm
import numpy as np

import pandas as pd
import seaborn as sns

import math
from tqdm import tqdm
import numpy as np

import pandas as pd
import seaborn as sns


for title in ['cifar10_resnet20']: #, 'cifar10_resnet32', 'cifar10_resnet44', 'cifar10_resnet56']:

    files = [os.path.join('./logs/{}/{}/default/'.format(title.split("_")[0], title), 'acc_vs_conf_ella.npy'),
             os.path.join('./logs/{}/{}/default/'.format(title.split("_")[0], title), 'acc_vs_conf_map.npy')]
    for job_id in ['lastl-full', 'lastl-kron', 'mfvi']: #, 'all-diag', 'all-kron'
        dir = './logs/{}/{}/{}/'.format(title.split("_")[0], title, job_id)
        files.append(os.path.join(dir, 'acc_vs_conf_{}.npy'.format(job_id)))
    labels = ['ELLA', 'MAP', 'MFVI-BF', 'LLA$^*$', 'LLA$^*$-KFAC'] #, 'LLA-Diag', 'LLA-KFAC'

    x_test = np.linspace(0, 1, 300)[:-1]
    ys = []
    for i, file in enumerate(files):
        y = []
        for run in [1]:
            one_y = 1-np.load(file)
            print(file, one_y.shape)
            y.append(np.stack([np.linspace(0,1,300)[:299], one_y], 1))

        runs = np.tile(np.arange(1, len(y)+1)[:, None], [1, y[-1].shape[0]]).reshape(-1)
        y = np.concatenate([np.array([labels[i] for _ in range(runs.shape[0])])[:,None], runs[:,None], np.concatenate(y)], 1)
        ys.append(y)

    ys = pd.DataFrame(data = np.concatenate(ys), columns = ['method', 'run', 'x', 'value'])
    ys.value = ys.value.astype(float)
    ys.x = ys.x.astype(float)
    print(ys)

    # plot predictive distribution
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    sns.lineplot(data=ys, x="x", y="value", hue='method', ci=None)
    ax.set_xlim(0., 0.999)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticklabels([1.0, 0.8, 0.6, 0.4, 0.2])
    # ax.set_title('CIFAR10+SVHN Error vs Uncertainty')
    ax.set_xlabel('Confidence Threshold $\\tau$')
    ax.set_ylabel('Error on examples with confidence $\leq \\tau$', fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:])
    ax.spines['bottom'].set_color('gray')
    ax.spines['top'].set_color('gray')
    ax.spines['right'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(color='lightgray', linestyle='--')

    plt.savefig('{}/acc_vs_conf.pdf'.format('/'.join(dir.split('/')[:-2])), format='pdf', dpi=1000, bbox_inches='tight')
