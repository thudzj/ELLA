import os
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})

import pandas as pd
import numpy.random as rnd
import seaborn as sns


for title in ['cifar10_resnet20', 'cifar10_resnet32', 'cifar10_resnet44', 'cifar10_resnet56']:

    files = [os.path.join('./logs/{}/{}/default/'.format(title.split("_")[0], title), 'corrupted_results_map.npy')]
    for job_id in ['default', 'lastl-full', 'lastl-kron', 'all-kron', 'all-kron']:
        dir = './logs/{}/{}/{}/'.format(title.split("_")[0], title, job_id)
        files.append(os.path.join(dir, 'corrupted_results_{}.npy'.format('ella' if job_id == 'default' else job_id)))
    labels = ['MAP', 'ELLA', 'LLA-LastL', 'LLA-LastL-KFAC', 'LLA', 'LLA-KFAC']


    for typ in ['Negative Log-likelihood', 'Accuracy', 'Expected Calibration Error']:
        if typ == 'Negative Log-likelihood':
            idx = 0
        elif typ == 'Accuracy':
            idx = 1
        elif typ == 'Expected Calibration Error':
            idx = 2

        data = np.stack([np.load(file)[:, :, idx] for file in files], 0)
        intensity = np.tile(np.arange(1, 1+data.shape[1]).reshape(-1, 1), (1, data.shape[2])).astype(np.int)
        data = np.concatenate([data.reshape(data.shape[0], -1), intensity.reshape(1, -1)], 0).T
        df = pd.DataFrame(data = data, columns = labels + ['Skew intensity'])
        print(df)
        df_plot = df.melt(id_vars='Skew intensity', value_vars=labels)

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)
        sns_plot = sns.boxplot(x='Skew intensity', y='value', hue='variable', data=df_plot, showfliers=False, linewidth=1)
        sns_plot.set(
            ylabel=typ
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels)
        ax.set_xticklabels([int(float(t.get_text()))  for t in ax.get_xticklabels()])
        ax.spines['bottom'].set_color('gray')
        ax.spines['top'].set_color('gray')
        ax.spines['right'].set_color('gray')
        ax.spines['left'].set_color('gray')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('Skew intensity', fontsize=20)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='lightgray', linestyle='--')
        plt.savefig('{}/{}_corruption_{}'.format('/'.join(dir.split('/')[:-2]), title, idx) +'.pdf', format='pdf', dpi=1000, bbox_inches='tight')
