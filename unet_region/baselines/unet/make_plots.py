from os.path import join as pjoin
import pandas as pd
import seaborn as sns
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

root_dir = '/home/ubelix/data/medical-labeling/unet_region/runs'
out_dir = '/home/ubelix/data/medical-labeling/unet_region'
seq_types = ['tweezer', 'cochlea', 'slitlamp', 'brain']


def dir_name_to_type(dir_name):
    digit_to_type_mapping = {i: seq_types[i] for i in range(len(seq_types))}
    seqs_dict = {v: [] for _, v in digit_to_type_mapping.items()}

    m = re.search('Dataset(.)', dir_name)
    return seq_types[int(m.group(1))]


def make_dataframe(root_dir, run_dir, dirs, fname, method):
    # store scores
    pds = []
    for dir_ in dirs:
        seq_type = dir_name_to_type(dir_)
        scores = np.genfromtxt(
            pjoin(root_dir, run_dir, dir_, fname), delimiter=',', dtype=None)

        index = pd.MultiIndex.from_tuples([(seq_type, dir_)],
                                          names=['seq_type', 'seq_n'])
        columns = [r[0].decode('utf-8') for r in scores]
        scores = [r[1] for r in scores]
        df = pd.DataFrame(scores, index=columns, columns=index)
        df = pd.concat([df], keys=[method], names=['method'], axis=1)
        pds.append(df)
    return pds


paths = {
    ('pascal_2019-05-29_08-20', 'pascal'): [
        'Dataset00', 'Dataset01', 'Dataset02', 'Dataset03', 'Dataset04',
        'Dataset05', 'Dataset20', 'Dataset21', 'Dataset22', 'Dataset23',
        'Dataset24', 'Dataset25', 'Dataset10', 'Dataset11', 'Dataset12',
        'Dataset13', 'Dataset20', 'Dataset21', 'Dataset22', 'Dataset23',
        'Dataset24', 'Dataset25', 'Dataset30', 'Dataset31', 'Dataset32',
        'Dataset33', 'Dataset34'
    ],
    ('Dataset00_2019-05-29_19-29', 'pascal_1f'): ['Dataset00', 'Dataset01', 'Dataset02',
                                   'Dataset03', 'Dataset04', 'Dataset05'],
    ('Dataset20_2019-05-29_21-33', 'pascal_1f'): ['Dataset20', 'Dataset21', 'Dataset22',
                                   'Dataset23', 'Dataset24', 'Dataset25'],
   ('Dataset30_2019-05-29_22-35', 'pascal_1f') : ['Dataset30', 'Dataset31', 'Dataset32',
                                   'Dataset33', 'Dataset34'],
    ('Dataset10_2019-05-29_20-31', 'pascal_1f'):
                   ['Dataset10', 'Dataset11', 'Dataset12', 'Dataset13']
}

dfs = []
for k, v in paths.items():
    df = pd.concat(
        make_dataframe(root_dir, k[0], v, 'scores.csv', k[1]), axis=1)
    dfs.append(df)

for k, v in paths.items():
    df = pd.concat(
        make_dataframe(root_dir, k[0], v, 'scores_chan_vese.csv', k[1]), axis=1)
    dfs.append(df)

dfs = pd.concat(dfs, axis=1)

import pdb; pdb.set_trace() ## DEBUG ##
metrics_to_plot = ['closed/f1', 'auc']

df_to_plot = pd.concat([dfs.loc[m] for m in metrics_to_plot], axis=1)
# df_to_plot = df_to_plot.T.xs('tweezer', level=1, axis=1).T
df_to_plot = df_to_plot.reset_index()
# df_to_plot = df_to_plot.loc['tweezer']
# ax = sns.boxplot(x="metric", y="score", hue="models",
#                  data=df_to_plot, palette="Set3")
# plt.show()

sns.set_style("whitegrid")
fig, ax = plt.subplots(len(metrics_to_plot), 1, figsize=(9, 8))
ax = ax.flatten()
for m, a in zip(metrics_to_plot, ax):
    sns.boxplot(
        y=m, x='seq_type', hue='method', data=df_to_plot, palette="Set3", ax=a)
    a.set(ylabel='Score', xlabel='Type', title=m)
    # ax.set_yticks(np.linspace(0, 1, 20).tolist(), minor=True)
    # ax.set_yticks(np.linspace(0, 1, 10))
    a.set_yticks(np.linspace(0, 1, 11))
    a.legend(loc='center right', bbox_to_anchor=(1.20, 0.8), borderaxespad=0.)
fig.tight_layout()
# Put the legend out of the figure

fig.savefig(pjoin(out_dir, 'all.png'))
fig.show()
