from os.path import join as pjoin
import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

root_dir = '/home/laurent.lejeune/medical-labeling/unet_region/runs'
out_dir = '/home/laurent.lejeune/medical-labeling/unet_region'
seq_types = ['tweezer', 'cochlea', 'slitlamp', 'brain']

def seqs_to_type(seqs):
    digit_to_type_mapping = {i: seq_types[i]
                             for i in range(len(seq_types))
    }
    seqs_dict = {v: [] for _, v in digit_to_type_mapping.items()}

    for s in seqs:
        m = re.search('Dataset(.)', s)
        seqs_dict[digit_to_type_mapping[int(m.group(1))]].append(s)

    # delete empty
    seqs_dict = {k: v for k, v in seqs_dict.items()
                 if(len(seqs_dict[k]) > 0)}

    return seqs_dict



dirs = {
    'pascal_2019-05-29_08-20': [
        'Dataset00', 'Dataset01', 'Dataset02', 'Dataset03', 'Dataset04',
        'Dataset05', 'Dataset10', 'Dataset11', 'Dataset12', 'Dataset13',
        'Dataset20', 'Dataset21', 'Dataset22', 'Dataset23', 'Dataset24',
        'Dataset25', 'Dataset30', 'Dataset31', 'Dataset32', 'Dataset33',
        'Dataset34'
    ],
    'Dataset00_2019-05-29_19-29': [
        'Dataset00', 'Dataset01', 'Dataset02', 'Dataset03', 'Dataset04',
        'Dataset05'
    ],
    'Dataset10_2019-05-29_20-31':
    ['Dataset10', 'Dataset11', 'Dataset12', 'Dataset13'],
    'Dataset20_2019-05-29_21-33': [
        'Dataset20', 'Dataset21', 'Dataset22', 'Dataset23', 'Dataset24',
        'Dataset25'
    ],
    'Dataset30_2019-05-29_22-35':
    ['Dataset30', 'Dataset31', 'Dataset32', 'Dataset33', 'Dataset34']
}

scores_dict = defaultdict(lambda: defaultdict(pd.DataFrame))

# store scores
for run_dir in dirs.keys():
    wrt_type = seqs_to_type(dirs[run_dir])
    for t in wrt_type.keys():
        pds = []
        for i, s in enumerate(wrt_type[t]):
            scores = pd.read_csv(
                pjoin(root_dir, run_dir, wrt_type[t][i], 'scores.csv'),
                names=['type', wrt_type[t][i]]).set_index('type')
            pds.append(scores)
            
        scores_dict[t][run_dir] = pd.concat(pds, axis=1)

metrics_to_plot = ['ellipse/f1',
                   # 'ellipse/tpr',
                   'fixed/f1',
                   # 'fixed/tpr',
                   'closed/f1',
                   'auc'
]

for i, t in enumerate(scores_dict.keys()):
    fig, ax = plt.subplots(1, len(scores_dict[t].keys()))
    for j, m in enumerate(scores_dict[t].keys()):

        # plot pascal only
        boxes = []
        for metric in metrics_to_plot:
            boxes.append(scores_dict[t][m].loc[metric])
        ax[j].boxplot(boxes,
                      positions=np.arange(len(metrics_to_plot)))
        ax[j].yaxis.grid(True,
                            linestyle='-',
                            which='both',
                            color='lightgrey',
                            alpha=0.5)
        ax[j].minorticks_on()
        ax[j].set_title('{} on {}'.format(m, t))

    plt.setp(ax, xticklabels=metrics_to_plot, ylim=[0, 1])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig(pjoin(out_dir, '{}.png'.format(t)))
    
