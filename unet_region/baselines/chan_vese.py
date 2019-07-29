import os
from os.path import join as pjoin
import pandas as pd
from unet_region.baselines.unet import params
from unet_region.acm_utils import make_init_ls_gaussian
import glob
import numpy as np
from skimage.segmentation import chan_vese
from skimage import io
from skimage import measure
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import f1_score, roc_curve, auc

dirs = {
    'Dataset00_2019-05-29_19-29': [
        'Dataset04', 'Dataset01', 'Dataset02', 'Dataset00', 'Dataset03',
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


def run_chan_vese(f_names, radius):
    pbar = tqdm.tqdm(total=len(f_names))
    maps = []
    for f in f_names:
        im = io.imread(f)
        init_ls = make_init_ls_gaussian(im.shape[1] // 2, im.shape[0] // 2,
                                        im.shape, radius * np.max(im.shape))
        ls = np.logical_not(chan_vese(im, init_level_set=init_ls))
        map_ = measure.label(ls)
        label_ = map_[im.shape[0] // 2, im.shape[1] // 2]
        maps.append(map_ == label_)
        pbar.update(1)

    pbar.close()
    return maps


def main(cfg):

    for run_dir in dirs.keys():
        for s in dirs[run_dir]:
            exp_path = pjoin(cfg.root_dir, run_dir, s)
            print('Running Chan-Vese using predictions of {}'.format(exp_path))
            frames = sorted(
                glob.glob(
                    pjoin(exp_path, 'preds_map', '*.png')))

            truths = sorted(
                glob.glob(pjoin(cfg.root_dir, run_dir, s, 'truths', '*.png')))

            truths = np.array([io.imread(f) for f in truths]).astype(bool)
            maps = np.array(run_chan_vese(frames, 0.05)).astype(int)

            out_dir = pjoin(cfg.root_dir, run_dir, s, 'chan_vese_maps')

            if (not os.path.exists(out_dir)):
                os.makedirs(out_dir)

            # save chan-vese maps
            for i, m in enumerate(maps):
                io.imsave(pjoin(out_dir, 'frame_{:04d}.png'.format(i)), m)

            # compute scores
            fpr, tpr, thresholds = roc_curve(truths.ravel(), maps.ravel())
            f1 = f1_score(truths.ravel(), maps.ravel())
            auc_ = auc(fpr, tpr)

            path_scores = pjoin(cfg.root_dir,
                                run_dir,
                                s,
                                'scores_chan_vese.csv')

            dict_scores = {'closed/fpr': fpr[1],
                           'closed/tpr': tpr[1],
                           'closed/f1': f1,
                           'auc': auc_}
            scores = pd.Series(dict_scores)
            print('saving scores to {}'.format(path_scores))
            scores.to_csv(path_scores)


if __name__ == "__main__":

    p = params.get_params()
    p.add('--root-dir')
    cfg = p.parse_args()
    cfg.n_workers = 0

    cfg.root_dir = '/home/ubelix/data/medical-labeling/unet_region/runs'

    main(cfg)
