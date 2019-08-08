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

    print('Running Chan-Vese using predictions of {}'.format(cfg.preds_dir))
    frames = sorted(
        glob.glob(
            pjoin(cfg.preds_dir, 'preds_map', '*.png')))

    truths = sorted(
        glob.glob(pjoin(cfg.preds_dir, 'truths', '*.png')))

    truths = np.array([io.imread(f) for f in truths]).astype(bool)
    maps = np.array(run_chan_vese(frames, 0.05)).astype(int)

    out_dir = pjoin(cfg.preds_dir, 'chan_vese_maps')

    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    # save chan-vese maps
    for i, m in enumerate(maps):
        io.imsave(pjoin(out_dir, 'frame_{:04d}.png'.format(i)), m)

    # compute scores
    fpr, tpr, thresholds = roc_curve(truths.ravel(), maps.ravel())
    f1 = f1_score(truths.ravel(), maps.ravel())
    auc_ = auc(fpr, tpr)

    path_scores = pjoin(cfg.preds_dir,
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
    p.add('--preds-dir')
    cfg = p.parse_args()

    chan_vese(cfg)
