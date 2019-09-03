from os.path import join as pjoin
import os
import yaml
from unet_region.models.unet import UNet
import torch
import pandas as pd
import numpy as np
from unet_region.pascal_voc_loader_patch import pascalVOCLoaderPatch
from unet_region.patch_loader import PatchLoader
from unet_region import utils as utls
import matplotlib.pyplot as plt
import tqdm
from skimage import segmentation, draw, io, transform
from skimage.measure import label
import configargparse
from imgaug import augmenters as iaa
from unet_region.my_augmenters import rescale_augmenter
from sklearn.metrics import f1_score, roc_curve, auc
import params
import numpy as np
from collections import defaultdict

def get_closest_label(label_map):
    # take label that is the closest to center
    w, h = label_map.shape
    x, y = np.meshgrid(np.arange(-w // 2, w // 2),
                       np.arange(-h // 2, h // 2))
    dist = np.linalg.norm(np.stack((x, y)), axis=0)

    dist_label = np.stack((label_map.ravel(), dist.ravel())).T
    dist_label = dist_label[dist_label[:, 0] != 0, :]
    if(dist_label.size == 0):
        return None

    dist_label = dist_label[np.argsort(dist_label[:, 1]), :]
    return dist_label[0, 0]

def main(cfg):

    train_cfg = yaml.safe_load(open(pjoin(cfg.run_dir, 'cfg.yml'), 'r'))

    if ('frames' in train_cfg.keys()):
        frames_of_train = train_cfg['frames']
    else:
        frames_of_train = None

    cp_path = pjoin(cfg.run_dir, 'checkpoints', 'checkpoint.pth.tar')

    ds_dir = os.path.split(cfg.in_dir)[1]

    locs = utls.read_locs_csv(cfg.csv_loc_file)

    cp = torch.load(cp_path, map_location=lambda storage, loc: storage)

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    batch_to_device = lambda batch: {
        k: v.type(torch.float).to(device)
        if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    in_shape = [cfg.in_shape] * 2

    transf = iaa.Sequential([iaa.Resize(in_shape), rescale_augmenter])

    model = UNet(
        in_channels=3,
        out_channels=1,
        depth=4,
        cuda=cfg.cuda,
        with_coordconv=cfg.coordconv,
        with_coordconv_r=cfg.coordconv_r,
        with_batchnorm=cfg.batch_norm)

    model.load_state_dict(cp)
        
    model = model.eval()

    pred_patch_loader = PatchLoader(
        cfg.in_dir,
        truth_type='hand',
        locs=locs,
        augmentation=transf,
        patch_rel_size=cfg.patch_rel_size)

    pred_loader = torch.utils.data.DataLoader(
        pred_patch_loader,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        collate_fn=pred_patch_loader.collate_fn)

    print('Data: {}'.format(cfg.in_dir))
    thr = 0.5

    print('Taking threshold: {}'.format(thr))
    ims = []
    preds = []
    preds_mask = []
    truth_contours = []
    truths = []
    boxes = []
    opt_boxes = []
    frames = []

    print('Generating predictions')
    pbar = tqdm.tqdm(total=len(pred_loader))
    for i, data in enumerate(pred_loader):
        data = batch_to_device(data)
        pred_ = model(data['image']).detach().cpu().numpy()
        im_ = data['image'].cpu().numpy().transpose((0, 2, 3, 1))
        truth_ = data['label/segmentation'].cpu().numpy().transpose((0, 2, 3,
                                                                     1))

        ims += [im_[i, ...] for i in range(im_.shape[0])]
        preds += [pred_[i, 0, ...] for i in range(im_.shape[0])]
        preds_mask += [pred_[i, 0, ...] > thr for i in range(im_.shape[0])]

        opt_boxes += [
            utls.get_opt_box(p, rotations=True)['box']
            for p in preds_mask[-cfg.batch_size:]
        ]

        truths += [truth_[i, ..., 0] for i in range(truth_.shape[0])]

        truth_contours += [
            segmentation.find_boundaries(truth_[i, ..., 0], mode='thick')
            for i in range(im_.shape[0])
        ]

        frames += data['idx']
        boxes += data['box']
        pbar.update(1)

    pbar.close()

    prev_dir = pjoin(cfg.run_dir, ds_dir, 'preds')
    if (not os.path.exists(prev_dir)):
        os.makedirs(prev_dir)

    maps_dir = pjoin(cfg.run_dir, ds_dir, 'preds_map')
    if (not os.path.exists(maps_dir)):
        os.makedirs(maps_dir)

    truth_dir = pjoin(cfg.run_dir, ds_dir, 'truths')
    if (not os.path.exists(truth_dir)):
        os.makedirs(truth_dir)

    # make preview images
    alpha = 0.5
    for i in range(len(ims)):
        im_ = ims[i]
        pred_ = preds[i]
        truth_ = truths[i]
        pred_mask_ = preds_mask[i]
        opt_mask = opt_boxes[i].get_ellipse_mask(shape=pred_mask_.shape)
        opt_box_mask = segmentation.find_boundaries(opt_mask, mode='thick')
        all_ = (1 - alpha) * im_ + alpha * np.repeat(
            pred_mask_[..., np.newaxis], 3, -1) * (0., 0., 1.)
        rr, cc = draw.circle(im_.shape[0] // 2, im_.shape[1] // 2, radius=4)
        all_[rr, cc] = (0, 1., 0)

        idx_cont_gt = np.where(truth_contours[i])
        all_[idx_cont_gt[0], idx_cont_gt[1], :] = (1., 0., 0.)

        idx_cont_entrance = np.where(opt_box_mask)
        all_[idx_cont_entrance[0], idx_cont_entrance[1], :] = (0., 1., 0.)

        closed_truth = label(truth_)
        closed_truth = closed_truth == closed_truth[im_.shape[0] // 2,
                                                    im_.shape[1] // 2]
        closed_truth = closed_truth.astype(float)

        io.imsave(pjoin(prev_dir, 'frame_{:04d}.png'.format(i)), all_)
        io.imsave(pjoin(maps_dir, 'frame_{:04d}.png'.format(i)), pred_)
        io.imsave(pjoin(truth_dir, 'frame_{:04d}.png'.format(i)), closed_truth)

    # make entrance masks
    types = ['ellipse', 'closed', 'fixed', 'truth']
    frame_dirs = {
        s: pjoin(cfg.run_dir, ds_dir, 'entrance_masks', s)
        for s in types
    }

    for _, p in frame_dirs.items():
        if (not os.path.exists(p)):
            os.makedirs(p)

    print('Making entrance masks')
    masks_on_patch = []
    preds_on_patch = []
    masks_on_img = []

    for i in range(len(ims)):
        pred_mask_ = preds_mask[i]
        box = boxes[i]

        # make masks
        ell_mask = opt_boxes[i].resize(box.shape).get_ellipse_mask()
        closed_mask = label(pred_mask_)

        closest_label = get_closest_label(closed_mask)
        if(closest_label is not None):
            closed_mask = (closed_mask == closest_label).astype(float)
            closed_mask = transform.resize(
                closed_mask, box.shape, mode='reflect').astype(bool)
        else:
            closed_mask = np.zeros(box.shape).astype(bool)

        radius_mask = np.zeros(box.shape, dtype=bool)
        rr, cc = draw.circle(
            *(box.shape[0] // 2, box.shape[1] // 2),
            box.orig_shape[0] * cfg.fix_radius,
            shape=box.shape)
        radius_mask[rr, cc] = True

        truth = transform.resize(
            truths[i], box.shape, anti_aliasing=True,
            mode='reflect').astype(bool)

        pred_ = transform.resize(preds[i], box.shape, anti_aliasing=True,
                                 mode='reflect')

        masks_on_patch_ = {
            'ellipse': ell_mask,
            'closed': closed_mask,
            'fixed': radius_mask,
            'truth': truth
        }

        masks_on_img_ = {
            k: (box.paste_patch(v)).astype(bool)
            for k, v in masks_on_patch_.items()
        }

        for k, v in masks_on_img_.items():
            io.imsave(
                pjoin(frame_dirs[k], 'frame_{:04d}.png'.format(frames[i])),
                v.astype(np.uint8) * 255)

        masks_on_patch.append(masks_on_patch_)
        preds_on_patch.append(pred_)
        masks_on_img.append(masks_on_img_)

    print('Computing scores')
    fname = 'scores.csv'
    # concatenate all dicts
    dict_merged = defaultdict(list)
    for d in masks_on_patch:
        for k, v in d.items():
            dict_merged[k].append(v)

    # list to array
    dict_merged_arr = {}
    for k, v in dict_merged.items():
        dict_merged_arr[k] = np.stack(v)

    # compute scores (masks)
    truth = dict_merged_arr['truth']

    dict_scores = {}
    for k, v in dict_merged_arr.items():
        fpr, tpr, thresholds = roc_curve(truth.ravel(), v.ravel())
        dict_scores[k + '/fpr'] = fpr[1]
        dict_scores[k + '/tpr'] = tpr[1]

        f1 = f1_score(truth.ravel(), v.ravel())

        dict_scores[k + '/f1'] = f1

    # compute scores (pred)
    fpr, tpr, thresholds = roc_curve(truth.ravel(),
                                     np.ravel(preds_on_patch))
    dict_scores['auc'] = auc(fpr, tpr)

    path_scores = pjoin(cfg.run_dir, ds_dir, 'scores.csv')

    scores = pd.Series(dict_scores)
    print('saving scores to {}'.format(path_scores))
    scores.to_csv(path_scores)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--run-dir', required=True)
    p.add('--in-dir', required=True)
    p.add('--csv-loc-file', required=True)

    cfg = p.parse_args()

    main(cfg)
