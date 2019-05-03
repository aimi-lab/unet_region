from os.path import join as pjoin
import os
import yaml
from pytorch_utils.models.unet import UNet
import munch
import torch
import pandas as pd
import numpy as np
from pytorch_utils.pascal_voc_loader_patch import pascalVOCLoaderPatch
from pytorch_utils.patch_loader import collate_fn_patch
from pytorch_utils.patch_loader import PatchLoader
from pytorch_utils import utils as utls
import matplotlib.pyplot as plt
import tqdm
from skimage import segmentation, draw, io, transform
import configargparse

p = configargparse.ArgParser(
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    default_config_files=['default.yaml'])

p.add('-v', help='verbose', action='store_true')

p.add('--run-dir', required=True)
p.add('--in-dir', required=True)
p.add('--csv-loc-file', required=True)
p.add('--coordconv', type=bool)
p.add('--coordconv-r', type=bool)
p.add('--in-shape', type=int)
p.add('--cuda', type=bool)

cfg = p.parse_args()

cp_path = pjoin(cfg.run_dir, 'checkpoints', 'checkpoint.pth.tar')

ds_dir = os.path.split(cfg.in_dir)[1]

locs = utls.read_locs_csv(cfg.csv_loc_file)

cp = torch.load(cp_path, map_location=lambda storage, loc: storage)

device = torch.device('cuda' if cfg.cuda else 'cpu')

batch_to_device = lambda batch: {
    k: v.type(torch.float).to(device) if (isinstance(v, torch.Tensor)) else v
    for k, v in batch.items()
}

model = UNet(
    in_channels=3,
    out_channels=1,
    depth=4,
    cuda=cfg.cuda,
    with_coordconv=cfg.coordconv,
    with_coordconv_r=cfg.coordconv_r,
    with_batchnorm=cfg.batch_norm)

model.load_state_dict(cp['state_dict'])
model.eval()

# Find threhsold on validation set
patch_loader = PatchLoader(
    pjoin(cfg.in_dir, 'input-frames'),
    locs,
    truth_dir=pjoin(cfg.in_dir, 'ground_truth-frames'),
    do_reshape=True,
    img_size=cfg.in_shape,
    patch_rel_size=cfg.patch_rel_size)

loader = torch.utils.data.DataLoader(
    patch_loader,
    batch_size=cfg.batch_size,
    # num_workers=params.num_workers)
    collate_fn=collate_fn_patch,
    num_workers=0)

thr = 0.78

ims = []
preds = []
preds_mask = []
truth_contours = []
truths = []
boxes = []
frames = []

pbar = tqdm.tqdm(total=len(loader))
for i, data in enumerate(loader):
    data = batch_to_device(data)
    pred_ = torch.sigmoid(model(data['image'])).detach().cpu().numpy()
    im_ = data['image'].cpu().numpy().transpose((0, 2, 3, 1))
    truth_ = data['label/segmentation'].cpu().numpy().transpose((0, 2, 3, 1))

    ims += [im_[i, ...] for i in range(im_.shape[0])]
    preds += [pred_[i, 0, ...] for i in range(im_.shape[0])]
    preds_mask += [pred_[i, 0, ...] > thr for i in range(im_.shape[0])]
    truth_contours += [
        segmentation.find_boundaries(truth_[i, ..., 0], mode='thick')
        for i in range(im_.shape[0])
    ]

    frames += data['frame_num']
    boxes += data['bbox']
    pbar.update(1)

pbar.close()

frame_dir = pjoin(cfg.run_dir, ds_dir, 'preds')
if(not os.path.exists(frame_dir)):
    os.makedirs(frame_dir)

# make preview images
alpha = 0.5
for i in range(len(ims)):
    im_ = ims[i]
    pred_ = preds[i]
    pred_mask_ = preds_mask[i]
    truth_contours_ = truth_contours[i]
    all_ = (1-alpha)*im_ + alpha * np.repeat(pred_mask_[..., np.newaxis], 3,
                                   -1) * (0., 0., 1.)
    rr, cc = draw.circle(im_.shape[0] // 2, im_.shape[1] // 2, radius=4)
    all_[rr, cc] = (0, 1., 0)

    idx_cont_gt = np.where(truth_contours[i])
    all_[idx_cont_gt[0],idx_cont_gt[1],:] = (1., 0., 0.)

    io.imsave(pjoin(frame_dir, 'frame_{:04d}.png'.format(i)),
              all_)

# make entrance masks
frame_dir = pjoin(cfg.run_dir, ds_dir, 'entrance_masks')

if(not os.path.exists(frame_dir)):
    os.makedirs(frame_dir)

for i in range(len(ims)):
    im_ = ims[i]
    pred_ = preds[i]
    box = boxes[i]
    pred_mask_ = transform.resize(preds_mask[i],
                                  box.shape,
                                  anti_aliasing=True,
                                  mode='reflect').astype(np.uint8)

    mask = box.paste_patch(pred_mask_) * 255
    io.imsave(pjoin(frame_dir, 'frame_{:04d}.png'.format(frames[i])),
              mask)
