from os.path import join as pjoin
import os
import yaml
from pytorch_utils.models.unet import UNet
import munch
import torch
import pandas as pd
import numpy as np
from pytorch_utils.pascal_voc_loader_patch import pascalVOCLoaderPatch
from pytorch_utils.pascal_voc_loader_patch import collate_fn_pascal_patch
from pytorch_utils.patch_loader import collate_fn_patch
from pytorch_utils.patch_loader import PatchLoader
from pytorch_utils import utils as utls
import matplotlib.pyplot as plt
import tqdm
from skimage import segmentation, draw, io, transform
from skimage.measure import label
import configargparse
from imgaug import augmenters as iaa
from my_augmenters import rescale_augmenter
from sklearn.metrics import f1_score, roc_curve
import params
import numpy as np
from collections import defaultdict

p = params.get_params()

p.add('--run-dir', required=True)
p.add('--in-dir', required=True)
p.add('--csv-loc-file', required=True)

cfg = p.parse_args()

train_cfg = yaml.safe_load(open(pjoin(cfg.run_dir, 'cfg.yml'), 'r'))

if ('frames' in train_cfg.keys()):
    frames_of_train = train_cfg['frames']
else:
    frames_of_train = None


def test_fn(x, y, z):
    print('lala')

cp_path = pjoin(cfg.run_dir, 'checkpoints', 'best_model.pth.tar')

ds_dir = os.path.split(cfg.in_dir)[1]

locs = utls.read_locs_csv(cfg.csv_loc_file)

cp = torch.load(cp_path, map_location=lambda storage, loc: storage)

device = torch.device('cuda' if cfg.cuda else 'cpu')

batch_to_device = lambda batch: {
    k: v.type(torch.float).to(device) if (isinstance(v, torch.Tensor)) else v
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

model.load_state_dict(cp['state_dict'])
model.eval()

# Find threhsold on validation set
if ('data_type' in train_cfg.keys()):
    in_dir_of_train = train_cfg['in_dir']

    val_patch_loader = PatchLoader(
        cfg,
        in_dir_of_train,
        truth_type='hand',
        fix_frames=frames_of_train,
        fake_len=100,
        img_size=cfg.in_shape,
        augmentation=transf,
        patch_rel_size=cfg.patch_rel_size)

    val_loader = torch.utils.data.DataLoader(
        val_patch_loader,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        collate_fn=collate_fn_patch)
else:
    val_patch_loader = pascalVOCLoaderPatch(
        '/home/laurent.lejeune/medical-labeling/',
        augmentations=transf,
        patch_rel_size=cfg.patch_rel_size)

    # pick random images
    inds = np.random.choice(np.arange(len(val_patch_loader)), 100)
    val_loader = torch.utils.data.DataLoader(
        val_patch_loader,
        batch_size=cfg.batch_size,
        # num_workers=cfg.n_workers,
        num_workers=0,
        sampler=torch.utils.data.SubsetRandomSampler(inds.tolist()),
        collate_fn=collate_fn_pascal_patch)

pred_patch_loader = PatchLoader(
    cfg,
    cfg.in_dir,
    truth_type='hand',
    locs=locs,
    img_size=cfg.in_shape,
    augmentation=transf,
    patch_rel_size=cfg.patch_rel_size)

pred_loader = torch.utils.data.DataLoader(
    pred_patch_loader,
    batch_size=cfg.batch_size,
    num_workers=cfg.n_workers,
    collate_fn=collate_fn_patch)

print('Data: {}'.format(cfg.in_dir))
print('Searching for optimal threshold')
# Search these thresholds
thr = np.linspace(0.3, 0.95, 40)

pred_scores_dict = {t: [] for t in thr}
pbar = tqdm.tqdm(total=len(val_loader))
for i, data in enumerate(val_loader):
    data = batch_to_device(data)
    pred_scores = []
    pred_ = torch.sigmoid(model(data['image'])).detach().cpu().numpy()
    im_ = data['image'].cpu().numpy().transpose((0, 2, 3, 1))
    truth_ = data['label/segmentation'].cpu().numpy()
    for t in thr:
        s_ = f1_score(truth_.ravel(), pred_.ravel() > t)
        pred_scores_dict[t].append(s_)
    pbar.update(1)

pbar.close()

pred_scores_dict = {k: np.mean(v) for k, v in pred_scores_dict.items()}

pred_pd = pd.DataFrame.from_dict(
    pred_scores_dict, columns=['f1'], orient='index')
path_scores = pjoin(cfg.run_dir, 'thr_pred_scores.csv')
print('saving scores to {}'.format(path_scores))
pred_pd.to_csv(path_scores)

thr = thr[np.argmax(
    [pred_scores_dict[k] for k, _ in pred_scores_dict.items()])]
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
    pred_ = torch.sigmoid(model(data['image'])).detach().cpu().numpy()
    im_ = data['image'].cpu().numpy().transpose((0, 2, 3, 1))
    truth_ = data['label/segmentation'].cpu().numpy().transpose((0, 2, 3, 1))

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

frame_dir = pjoin(cfg.run_dir, ds_dir, 'preds')
if (not os.path.exists(frame_dir)):
    os.makedirs(frame_dir)

# make preview images
alpha = 0.5
print('Making preview images in {}'.format(frame_dir))
for i in range(len(ims)):
    im_ = ims[i]
    pred_ = preds[i]
    pred_mask_ = preds_mask[i]
    opt_mask = opt_boxes[i].get_ellipse_mask(shape=pred_mask_.shape)
    opt_box_mask = segmentation.find_boundaries(opt_mask, mode='thick')
    truth_contours_ = truth_contours[i]
    all_ = (1 - alpha) * im_ + alpha * np.repeat(pred_mask_[..., np.newaxis],
                                                 3, -1) * (0., 0., 1.)
    rr, cc = draw.circle(im_.shape[0] // 2, im_.shape[1] // 2, radius=4)
    all_[rr, cc] = (0, 1., 0)

    idx_cont_gt = np.where(truth_contours[i])
    all_[idx_cont_gt[0], idx_cont_gt[1], :] = (1., 0., 0.)

    idx_cont_entrance = np.where(opt_box_mask)
    all_[idx_cont_entrance[0], idx_cont_entrance[1], :] = (0., 1., 0.)

    io.imsave(pjoin(frame_dir, 'frame_{:04d}.png'.format(i)), all_)

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
masks_on_img = []

for i in range(len(ims)):
    pred_ = preds[i]
    pred_mask_ = preds_mask[i]
    
    box = boxes[i]

    # make masks
    ell_mask = opt_boxes[i].resize(box.shape).get_ellipse_mask()
    closed_mask = label(pred_mask_)
    closed_mask = closed_mask == closed_mask[closed_mask.shape[0] // 2,
                                             closed_mask.shape[1] // 2]
    closed_mask = transform.resize(
        closed_mask,
        box.shape,
        mode='reflect').astype(bool)

    radius_mask = np.zeros(box.shape, dtype=bool)
    rr, cc = draw.circle(
        *(box.shape[0] // 2, box.shape[1] // 2),
        box.orig_shape[0] * cfg.fix_radius,
        shape=box.shape)
    radius_mask[rr, cc] = True

    truth = transform.resize(
        truths[i],
        box.shape,
        anti_aliasing=True,
        mode='reflect').astype(bool)

    masks_on_patch_ = {
        'ellipse': ell_mask,
        'closed': closed_mask,
        'fixed': radius_mask,
        'truth': truth
    }

    masks_on_img_ = {k: (box.paste_patch(v)).astype(bool)
             for k, v in masks_on_patch_.items()}

    for k, v in masks_on_img_.items():
        io.imsave(
            pjoin(frame_dirs[k], 'frame_{:04d}.png'.format(frames[i])),
            v.astype(np.uint8) * 255)

    masks_on_patch.append(masks_on_patch_)
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

# compute scores
truth = dict_merged_arr['truth']

dict_scores = {}
for k, v in dict_merged_arr.items():
    fpr, tpr, thresholds = roc_curve(
        truth.ravel(),
        v.ravel())
    dict_scores[k + '/fpr'] = fpr[1]
    dict_scores[k + '/tpr'] = tpr[1]

    f1 = f1_score(truth.ravel(),
                    v.ravel())
    dict_scores[k + '/f1'] = f1

path_scores = pjoin(cfg.run_dir, ds_dir, 'scores.csv')

scores = pd.Series(dict_scores)
print('saving scores to {}'.format(path_scores))
scores.to_csv(path_scores)
