from pytorch_utils.pascal_voc_loader_patch import pascalVOCLoaderPatch
from pytorch_utils.patch_loader import PatchLoader
from pytorch_utils.patch_loader import collate_fn_patch
from pytorch_utils.pascal_voc_loader_patch import collate_fn_pascal_patch
from pytorch_utils.models.unet import UNet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from os.path import join as pjoin
import os
import datetime
import yaml
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
import pandas as pd
from trainer import Trainer
import configargparse
from my_augmenters import rescale_augmenter

p = configargparse.ArgParser(
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    default_config_files=['default.yaml'])

p.add('-v', help='verbose', action='store_true')

p.add('--data-type')
p.add('--frames', action='append', type=int)
p.add('--n-patches', type=int)
p.add('--checkpoint-path')
p.add('--epochs', type=int)
p.add('--lr', type=float)
p.add('--momentum', type=float)
p.add('--alpha', type=float)
p.add('--eps', type=float)
p.add('--ds-split', type=float)
p.add('--ds-shuffle', type=bool)
p.add('--weight-decay', type=float)
p.add('--batch-size', type=int)
p.add('--batch-norm', type=bool)
p.add('--n-workers', type=int)
p.add('--cuda', default=False, action='store_true')
p.add('--out-dir', required=True)
p.add('--in-dir', required=True)
p.add('--coordconv', type=bool)
p.add('--coordconv-r', type=bool)
p.add('--in-shape', type=int)
p.add('--loss-size', type=float)
p.add('--patch-rel-size', type=float)
p.add('--aug-noise', type=float)
p.add('--aug-flip-proba', type=float)
p.add('--aug-some', type=int)

cfg = p.parse_args()

print(cfg)

d = datetime.datetime.now()
run_dir = pjoin(cfg.out_dir, 'runs', '{:%Y-%m-%d_%H-%M-%S}'.format(d))

in_shape = [cfg.in_shape] * 2

transf = iaa.Sequential([
    iaa.SomeOf(3, [
        iaa.Affine(rotate=iap.Uniform(-15., 15.)),
        iaa.Affine(shear=iap.Uniform(-15., -15.)),
        iaa.Fliplr(1.),
        iaa.Flipud(1.),
        iaa.GaussianBlur(sigma=iap.Uniform(0.0, 0.05))
    ]),
    iaa.Resize(in_shape),
    rescale_augmenter
])

if cfg.data_type == 'pascal':
    loader = pascalVOCLoaderPatch(
        cfg.in_dir,
        patch_rel_size=cfg.patch_rel_size,
        augmentation=transf,
        cuda=cfg.cuda,
        do_reshape=True,
        img_size=cfg.in_shape)
    collate_fn = collate_fn_pascal_patch
elif cfg.data_type == 'medical':
    loader = PatchLoader(
        cfg,
        cfg.in_dir,
        'hand',
        make_opt_box=False,
        fix_frames=cfg.frames,
        augmentation=transf)
    collate_fn = collate_fn_patch
else:
    raise Exception('data-type must be pascal or medical')

# Creating data indices for training and validation splits:
validation_split = 1 - cfg.ds_split
random_seed = 42

if (len(cfg.frames) == 0):
    indices = list(range(len(loader)))
else:
    indices = np.random.choice(
        list(range(len(cfg.frames))), size=cfg.n_patches, replace=True)

split = int(np.floor(validation_split * len(indices)))

np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

if (not os.path.exists(run_dir)):
    os.makedirs(run_dir)

pd.DataFrame(train_indices).to_csv(pjoin(run_dir, 'train_sample.csv'))
pd.DataFrame(val_indices).to_csv(pjoin(run_dir, 'val_sample.csv'))

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(
    loader,
    batch_size=cfg.batch_size,
    num_workers=cfg.n_workers,
    collate_fn=collate_fn,
    sampler=train_sampler)

if (cfg.data_type == 'pascal'):
    val_loader = torch.utils.data.DataLoader(
        loader,
        num_workers=cfg.n_workers,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        sampler=valid_sampler)
else:
    val_loader = train_loader

dataloaders = {'train': train_loader, 'val': val_loader}

model = UNet(
    in_channels=3,
    out_channels=1,
    depth=4,
    cuda=cfg.cuda,
    with_coordconv=cfg.coordconv,
    with_coordconv_r=cfg.coordconv_r,
    with_batchnorm=cfg.batch_norm)

# Save cfg
with open(pjoin(run_dir, 'cfg.yml'), 'w') as outfile:
    yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

trainer = Trainer(model, dataloaders, cfg, run_dir)
trainer.train()
