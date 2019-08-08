from unet_region.pascal_voc_loader_patch import pascalVOCLoaderPatch
from unet_region.sub_sampler import SubsetSampler
from unet_region.patch_loader import PatchLoader
from unet_region.models.unet import UNet
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
from my_augmenters import rescale_augmenter
import params


def main(cfg):

    d = datetime.datetime.now()

    if (cfg.data_type == 'medical'):
        ds_dir = os.path.split(cfg.in_dir)[-1]
    else:
        ds_dir = cfg.data_type

    in_shape = [cfg.in_shape] * 2

    transf = iaa.Sequential([
        iaa.Invert(0.5) if 'Dataset1' in cfg.in_dir else iaa.Noop(),
        iaa.SomeOf(3, [
            iaa.Affine(rotate=iap.Uniform(-15., 15.)),
            iaa.Affine(shear=iap.Uniform(-15., -15.)),
            iaa.Fliplr(1.),
            iaa.Flipud(1.),
            iaa.GaussianBlur(sigma=iap.Uniform(0.0, 0.05))
        ]),
        iaa.Resize(in_shape), rescale_augmenter
    ])

    if cfg.data_type == 'pascal':
        loader = pascalVOCLoaderPatch(
            cfg.in_dir,
            patch_rel_size=cfg.patch_rel_size,
            augmentations=transf)
    elif cfg.data_type == 'medical':
        loader = PatchLoader(
            cfg.in_dir,
            'hand',
            fake_len=cfg.fake_len,
            make_opt_box=False,
            fix_frames=cfg.frames,
            augmentation=transf)
    else:
        raise Exception('data-type must be pascal or medical')

    # Creating data indices for training and validation splits:
    validation_split = 1 - cfg.ds_split

    if (cfg.frames is None):
        indices = list(range(len(loader)))
    else:
        indices = np.random.choice(
            list(range(len(cfg.frames))), size=cfg.n_patches, replace=True)

    split = int(np.floor(validation_split * len(indices)))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # train_sampler = SubsetRandomSampler(train_indices)
    train_sampler = SubsetSampler(train_indices)

    # keep validation set consistent accross epochs
    valid_sampler = SubsetSampler(val_indices)

    # each batch will give different locations / augmentations
    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        # num_workers=0,
        collate_fn=loader.collate_fn,
        worker_init_fn=loader.worker_init_fn,
        sampler=train_sampler)

    # each batch will give same locations / augmentations
    val_loader = torch.utils.data.DataLoader(
        loader,
        num_workers=cfg.n_workers,
        # num_workers=0,
        batch_size=cfg.batch_size,
        collate_fn=loader.collate_fn,
        worker_init_fn=loader.worker_init_fn_dummy,
        sampler=valid_sampler)

    for e in range(2):
        for i, d in enumerate(train_loader):
            print('[train]: epoch {}, batch {}, loc {}, idx {}'.format(e, i, d['loc'], d['label/idx']))
            if(i > 2):
                break

    for e in range(2):
        for i, d in enumerate(val_loader):
            print('[val]: epoch {}, batch {}, loc {}'.format(e, i, d['loc']))
            if(i > 10):
                break

    return cfg


p = params.get_params()

p.add('--out-dir')
p.add('--in-dir')
cfg = p.parse_args()

root_dir = '/home/laurent.lejeune/medical-labeling'

cuda = False

cfg.out_dir = pjoin(root_dir, 'unet_region')
cfg.in_dir = pjoin(root_dir, 'Dataset00')
cfg.data_type = 'medical'
cfg.cuda = cuda
cfg.checkpoint_path = None


main(cfg)
