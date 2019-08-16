from unet_region.pascal_voc_loader_patch import pascalVOCLoaderPatch
from unet_region.patch_loader import PatchLoader
from unet_region.baselines.rbf.rbf import RBFNet
from unet_region.models.unet import UNet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import RandomSampler
from unet_region.sub_sampler import SubsetSampler
import torch
from os.path import join as pjoin
import os
import datetime
import yaml
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
import pandas as pd
from unet_region.my_augmenters import rescale_augmenter
from unet_region.baselines.rbf import params


def main(cfg):

    d = datetime.datetime.now()

    if (cfg.data_type == 'medical'):
        ds_dir = os.path.split(cfg.in_dir)[-1]
    else:
        ds_dir = cfg.data_type

    run_dir = pjoin(cfg.out_dir, '{}_{:%Y-%m-%d_%H-%M}'.format(
        ds_dir, d))

    in_shape = [cfg.in_shape] * 2

    transf = iaa.Sequential([
        iaa.Invert(0.5) if 'Dataset1' in ds_dir else iaa.Noop(),
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
            pjoin(cfg.in_dir, 'VOCdevkit'),
            patch_rel_size=cfg.patch_rel_size,
            augmentations=transf)
    elif cfg.data_type == 'medical':
        loader = PatchLoader(
            cfg.in_dir,
            'hand',
            fake_len=cfg.fake_len,
            fix_frames=cfg.frames,
            augmentation=transf)
    else:
        raise Exception('data-type must be pascal or medical')

    # Creating data indices for training and validation splits:
    validation_split = 1 - cfg.ds_split
    random_seed = 42

    if (cfg.frames is None):
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

    # keep validation set consistent accross epochs
    valid_sampler = SubsetSampler(val_indices)

    # each batch will give different locations / augmentations
    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        collate_fn=loader.collate_fn,
        worker_init_fn=loader.worker_init_fn,
        sampler=train_sampler)

    # each batch will give same locations / augmentations
    val_loader = torch.utils.data.DataLoader(
        loader,
        num_workers=cfg.n_workers,
        batch_size=cfg.batch_size,
        collate_fn=loader.collate_fn,
        worker_init_fn=loader.worker_init_fn_dummy,
        sampler=valid_sampler)

    # loader for previewing images
    prev_sampler = SubsetRandomSampler(val_indices)
    prev_loader = torch.utils.data.DataLoader(
        loader,
        num_workers=cfg.n_workers,
        collate_fn=loader.collate_fn,
        sampler=prev_sampler,
        batch_size=4,
        drop_last=True)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'prev': prev_loader
    }

    model = RBFNet(in_shape=in_shape)

    # convert batch to device
    batch_to_device = lambda batch: {
        k: v.to(model.device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    for data in dataloaders['train']:
        data = batch_to_device(data)

        import pdb; pdb.set_trace() ## DEBUG ##
        out = model(data)
        

    cfg.run_dir = run_dir


    return cfg


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-dir')
    p.add('--in-dir')
    p.add('--checkpoint-path')

    cfg = p.parse_args()

    cfg.out_dir = '/home/ubelix/runs/rbf'
    cfg.in_dir = '/home/ubelix/data/medical-labeling/Dataset00'
    cfg.data_type = 'medical'

    main(cfg)
