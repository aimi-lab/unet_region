from unet_region.pascal_voc_loader_patch import pascalVOCLoaderPatch
from unet_region.patch_loader import PatchLoader
from unet_region.learning_loader import LearningLoader
from unet_region.my_augmenters import rescale_augmenter, Normalize
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
import params
from coord_net import CoordNet
from trainer import Trainer

def main(cfg):

    d = datetime.datetime.now()

    if(cfg.data_type == 'medical'):
        ds_dir = os.path.split(cfg.in_dir)[-1]
    else:
        ds_dir = cfg.data_type

    run_dir = pjoin(cfg.out_dir, 'runs', '{}_{:%Y-%m-%d_%H-%M}'.format(
        ds_dir, d))

    in_shape = [cfg.in_shape] * 2

    transf = iaa.Sequential([
        # iaa.Invert(0.5) if 'Dataset1' in cfg.in_dir else iaa.Noop(),
        iaa.Noop(),
        iaa.SomeOf(3, [
            iaa.Affine(rotate=iap.Uniform(-15., 15.)),
            iaa.Affine(shear=iap.Uniform(-15., -15.)),
            iaa.Fliplr(1.),
            iaa.Flipud(1.),
            iaa.GaussianBlur(sigma=iap.Uniform(0.0, 0.05))
        ]),
        iaa.Resize(in_shape), rescale_augmenter
    ])

    normalization = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

    if cfg.data_type == 'pascal':
        loader = pascalVOCLoaderPatch(
            cfg.in_dir, patch_rel_size=cfg.patch_rel_size)
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

    loader = LearningLoader(loader,
                            augmentations=transf,
                            normalization=normalization)

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

    dataloaders = {'train': train_loader,
                   'val': val_loader,
                   'prev': prev_loader}

    model = CoordNet(cuda=cfg.cuda)

    cfg.run_dir = run_dir

    # Save cfg
    with open(pjoin(run_dir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    trainer = Trainer(model, dataloaders, cfg, run_dir)
    trainer.train()

    return cfg

if __name__ == "__main__":

    p = params.get_params()

    cfg = p.parse_args()
    cfg.data_type = 'pascal'
    cfg.out_dir = '/home/ubelix/medical-labeling/unet_region'
    cfg.in_dir = '/home/laurent.lejeune/medical-labeling'
    cfg.checkpoint_path = None
    cfg.n_workers = 0

    main(cfg)
