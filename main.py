from pytorch_utils.pascal_voc_loader_patch import pascalVOCLoaderPatch
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
import numpy as np
import pandas as pd
from trainer import Trainer

root = '/home/laurent.lejeune/medical-labeling'
d = datetime.datetime.now()
run_dir = pjoin(root, 'unet_region', 'runs', '{:%Y-%m-%d_%H-%M-%S}'.format(d))

params = {
    'epochs': 80,
    'lr': 1e-4,
    'momentum': 0.9,
    'alpha': 0.99,
    'eps': 1e-08,
    'ds_split': 0.8,
    'ds_shuffle': True,
    'weight_decay': 0,
    'batch_size': 8,
    'batch_norm': True,
    'num_workers': 4,
    'cuda': False,
    'run_dir': run_dir,
    'with_coordconv': True,
    'with_coordconv_r': True,
    'run_dir': run_dir,
    'in_shape': 256,
    'patch_rel_size': 0.3,
    'loss_size': 0.4,
    'aug_noise': 0.1,
    'aug_flip_proba': 0.5,
    'aug_some': 2
}

in_shape = [params['in_shape']] * 2

transf = iaa.Sequential([
    iaa.SomeOf(
        2, [
            iaa.Fliplr(params['aug_flip_proba']),
            iaa.AdditiveGaussianNoise(scale=params['aug_noise'] * 255)
        ],
        random_order=True)
])

loader = pascalVOCLoaderPatch(
    root,
    patch_rel_size=params['patch_rel_size'],
    augmentations=transf,
    cuda=params['cuda'],
    do_reshape=True,
    img_size=params['in_shape'])

# Creating data indices for training and validation splits:
dataset_size = len(loader)
validation_split = 1 - params['ds_split']
random_seed = 42
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if params['ds_shuffle']:
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
    batch_size=params['batch_size'],
    num_workers=params['num_workers'],
    collate_fn=collate_fn_pascal_patch,
    sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    loader,
    num_workers=params['num_workers'],
    batch_size=params['batch_size'],
    collate_fn=collate_fn_pascal_patch,
    sampler=valid_sampler)

dataloaders = {'train': train_loader, 'val': val_loader}

model = UNet(
    in_channels=3,
    out_channels=1,
    depth=4,
    cuda=params['cuda'],
    with_coordconv=params['with_coordconv'],
    with_coordconv_r=params['with_coordconv_r'],
    with_batchnorm=params['batch_norm'])

# Save cfg
with open(pjoin(run_dir, 'cfg.yml'), 'w') as outfile:
    yaml.dump(params, stream=outfile, default_flow_style=False)

trainer = Trainer(model, dataloaders, params, run_dir)
trainer.train()
