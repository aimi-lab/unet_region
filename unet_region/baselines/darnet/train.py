from unet_region.pascal_voc_loader_patch import pascalVOCLoaderPatch
from unet_region.patch_loader import PatchLoader
from skimage import transform, segmentation
from unet_region.sub_sampler import SubsetSampler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import RandomSampler
import torch
from os.path import join as pjoin
import os
import datetime
import yaml
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
import pandas as pd
from unet_region.baselines.darnet.models.drn_contours import DRNContours
from unet_region.baselines.darnet.losses.losses import DistanceLossFast
from unet_region.baselines.darnet.trainer import Trainer
from unet_region.baselines.darnet import params
from unet_region.my_augmenters import rescale_augmenter, Normalize
from unet_region.baselines.darnet.utils import my_utils as utls


class ModelContours(torch.nn.Module):
    def __init__(self, network, restore):
        super(ModelContours, self).__init__()
        self.net = network
        print("Loading checkpoint from {}".format(restore))
        checkpoint = torch.load(restore, map_location='cpu')
        print("Loaded checkpoint")
        self.net.load_state_dict(checkpoint)
        self.distance_loss = DistanceLossFast()

    def forward(self, sample):
        loss, out = self.net(sample)
        beta_scale_norm = 0.005
        beta_scale_postnorm = 1.0
        kappa_scale = 0.1

        beta = torch.tanh(out['beta'] * beta_scale_norm) * beta_scale_postnorm
        kappa = out['kappa'] * kappa_scale

        out_dl = self.distance_loss(
            sample['init_contour_radii'], sample['interp_radii'],
            sample['init_contour_origin'], beta, out['data'], kappa,
            sample['interp_angles'], sample['delta_angles'])

        # Recover initial contour
        init_contour_origin = torch.repeat_interleave(
            sample['init_contour_origin'].unsqueeze(1), out_dl['rho'].shape[1],
            1).squeeze()
        rho_cos_theta = sample['init_contour_radii'] * torch.cos(
            sample['interp_angles'])
        rho_sin_theta = sample['init_contour_radii'] * torch.sin(
            sample['interp_angles'])
        joined = torch.stack([rho_cos_theta, rho_sin_theta], dim=-1).squeeze()
        contour = init_contour_origin + joined
        init_x = contour[..., 0]
        init_y = contour[..., 1]

        out['init_x'] = init_x
        out['init_y'] = init_y

        # merge outputs from backbone and active contour outputs
        out = {**out, **out_dl}
        return loss, out


# Get model and loss
class ModelPretrain(torch.nn.Module):
    def __init__(self, with_coordconv=False,
                 with_coordconv_r=False):
        super(ModelPretrain, self).__init__()
        self.net = DRNContours(with_coordconv=with_coordconv,
                               with_coordconv_r=with_coordconv_r)
        self.l1_loss_func = torch.nn.SmoothL1Loss()

    def forward(self, sample):
        output = self.net(sample['image'])
        assert len(output) == 3 or len(output) == 6
        beta, data, kappa = output[-3:]
        beta = beta.unsqueeze(1)
        kappa = kappa.unsqueeze(1)
        data = data.unsqueeze(1)
        loss = self.l1_loss_func(beta, sample['label/edt_beta'])
        loss += self.l1_loss_func(data, sample['label/edt_D'])
        loss += self.l1_loss_func(
            kappa, sample['label/edt_kappa'])

        if len(output) == 6:
            beta0, data0, kappa0 = output[:3]
            beta0 = beta0.unsqueeze(1)
            kappa0 = kappa0.unsqueeze(1)
            data0 = data0.unsqueeze(1)
            loss += self.l1_loss_func(
                beta0, sample['label/edt_beta'])
            loss += self.l1_loss_func(
                data0, sample['label/edt_D'])
            loss += self.l1_loss_func(
                kappa0, sample['label/edt_kappa'])
        return loss, {'kappa': kappa, 'beta': beta, 'data': data}


def main(cfg):

    d = datetime.datetime.now()

    if (cfg.phase != 'pascal'):
        ds_dir = os.path.split(cfg.in_dir)[-1]
    else:
        ds_dir = cfg.data_type

    run_dir = pjoin(cfg.out_dir, '{}_{:%Y-%m-%d_%H-%M}'.format(ds_dir, d))

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
            pjoin(cfg.in_dir, 'VOC2012'),
            patch_rel_size=cfg.patch_rel_size,
            augmentations=transf)
    elif cfg.data_type == 'medical':
        loader = PatchLoader(
            pjoin(cfg.in_dir),
            'hand',
            fake_len=cfg.fake_len,
            make_opt_box=False,
            fix_frames=cfg.frames,
            augmentation=transf)
    else:
        raise Exception('data-type must be pascal or medical')

    normalization = Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if cfg.phase == 'pascal':
        loader = pascalVOCLoaderPatch(
            cfg.in_dir,
            patch_rel_size=cfg.patch_rel_size,
            normalization=normalization,
            late_fn=lambda y: utls.process_truth_dar(y, cfg.n_nodes, cfg.
                                                     init_radius),
            augmentations=transf)
    elif (cfg.phase == 'data' or cfg.phase == 'contours'):
        loader = PatchLoader(
            cfg.in_dir,
            'hand',
            fake_len=cfg.fake_len,
            late_fn=lambda y: utls.process_truth_dar(y, cfg.n_nodes, cfg.
                                                     init_radius),
            fix_frames=cfg.frames,
            normalization=normalization,
            augmentation=transf)
    else:
        raise Exception('phase must be pascal, data, or contours')

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
        sampler=train_sampler,
        drop_last=True)

    # each batch will give same locations / augmentations
    val_loader = torch.utils.data.DataLoader(
        loader,
        num_workers=cfg.n_workers,
        batch_size=cfg.batch_size,
        collate_fn=loader.collate_fn,
        worker_init_fn=loader.worker_init_fn_dummy,
        sampler=valid_sampler,
        drop_last=True)

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

    model = ModelPretrain(cfg.coordconv, cfg.coordconv_r)
    if cfg.phase == 'contours':
        model = ModelContours(model, cfg.checkpoint_path)

    cfg.run_dir = run_dir

    # Save cfg
    with open(pjoin(run_dir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    trainer = Trainer(model, dataloaders, cfg, run_dir)
    trainer.train()

    return cfg


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-dir', required=True)
    p.add('--in-dir', required=True)
    p.add('--checkpoint-path', default=None)

    cfg = p.parse_args()

    p.add(
        '--phase',
        required=True,
        help=
        'pascal (pretrain), data (pretrain), contours. Adequate values for in-dir and checkpoint-path must be provided'
    )
    p.add('--checkpoint-path', default=None)
    cfg = p.parse_args()

    # p.add('--out-dir')
    # p.add('--in-dir')
    # p.add('--checkpoint-path',
    #       default=None)
    # p.add('--phase')
    # cfg = p.parse_args()
    # cfg.n_workers = 0
    # cfg.in_dir = '/home/ubelix/data/medical-labeling/Dataset00'
    # cfg.out_dir = '/home/ubelix/runs/scratch'
    # cfg.checkpoint_path = '/home/ubelix/runs/darnet/Dataset00_2019-08-09_13-05/checkpoints/checkpoint_ls.pth.tar'
    # cfg.phase = 'contours'

    # cfg.coordconv = True
    # cfg.coordconv_r = True

    main(cfg)
