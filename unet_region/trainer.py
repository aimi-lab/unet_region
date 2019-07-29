import munch
import logging
import pytorch_utils.utils as utls
from tensorboardX import SummaryWriter
from torchvision import utils as tutls
import torch
import torch.optim as optim
import tqdm
from os.path import join as pjoin
from loss_dsac import LossDSAC
from skimage import transform, draw, util, segmentation
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from loss_direct import LossDirect
from loss_t import LossT
from acm_utils import acm_ls, make_init_ls_gaussian


class Trainer:
    def __init__(self, model, dataloaders, cfg, run_dir):
        self.model = model
        self.dataloaders = dataloaders
        self.run_dir = run_dir
        self.cfg = cfg

        self.device = torch.device('cuda' if self.cfg.cuda else 'cpu')

        self.criterion_data = torch.nn.SmoothL1Loss()
        self.criterion_beta = torch.nn.SmoothL1Loss()
        self.criterion_kappa = torch.nn.SmoothL1Loss()

        utls.setup_logging(run_dir)
        self.logger = logging.getLogger('coord_net')
        self.writer = SummaryWriter(run_dir)

        # convert batch to device
        self.batch_to_device = lambda batch: {
            k: v.to(self.device) if (isinstance(v, torch.Tensor)) else v
            for k, v in batch.items()
        }

        if (cfg.checkpoint_path is not None):
            self.logger.info('Loading checkpoint: {}'.format(
                cfg.checkpoint_path))
            checkpoint = torch.load(cfg.checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'])

        # self.optimizer = optim.Adam(
        #     model.parameters(),
        #     lr=self.cfg.lr,
        #     weight_decay=self.cfg.weight_decay)
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=self.cfg.lr,
            nesterov=True,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay)

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 50, gamma=0.5, last_epoch=-1)

        self.logger.info('run_dir: {}'.format(self.run_dir))

        self.acm_fun = lambda phi, v, m: acm_ls(
            phi, v, m, 1, cfg.n_iters, vec_field=True)

    def pretrain(self):

        best_loss = 0.
        for epoch in range(self.cfg.epochs_pretrain):

            self.logger.info('Epoch {}/{}'.format(epoch + 1,
                                                  self.cfg.epochs_pretrain))

            # Each epoch has a training and validation phase
            for phase in self.dataloaders.keys():
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss_data = 0.0
                running_loss_beta = 0.0
                running_loss_kappa = 0.0

                if (phase in ['train', 'val']):
                    # Iterate over data.
                    pbar = tqdm.tqdm(total=len(self.dataloaders[phase]))
                    samp_ = 0
                    for i, data in enumerate(self.dataloaders[phase]):
                        data = self.batch_to_device(data)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            out_data, out_beta, out_kappa = self.model(
                                data['image'])

                            loss_data = self.criterion_data(
                                out_data, data['label/edt_D'])
                            loss_beta = self.criterion_beta(
                                out_beta, data['label/edt_beta'])
                            loss_kappa = self.criterion_kappa(
                                out_kappa, data['label/edt_kappa'])

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss_data.backward(retain_graph=True)
                                loss_beta.backward(retain_graph=True)
                                loss_kappa.backward()

                                self.optimizer.step()
                                self.lr_scheduler.step()

                        running_loss_data += loss_data.cpu().detach().item()
                        running_loss_beta += loss_beta.cpu().detach().item()
                        running_loss_kappa += loss_kappa.cpu().detach().item()
                        loss_data_ = running_loss_data / (
                            (i + 1) * self.cfg.batch_size)
                        loss_beta_ = running_loss_beta / (
                            (i + 1) * self.cfg.batch_size)
                        loss_kappa_ = running_loss_kappa / (
                            (i + 1) * self.cfg.batch_size)
                        loss_ = loss_data_ + loss_beta_ + loss_kappa_

                        pbar.set_description(
                            '[{}] : L_data {:.4f}, L_beta {:.4f}, L_kappa {:.4f} L_tot {:.4f}'
                            .format(phase,
                                    loss_data_,
                                    loss_beta_,
                                    loss_kappa_,
                                    loss_))
                        pbar.update(1)
                        samp_ += 1
                        self.writer.add_scalar('{}/lr_ls'.format(phase),
                                               self.lr_scheduler.get_lr()[-1],
                                               epoch)


                    pbar.close()
                    self.writer.add_scalar('{}/loss_data'.format(phase),
                                           loss_data_, epoch)
                    self.writer.add_scalar('{}/loss_beta'.format(phase),
                                           loss_beta_, epoch)
                    self.writer.add_scalar('{}/loss_kappa'.format(phase),
                                           loss_kappa_,
                                           epoch)
                    self.writer.add_scalar('{}/loss_tot'.format(phase), loss_,
                                           epoch)

                # make preview images
                if phase == 'prev':
                    with torch.no_grad():

                        data = next(iter(self.dataloaders[phase]))
                        data = self.batch_to_device(data)
                        out_data, out_beta, out_kappa = self.model(data['image'])

                        img = make_preview_grid(
                            data, {
                                'truth': data['label/segmentation'],
                                'data': out_data,
                                'data_truth': data['label/edt_D'],
                                'kappa': out_kappa,
                                'kappa_truth': data['label/edt_kappa'],
                                'beta': out_beta,
                                'beta_truth': data['label/edt_beta'],
                            })
                        self.writer.add_image('test/img',
                                              np.moveaxis(img, -1, 0), epoch)

                # save checkpoint
                if phase == 'val':
                    is_best = False
                    if (loss_ < best_loss):
                        is_best = True
                        best_loss = loss_
                    path = pjoin(self.run_dir, 'checkpoints')
                    utls.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'model': self.model,
                            'best_iou': best_loss,
                            'optimizer': self.optimizer.state_dict()
                        },
                        is_best,
                        path=path,
                        fname_cp='checkpoint_ls.pth.tar',
                        fname_bm='best_model_ls.pth.tar')


def normalize_map(a):
    a -= a.min()
    a /= a.max()
    return a


def make_preview_grid(data,
                      outputs,
                      color_truth=(1, 0, 0),
                      color_center=(0, 1, 0)):

    batch_size = data['image'].shape[0]

    ims = []

    cmap = plt.get_cmap('viridis')

    outputs_maps = {
        k: np.zeros((batch_size, data['image'].shape[-1],
                     data['image'].shape[-2], 3))
        for k in outputs.keys()
    }
    outputs_ranges = {k: np.zeros((batch_size, 2)) for k in outputs.keys()}

    # make padded images
    pw = int(0.02 * data['image'].shape[-1])

    my_pad = lambda x: util.pad(
        x, ((pw, pw), (pw, pw), (0, 0)), mode='constant', constant_values=1.)

    for i in range(batch_size):
        im_ = data['image_unnormalized'][i]

        # draw center pixel
        rr, cc = draw.circle(
            im_.shape[0] / 2 + 0.5, im_.shape[1] / 2 + 0.5, radius=1)
        im_[rr, cc, ...] = color_center

        ims.append(im_)

        for k in outputs.keys():
            out_ = outputs[k][i, 0, ...].detach().cpu().numpy()
            outputs_ranges[k][i] = (out_.min(), out_.max())
            out_ = cmap(normalize_map(out_.copy()))[..., 0:3]
            outputs_maps[k][i, ...] = out_

    all_ = np.concatenate([
        np.concatenate([my_pad(ims[i])] + \
                        [my_pad(outputs_maps[k][i, ...])
                         for k in outputs_maps.keys()], axis=1)
        for i in range(len(ims))
    ],
                          axis=0)

    header = Image.fromarray((255 * np.ones(
        (ims[0].shape[0], all_.shape[1], 3))).astype(np.uint8))
    drawer = ImageDraw.Draw(header)
    font = ImageFont.truetype("sans-serif.ttf", 25)
    text_header = ''
    for k in outputs_maps.keys():
        text_header += '{} / '.format(k)

    text_ranges = ''
    for i in range(outputs_ranges[k].shape[0]):
        for k, v in outputs_ranges.items():
            text_ranges += '[{:.2f}, {:.2f}] / '.format(
                outputs_ranges[k][i, 0], outputs_ranges[k][i, 1])
        text_ranges += '\n'

    text_ = text_header + '\n' + text_ranges
    drawer.text((0, 0), text_, (0, 0, 0), font=font)
    header = np.array(header)

    all_ = np.concatenate((header, (255 * all_).astype(np.uint8)), axis=0)

    return all_
