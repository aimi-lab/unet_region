import munch
import logging
import unet_region.utils as utls
from tensorboardX import SummaryWriter
from torchvision import utils as tutls
import torch
import torch.optim as optim
import tqdm
from os.path import join as pjoin
from skimage import transform, draw, util, segmentation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from PIL import Image, ImageFont, ImageDraw


class Trainer:
    def __init__(self, model, dataloaders, cfg, run_dir):
        self.model = model
        self.dataloaders = dataloaders
        self.run_dir = run_dir
        self.cfg = cfg

        self.device = torch.device('cuda' if self.cfg.cuda else 'cpu')

        self.model = self.model.to(self.device)

        utls.setup_logging(run_dir)
        self.logger = logging.getLogger('coord_net')
        self.writer = SummaryWriter(run_dir)

        # convert batch to device
        self.batch_to_device = lambda batch: {
            k: v.to(self.device) if (isinstance(v, torch.Tensor)) else v
            for k, v in batch.items()
        }

        self.optimizer = optim.SGD(
            model.parameters(),
            lr=self.cfg.lr,
            nesterov=True,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay)

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, gamma=cfg.gamma, step_size=cfg.patience)

        self.logger.info('run_dir: {}'.format(self.run_dir))

    def train(self):

        best_loss = 0.
        for epoch in range(self.cfg.epochs):

            self.logger.info('Epoch {}/{}'.format(epoch + 1,
                                                  self.cfg.epochs))

            # Each epoch has a training and validation phase
            for phase in self.dataloaders.keys():
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0

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
                            loss, out = self.model(data)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()

                                self.optimizer.step()
                                self.lr_scheduler.step()

                        running_loss += loss.cpu().detach().item()

                        loss_ = running_loss / (
                            (i + 1) * self.cfg.batch_size)

                        pbar.set_description(
                            '[{}] : Loss {:.4f}'
                            .format(phase,
                                    loss_))
                        pbar.update(1)
                        samp_ += 1
                        self.writer.add_scalar('{}/lr_ls'.format(phase),
                                               self.lr_scheduler.get_lr()[-1],
                                               epoch)

                        break

                    pbar.close()
                    self.writer.add_scalar('{}/loss'.format(phase),
                                           loss_, epoch)

                # make preview images
                if phase == 'prev':
                    with torch.no_grad():

                        data = next(iter(self.dataloaders[phase]))
                        data = self.batch_to_device(data)
                        _, out = self.model(data)

                        images = np.array(data['image_unnormalized'])
                        images = images.transpose((0, -1, 1, 2))
                        images = torch.from_numpy(images).float()

                        contours = None
                        if('contour_pred' in out.keys()):
                            contours = {'target': data['interp_xy'],
                                        'pred': out['contour_pred']}

                        img = make_preview_grid(
                            images,
                            {
                                'truth': data['label/segmentation'],
                                'data': out['data'].unsqueeze(1),
                                'data_truth': data['label/edt_D'],
                                'kappa': out['kappa'].unsqueeze(1),
                                'kappa_truth': data['label/edt_kappa'],
                                'beta': out['beta'].unsqueeze(1),
                                'beta_truth': data['label/edt_beta'],
                            },
                            contours=contours,
                            font='../../sans-serif.ttf')
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


def make_preview_grid(images,
                      outputs,
                      contours=None,
                      color_center=(0, 1, 0),
                      font='sans-serif.ttf'):

    batch_size = images.shape[0]

    ims = []

    cmap = plt.get_cmap('viridis')

    outputs_maps = {
        k: np.zeros((batch_size, images.shape[-1],
                     images.shape[-2], 3))
        for k in outputs.keys()
    }
    outputs_ranges = {k: np.zeros((batch_size, 2)) for k in outputs.keys()}

    # make padded images
    pw = int(0.02 * images.shape[-1])

    my_pad = lambda x: util.pad(
        x, ((pw, pw), (pw, pw), (0, 0)), mode='constant', constant_values=1.)

    for i in range(batch_size):
        im_ = np.rollaxis(images[i].detach().cpu().numpy(), 0, start=3)

        # draw center pixel
        rr, cc = draw.circle(
            im_.shape[0] / 2 + 0.5, im_.shape[1] / 2 + 0.5, radius=1)
        im_[rr, cc, ...] = color_center

        if(contours is not None):
            color = iter(cm.rainbow(np.linspace(0, 1, len(contours.keys()))))
            for j, k in enumerate(contours.keys()):
                # plot each point
                contour_ = contours[k][i, ...].squeeze().detach().cpu().numpy()
                c = next(color)
                for l in contour_.T:
                    rr, cc = draw.circle(l[1], l[0], radius=2, shape=im_.shape[:2])
                    im_[rr, cc, ...] = c[:3]
                # plot polygon
                import pdb; pdb.set_trace() ## DEBUG ##
                rr, cc = draw.polygon_perimeter(contour_[:, 1], contour_[:, 0], shape=im_.shape[:2], clip=True)
                im_[rr, cc, ...] = c[:3]

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
    font = ImageFont.truetype(font, 25)
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
