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
from skimage import transform, draw, util
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

class Trainer:
    def __init__(self, model, dataloaders, cfg, run_dir):
        self.model = model
        self.dataloaders = dataloaders
        self.run_dir = run_dir
        self.cfg = cfg

        self.device = torch.device('cuda' if self.cfg.cuda else 'cpu')

        # self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion_ls = torch.nn.MSELoss()

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

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.lr,
            eps=self.cfg.eps,
            weight_decay=self.cfg.weight_decay)

        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                             0.3/20)


        self.logger.info('run_dir: {}'.format(self.run_dir))

    def train_level_set(self):

        best_loss = 0.
        for epoch in range(self.cfg.epochs_ls):

            self.logger.info('Epoch {}/{}'.format(epoch + 1,
                                                  self.cfg.epochs_ls))

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

                        import pdb; pdb.set_trace() ## DEBUG ##
                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            _, _, out = self.model(data['image'])

                            loss = self.criterion_ls(out,
                                                     data['label/tsdf'])

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()
                                self.lr_scheduler.step()

                        running_loss += loss
                        loss_ = running_loss / ((i + 1) * self.cfg.batch_size)
                        pbar.set_description(
                            '[{}] : mse {:.4f}'.format(
                                phase, loss_))
                        pbar.update(1)
                        samp_ += 1
                        self.writer.add_scalar('{}/lr_ls'.format(phase),
                                            self.lr_scheduler.get_lr(),
                                            epoch)

                    pbar.close()
                    self.writer.add_scalar('{}/mse_ls'.format(phase),
                                           loss_,
                                           epoch)

                # make preview images
                if phase == 'prev':
                    data = next(iter(self.dataloaders[phase]))
                    data = self.batch_to_device(data)
                    _, _, out = self.model(data['image'])

                    img = make_preview_grid(data, out)
                    self.writer.add_image('test/img',
                                          np.moveaxis(img, -1, 0),
                                          epoch)

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
                            'state_dict': self.model.state_dict(),
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

def make_preview_grid(data, out, color_pred=(0, 0, 1),
                      color_truth=(1, 0, 0),
                      color_center=(0, 1, 0)):

    batch_size = data['image'].shape[0]

    ims = []

    betas = []
    kappas = []
    alphas = []
    edges = []

    betas_maps = []
    kappas_maps = []
    edges_maps = []


    cmap = plt.get_cmap('viridis')

    for i in range(batch_size):
        im_ = data['image'][i].detach().cpu().numpy()
        im_ = np.rollaxis(im_, 0, 3)
        snake_ = out['snakes'][i]

        # draw predicted snake
        rr, cc = draw.polygon_perimeter(
            snake_[:, 1], snake_[:, 0],
            shape=im_.shape[:2])
        im_[rr, cc, ...] = color_pred

        # draw target snake
        snake_truth = data['label/nodes'][i]
        rr, cc = draw.polygon_perimeter(
            snake_truth[:, 1], snake_truth[:, 0],
            shape=im_.shape[:2])
        im_[rr, cc, ...] = color_truth

        # draw center pixel
        rr, cc = draw.circle(
            im_.shape[0] / 2 + 0.5,
            im_.shape[1] / 2 + 0.5,
            radius=1)
        im_[rr, cc, ...] = color_center

        ims.append(im_)

        edges_ = out['edges'][i, 0, ...].detach().cpu().numpy()
        edges_maps.append(cmap(normalize_map(edges_.copy()))[..., 0:3])

        beta_ = out['beta'][i, 0, ...].detach().cpu().numpy()
        betas_maps.append(cmap(normalize_map(beta_.copy()))[..., 0:3])

        kappa_ = out['kappa'][i, 0, ...].detach().cpu().numpy()
        kappas_maps.append(cmap(normalize_map(kappa_.copy()))[..., 0:3])

        alpha_ = out['alpha'][i, 0, ...].detach().cpu().numpy().mean()
        alphas.append(alpha_)

        betas.append(beta_)
        kappas.append(kappa_)
        edges.append(edges_)

    # make padded images
    pw = int(0.02*ims[0].shape[0])

    ims = [
        util.pad(
            ims[i], ((pw, pw), (pw, pw), (0, 0)),
            mode='constant',
            constant_values=1.) for i in range(len(ims))
    ]
    edges_maps = [
        util.pad(
            edges_maps[i], ((pw, pw), (pw, pw), (0, 0)),
            mode='constant',
            constant_values=1.) for i in range(len(betas))
    ]
    betas_maps = [
        util.pad(
            betas_maps[i], ((pw, pw), (pw, pw), (0, 0)),
            mode='constant',
            constant_values=1.) for i in range(len(betas))
    ]
    kappas_maps = [
        util.pad(
            kappas_maps[i], ((pw, pw), (pw, pw), (0, 0)),
            mode='constant',
            constant_values=1.) for i in range(len(kappas))
    ]

    all_ = np.concatenate([
        np.concatenate([ims[i],
                        edges_maps[i],
                        betas_maps[i],
                        kappas_maps[i]], axis=1)
        for i in range(len(ims))
    ],
                          axis=0)

    header = Image.fromarray((255*np.ones((ims[0].shape[0],
                                           all_.shape[1],
                                           3))).astype(np.uint8))
    drawer = ImageDraw.Draw(header)
    font = ImageFont.truetype("sans-serif.ttf", 25)
    text_ = 'image / data / beta / kappa\nalpha: {}\ne_min: {}\ne_max: {}\nb_min: {}\nb_max: {}\nk_min: {}\nk_max: {}'.format(
        alphas,
        [np.min(e) for e in edges],
        [np.max(e) for e in edges],
        [np.min(b) for b in betas],
        [np.max(b) for b in betas],
        [np.min(k) for k in kappas],
        [np.max(k) for k in kappas]
    )
    drawer.text((0, 0), text_, (0, 0, 0), font=font)
    header = np.array(header)

    all_ = np.concatenate((header,
                           (255*all_).astype(np.uint8)), axis=0)
    
    return all_
