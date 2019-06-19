import munch
import logging
import pytorch_utils.utils as utls
import dsac_utils as dutls
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
        self.criterion = LossDSAC()

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

    def train(self):

        self.logger.info('run_dir: {}'.format(self.run_dir))

        best_iou = 0.
        for epoch in range(self.cfg.epochs):

            self.logger.info('Epoch {}/{}'.format(epoch + 1, self.cfg.epochs))

            # Each epoch has a training and validation phase
            for phase in self.dataloaders.keys():
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_iou = 0.0
                running_inter = 0.0
                running_union = 0.0

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
                            out = forward_and_infer(
                                self.model, self.cfg.in_shape,
                                self.cfg.init_radius, self.cfg.length_snake,
                                self.cfg.gamma, self.cfg.max_px_move,
                                self.cfg.n_iter, data)

                            loss_data, gradients = self.criterion(
                                out['edges'], out['alpha'], out['beta'],
                                out['kappa'], out['snakes'],
                                data['label/segmentation'],
                                data['label/nodes'])

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                out['edges'].backward(
                                    gradients['edges'], retain_graph=True)
                                out['alpha'].backward(
                                    gradients['alpha'], retain_graph=True)
                                out['beta'].backward(
                                    gradients['beta'], retain_graph=True)
                                out['kappa'].backward(gradients['kappa'])
                                self.optimizer.step()

                        running_iou += np.mean(loss_data['iou'])
                        running_inter += np.mean(loss_data['intersection'])
                        running_union += np.mean(loss_data['union'])
                        iou_ = running_iou / ((i + 1) * self.cfg.batch_size)
                        inter_ = running_inter / (
                            (i + 1) * self.cfg.batch_size)
                        union_ = running_union / (
                            (i + 1) * self.cfg.batch_size)
                        pbar.set_description(
                            '[{}] : iou {:.4f}, inter {:.4f}, union {:.4f}'.format(
                                phase, iou_, inter_, union_))
                        pbar.update(1)
                        samp_ += 1

                    pbar.close()
                    self.writer.add_scalar('{}/iou'.format(phase), iou_, epoch)
                    self.writer.add_scalar('{}/inter'.format(phase), inter_,
                                           epoch)
                    self.writer.add_scalar('{}/union'.format(phase), union_,
                                           epoch)

                # make preview images
                if phase == 'prev':
                    data = next(iter(self.dataloaders[phase]))
                    data = self.batch_to_device(data)
                    out = forward_and_infer(
                        self.model, self.cfg.in_shape, self.cfg.init_radius,
                        self.cfg.length_snake, self.cfg.gamma,
                        self.cfg.max_px_move, self.cfg.n_iter, data)

                    img = make_preview_grid(data, out)
                    self.writer.add_image('test/img',
                                          np.moveaxis(img, -1, 0),
                                          epoch)

                # save checkpoint
                if phase == 'val':
                    is_best = False
                    if (iou_ < best_iou):
                        is_best = True
                        best_iou = iou_
                    path = pjoin(self.run_dir, 'checkpoints')
                    utls.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'best_iou': best_iou,
                            'optimizer': self.optimizer.state_dict()
                        },
                        is_best,
                        path=path)


def forward_and_infer(model, in_shape, init_radius, length_snake, gamma,
                      max_px_move, n_iter, data, verbose=False):

    edges, alpha, beta, kappa = model(data['image'])

    edges_np = edges.clone().detach().cpu().numpy()
    alpha_np = alpha.clone().detach().cpu().numpy()
    beta_np = beta.clone().detach().cpu().numpy()
    kappa_np = kappa.clone().detach().cpu().numpy()

    # generate initial snake
    init_snake = dutls.make_init_snake(
        int(in_shape * init_radius), in_shape, length_snake)

    # run inference on batch
    snakes = dutls.acm_inference(edges_np, alpha_np, beta_np, kappa_np,
                                 init_snake, gamma, max_px_move, n_iter,
                                 verbose=verbose)

    # plot_preview(data['image'].detach().cpu().numpy(),
    #              edges_np, alpha_np, beta_np, kappa_np,
    #              init_snake, snakes, 0)

    return {
        'edges': edges,
        'alpha': alpha,
        'beta': beta,
        'kappa': kappa,
        'snakes': snakes
    }

def plot_preview(ims, edges, alpha, beta, kappa,
                 init_snake, snakes, n):

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))
    im_ = np.moveaxis(ims[n, ...], 0, -1)
    ax[0, 0].imshow(im_)
    ax[0, 0].plot(init_snake[:, 0], init_snake[:, 1], '--b', lw=3)
    ax[0, 0].plot(snakes[n][:, 0], snakes[n][:, 1], '--r', lw=3)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])

    ax[0, 1].imshow(edges[n, 0, ...])
    ax[0, 1].set_title('edges')
    ax[1, 0].imshow(beta[n, 0, ...])
    ax[1, 0].set_title('beta')
    ax[1, 1].imshow(kappa[n, 0, ...])
    ax[1, 1].set_title('kappa')
    ax[2, 0].imshow(alpha[n, 0, ...])
    ax[2, 0].set_title('alpha')

    fig.show()
 

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
