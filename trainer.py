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
from loss_splines import LossSplines
from skimage import transform
import numpy as np


class Trainer:
    def __init__(self, model, dataloaders, cfg, run_dir):
        self.model = model
        self.dataloaders = dataloaders
        self.run_dir = run_dir
        self.cfg = cfg

        self.device = torch.device('cuda' if self.cfg.cuda else 'cpu')

        # self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion = LossSplines()

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

        best_loss = float('inf')
        for epoch in range(self.cfg.epochs):

            self.logger.info('Epoch {}/{}'.format(epoch + 1, self.cfg.epochs))

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
                            res = self.model(data['image'])
                            data, alpha, beta, kappa = res

                            # generate initial snake
                            init_snake = dutls.make_init_snake(
                                int(self.cfg.in_shape*self.cfg.init_radius),
                                self.cfg.in_shape,
                                self.cfg.length_snake)
                            init_snake = torch.from_numpy(init_snake)
                            init_snake = init_snake.type(torch.float)

                            # run inference on batch
                            snakes = dutls.acm_inference(data, alpha,
                                                         beta,
                                                         kappa,
                                                         init_snake,
                                                         self.cfg.gamma,
                                                         self.cfg.max_px_move,
                                                         self.cfg.delta_s)

                            loss = self.criterion(out,
                                                  data['label/segmentation'])

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()
                        running_loss += loss.cpu().detach().numpy()
                        loss_ = running_loss / ((i + 1) * self.cfg.batch_size)
                        pbar.set_description('[{}] loss: {:.4f}'.format(
                            phase, loss_))
                        pbar.update(1)
                        samp_ += 1

                    pbar.close()
                    self.writer.add_scalar('{}/loss'.format(phase),
                                           loss_,
                                           epoch)

                # make preview images
                if phase == 'prev':
                    data = next(iter(self.dataloaders[phase]))
                    data = self.batch_to_device(data)
                    pred_ = self.model(data['image']).cpu()
                    pred_ = [
                        pred_[i, ...].repeat(3, 1, 1)
                        for i in range(pred_.shape[0])
                    ]
                    im_ = data['image_unnormalized'].cpu()
                    im_ = [im_[i, ...] for i in range(im_.shape[0])]
                    truth_ = data['label/segmentation'].cpu()
                    truth_ = [
                        truth_[i, ...].repeat(3, 1, 1)
                        for i in range(truth_.shape[0])
                    ]
                    all_ = [
                        tutls.make_grid([im_[i], truth_[i], pred_[i]],
                                        nrow=len(pred_),
                                        padding=10,
                                        pad_value=1.)
                        for i in range(len(truth_))
                    ]
                    all_ = torch.cat(all_, dim=1)
                    self.writer.add_image('test/img', all_, epoch)

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
                            'best_loss': best_loss,
                            'optimizer': self.optimizer.state_dict()
                        },
                        is_best,
                        path=path)

