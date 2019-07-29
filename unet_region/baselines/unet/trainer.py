import munch
import logging
from unet_region import utils as utls
from tensorboardX import SummaryWriter
from torchvision import utils as tutls
import torch
import torch.optim as optim
import tqdm
from os.path import join as pjoin
from region_loss import MSERegionLoss
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
        self.criterion = MSERegionLoss(
            self.cfg.loss_size,
            self.cfg.in_shape,
            device=self.device,
            lambda_=self.cfg.loss_lambda)

        utls.setup_logging(run_dir)
        self.logger = logging.getLogger('unet_region')
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

        self.optimizer = optim.SGD(
            model.parameters(),
            momentum=self.cfg.momentum,
            lr=self.cfg.lr,
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
                            out = self.model(data['image'])
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
                    im_ = data['image'].cpu()
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
                            'model': self.model,
                            'best_loss': best_loss,
                            'optimizer': self.optimizer.state_dict()
                        },
                        is_best,
                        path=path)


def transform_truth(truth, shape):

    truth = transform.resize(truth, shape, anti_aliasing=True, mode='reflect')

    truth = np.array(truth)
    truth = torch.from_numpy(truth[np.newaxis, ...])
    truth = truth.type(torch.float32)

    return truth


def transform_img(img, shape, normalize=None):

    img = transform.resize(img, shape, anti_aliasing=True, mode='reflect')

    img = [img[..., c] for c in range(img.shape[-1])]

    if (normalize is not None):
        for c in range(3):
            img[c] = img[c] - normalize['mean'][c]

        for c in range(3):
            img[c] = img[c] / normalize['mean'][c]

    img = np.array(img)
    img = torch.from_numpy(img[np.newaxis, ...])
    img = img.type(torch.float32)

    return img
