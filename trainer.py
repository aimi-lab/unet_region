import munch
import logging
import pytorch_utils.utils as utls
from tensorboardX import SummaryWriter
from torchvision import utils as tutls
import torch
import torch.optim as optim
import tqdm
from os.path import join as pjoin
from region_loss import BCERegionLoss


class Trainer:
    def __init__(self, model, dataloaders, params, run_dir):
        self.model = model
        self.dataloaders = dataloaders
        self.params = munch.Munch(params)
        self.run_dir = run_dir

        self.device = torch.device('cuda' if self.params.cuda else 'cpu')

        # self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion = BCERegionLoss(self.params.loss_size,
                                       self.params.in_shape)

        utls.setup_logging(run_dir)
        self.logger = logging.getLogger('unet_region')
        self.writer = SummaryWriter(run_dir)

        # convert batch to device
        self.batch_to_device = lambda batch: {
            k: v.to(self.device) if (isinstance(v, torch.Tensor)) else v
            for k, v in batch.items()
        }

        self.optimizer = optim.RMSprop(
            model.parameters(),
            momentum=self.params.momentum,
            lr=self.params.lr,
            alpha=self.params.alpha,
            eps=self.params.eps,
            weight_decay=self.params.weight_decay)

    def train(self):

        self.logger.info('run_dir: {}'.format(self.run_dir))

        best_loss = float('inf')
        for epoch in range(self.params.epochs):

            self.logger.info('Epoch {}/{}'.format(epoch + 1,
                                                  self.params.epochs))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0

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
                        loss = self.criterion(out, data['label/segmentation'])

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.cpu().detach().numpy()
                    loss_ = running_loss / ((i + 1) * self.params.batch_size)
                    pbar.set_description('[{}] loss: {:.4f}'.format(
                        phase, loss_))
                    pbar.update(1)
                    samp_ += 1

                pbar.close()
                self.writer.add_scalar('{}/loss'.format(phase), loss_, epoch)

                # make test images
                if phase == 'val':
                    data_preview = next(iter(self.dataloaders[phase]))
                    pred_ = torch.sigmoid(self.model(data['image'])).cpu()
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
                                        nrow=self.params.batch_size,
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
