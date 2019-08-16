import torch
from skimage.draw import circle
import numpy as np
import torch.nn.functional as F
from torch.nn import NLLLoss, LogSoftmax, CrossEntropyLoss
import matplotlib.pyplot as plt


class BCERegionLoss(torch.nn.Module):
    """
    Computes BCE on a circular region defined by radius_rel
    """

    def __init__(self,
                 region_rel_size,
                 size,
                 device=torch.device('cpu'),
                 lambda_=1,
                 pos_thr=0.8):
        """
        lambda_ penalizes on background
        """
        super(BCERegionLoss, self).__init__()

        self.size = int(region_rel_size * size)
        self.size += 0 if self.size % 2 else 1
        self.lambda_ = lambda_
        self.pos_thr = pos_thr

        self.weights = torch.tensor([self.lambda_, 1.]).to(device)
        # self.soft_max = LogSoftmax(dim=1)
        # self.loss_fn = NLLLoss(reduction='none')
        self.loss_fn = CrossEntropyLoss(weight=self.weights)

        self.eps = 10e-10

    def forward(self, input, target):

        size_in = input.shape[-1]
        input = input[..., size_in // 2 - self.size // 2:size_in // 2 +
                      self.size // 2 + 1, size_in // 2 -
                      self.size // 2:size_in // 2 + self.size // 2 + 1]
        target = target[..., size_in // 2 - self.size // 2:size_in // 2 +
                        self.size // 2 + 1, size_in // 2 -
                        self.size // 2:size_in // 2 + self.size // 2 + 1]

        input = torch.cat((1 - input, input), dim=1)
        target = target.type(torch.long)
        target = target.squeeze(dim=1)
        loss = self.loss_fn(input, target)


        return loss
