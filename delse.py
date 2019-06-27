import torchvision.models as models
import torch
import numpy as np
import math
from torch.nn import functional as F
from psp import PyramidPooling


class DELSE(torch.nn.Module):
    """
    TODO: change architecture with skip connections etc..
    """

    def __init__(self,
                 pool_sizes=[1, 2, 4, 8],
                 cuda=True,
                 with_coordconv=True,
                 with_coordconv_r=True):

        super(DELSE, self).__init__()

        self.device = torch.device('cuda' if cuda else 'cpu')

        self.backbone = models.resnet.resnet101(pretrained=True)
        self.backbone = torch.nn.Sequential(
            *list(self.backbone.children())[:-2])

        self.motion_branch = PyramidPooling(
            self.backbone[-1][-1].conv3.out_channels,
            pool_sizes,
            with_coordconv,
            with_coordconv_r)
        self.modulation_branch = PyramidPooling(
            self.backbone[-1][-1].conv3.out_channels,
            pool_sizes,
            with_coordconv,
            with_coordconv_r)
        self.level_set_branch = PyramidPooling(
            self.backbone[-1][-1].conv3.out_channels,
            pool_sizes,
            with_coordconv,
            with_coordconv_r)

        self.to(self.device)

    def forward(self, x):

        features = self.backbone(x)

        motion_map = self.motion_branch(features)
        modulation_map = self.modulation_branch(features)
        level_set_map = self.level_set_branch(features)

        return motion_map, modulation_map, level_set_map
