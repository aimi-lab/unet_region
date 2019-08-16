import torchvision.models as models
import torch
from torch import nn
import numpy as np
import math
from torch.nn import functional as F
from myresnet import MyResNet


class DarNet(torch.nn.Module):
    """
    """

    def __init__(self,
                 pool_sizes=[1, 2, 3, 6],
                 with_coordconv=True,
                 with_coordconv_r=True):

        super(DarNet, self).__init__()


        self.backbone = MyResNet(pretrained=True)

        self.data_branch = nn.Sequential(
            *[PSPModule(self.backbone.feats_conv.out_channels,
                                       1024, pool_sizes),
              nn.Dropout2d(p=0.3),
              PSPUpsample(1024, 256),
              nn.Dropout2d(p=0.15),
              PSPUpsample(256, 64),
              nn.Dropout2d(p=0.15),
              PSPUpsample(64, 64),
              nn.Dropout2d(p=0.15),
              nn.Conv2d(64, 1, kernel_size=1),
              nn.ReLU()])

        self.beta_branch = nn.Sequential(
            *[PSPModule(self.backbone.feats_conv.out_channels,
                                       1024, pool_sizes),
              nn.Dropout2d(p=0.3),
              PSPUpsample(1024, 256),
              nn.Dropout2d(p=0.15),
              PSPUpsample(256, 64),
              nn.Dropout2d(p=0.15),
              PSPUpsample(64, 64),
              nn.Dropout2d(p=0.15),
              nn.Conv2d(64, 1, kernel_size=1),
              nn.ReLU()])

        self.kappa_branch = nn.Sequential(
            *[PSPModule(self.backbone.feats_conv.out_channels,
                                       1024, pool_sizes),
              nn.Dropout2d(p=0.3),
              PSPUpsample(1024, 256),
              nn.Dropout2d(p=0.15),
              PSPUpsample(256, 64),
              nn.Dropout2d(p=0.15),
              PSPUpsample(64, 64),
              nn.Dropout2d(p=0.15),
              nn.Conv2d(64, 1, kernel_size=1),
              nn.ReLU()])


    def forward(self, x):

        features = self.backbone(x)

        data_map = self.data_branch(features)
        beta_map = self.beta_branch(features)
        kappa_map = self.kappa_branch(features)

        return data_map, beta_map, kappa_map


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(
            features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.upsample(input=stage(feats), size=(h, w), mode='bilinear')
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.PReLU())

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)
