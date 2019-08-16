# Detection part of BlitzNet ("BlitzNet: A Real-Time Deep Network for Scene Understanding" Nikita Dvornik et al. ICCV17)

import torch
import torch.nn as nn
import torch.nn.functional as F 

from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck
from collections import OrderedDict
from unet_region.baselines.rbf.non_local_embedded_gaussian import NONLocalBlock2D


class Backbone(object):

    def __init__(self):
        super(Backbone, self).__init__()
        self.Base = resnet50(pretrained=True)
        self.Down = nn.Sequential(OrderedDict([
            ('layer5', nn.Sequential(Bottleneck(2014, 512, 2, shortcut()), 
                                     Bottleneck(2048, 512, 1))),
            ('layer6', nn.Sequential(Bottleneck(2014, 512, 2, shortcut()), 
                                     Bottleneck(2048, 512, 1))),
            ('layer7', nn.Sequential(Bottleneck(2014, 512, 2, shortcut()), 
                                     Bottleneck(2048, 512, 1))),
            ('layer8', nn.Sequential(Bottleneck(2014, 512, 2, shortcut()), 
                                     Bottleneck(2048, 512, 1)))]))

        self.non_local = NONLocalBlock2D()
        self.skip_layers = ['layer1', 'layer2', 'layer3', 'layer4']


    def forward(self, x):
        skips, xs = [], []
        for name, m in self.Base._modules.items():
            x = m(x)
            if name in self.skip_layers:
                skips.append(x)
            if name == 'layer3':
                x = self.non_local(x)
            if name == 'avgpool':
                break

        for name, m in self.Down._modules.items():
            if name in self.skip_layers:
                x = m(x)
                skips.append(x)

        ind = -2 
        for name, m in self.Up._modules.items():
            if name in self.pred_layers:
                x = m(x, skips[ind])
                xs.append(x)
                ind -= 1

        xs = torch.stack(xs)
        return xs

def shortcut(in_channels=2048, out_channels=2048, stride=2):
    #if in_channels == out_channels:    # in BlitzNet's tensorflow implementation
    #    return nn.MaxPool2d(1, stride=stride)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels))
