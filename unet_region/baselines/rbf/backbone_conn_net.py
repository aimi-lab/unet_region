# Detection part of BlitzNet ("BlitzNet: A Real-Time Deep Network for Scene Understanding" Nikita Dvornik et al. ICCV17)

import torch
import torch.nn as nn
import torch.nn.functional as F 

from torchvision.models import resnet50, Bottleneck
from collections import OrderedDict
from unet_region.baselines.conn_net.non_local_embedded_gaussian import NONLocalBlock2D


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
        self.Up = nn.Sequential(OrderedDict([
            ('rev_layer7', BottleneckSkip(2048, 2048, 512)),
            ('rev_layer6', BottleneckSkip(2048 if self.pred_layers[0] == 'rev_layer6' else 512, 2048, 512)),
            ('rev_layer5', BottleneckSkip(512,  2048, 512)),
            ('rev_layer4', BottleneckSkip(512,  2048, 512)),
            ('rev_layer3', BottleneckSkip(512,  1024, 512)),
            ('rev_layer2', BottleneckSkip(512,  512,  512)),
            ('rev_layer1', BottleneckSkip(512,  256,  512))]))

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


class BottleneckSkip(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, stride=1, 
                    mode='bilinear'):
        super().__init__()
        self.mode = mode

        if stride != 1 or in_channels1 != out_channels:
            self.shortcut = shortcut(in_channels1, out_channels, stride)
        else:
            self.shortcut = nn.Sequential()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels1 + in_channels2, out_channels // 4, 1, bias=False)
            nn.BatchNorm2d(out_channels // 4)
            nn.ReLU(inplace=True)
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1, stride=stride, bias=False)
            nn.BatchNorm2d(out_channels // 4)
            nn.ReLU(inplace=True)
            nn.Conv2d(out_channels // 4, out_channels, 1, bias=False)
            nn.BatchNorm2d(out_channels))

    def forward(self, x, skip):
        x = F.upsample(x, size=skip.size()[2:], mode=self.mode)
        shortcut = self.shortcut(x)
        residual = self.residual(torch.cat([x, skip], dim=1))

        return F.relu(shortcut + residual, inplace=True)



def shortcut(in_channels=2048, out_channels=2048, stride=2):
    #if in_channels == out_channels:    # in BlitzNet's tensorflow implementation
    #    return nn.MaxPool2d(1, stride=stride)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels))
