from __future__ import print_function, division

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from unet_region.baselines.darnet.models import drn
from unet_region.baselines.darnet.models.drn import BasicBlock
from unet_region.baselines.darnet.models import coordconv

def make_addcoords(in_channels, with_coordconv, with_r):
    # simple helper to generate coordconv
    
    addcoords = None
    if(with_coordconv):
        in_channels += 2
        if(with_r):
            in_channels += 1
            addcoords = coordconv.AddCoords(with_r=True)
        else:
            addcoords = coordconv.AddCoords(with_r=False)
    return addcoords, in_channels


class ConvBlock(nn.Module):
    def __init__(self, in_planes,
                 out_planes,
                 dilation,
                 with_coordconv=False,
                 with_coordconv_r=False):
        super(ConvBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes

        self.addcoords, self.in_planes = make_addcoords(self.in_planes,
                                                       with_coordconv,
                                                       with_coordconv_r)

        self.downsample = nn.Sequential(
            nn.Conv2d(
                self.in_planes, self.out_planes,
                kernel_size=1,
                stride=1,
                bias=False),
            nn.BatchNorm2d(out_planes))
        self.convblock = BasicBlock(
            self.in_planes,
            self.out_planes,
            dilation=(dilation, dilation),
            downsample=self.downsample)

    def forward(self, x):

        if(self.addcoords is not None):
            x = self.addcoords(x)

        x = self.convblock(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, inplanes, planes,
                 with_coordconv=False,
                 with_coordconv_r=False):
        super(UpConvBlock, self).__init__()

        self.inplanes = inplanes

        self.upconv1 = nn.ConvTranspose2d(
            inplanes,
            planes,
            4,
            stride=2,
            padding=1,
            output_padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                inplanes, planes, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes))
        self.addcoords, self.inplanes = make_addcoords(self.inplanes,
                                                       with_coordconv,
                                                       with_coordconv_r)

    def forward(self, x):
        residual = x

        if(self.addcoords is not None):
            x = self.addcoords(x)

        x = self.upconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        residual = self.upsample(residual)
        x = self.relu(x + residual)
        return x


class DRNContours(nn.Module):
    def __init__(self,
                 model_name='drn_d_22',
                 classes=3,
                 pretrained=True,
                 with_coordconv=False,
                 with_coordconv_r=False):
        super(DRNContours, self).__init__()

        self.with_coordconv = with_coordconv
        self.with_coordconv_r = with_coordconv_r

        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.upconv1 = UpConvBlock(model.out_dim, model.out_dim // 2,
                                   with_coordconv, with_coordconv_r)
        self.upconv2 = UpConvBlock(model.out_dim // 2, model.out_dim // 4,
                                   with_coordconv, with_coordconv_r)
        self.upconv3 = UpConvBlock(model.out_dim // 4, model.out_dim // 8,
                                   with_coordconv, with_coordconv_r)
        self.predict_1 = nn.Conv2d(
            model.out_dim // 8,
            classes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)

        # Combine predictions with output of upconv 3 to further refine
        self.conv4 = ConvBlock(model.out_dim // 8 + classes,
                               model.out_dim // 16,
                               1,
                               with_coordconv, with_coordconv_r)
        self.conv5 = ConvBlock(model.out_dim // 16,
                               model.out_dim // 32,
                               2,
                               with_coordconv, with_coordconv_r)
        self.conv6 = ConvBlock(model.out_dim // 32,
                               model.out_dim // 32,
                               1,
                               with_coordconv, with_coordconv_r)
        self.predict_2 = nn.Conv2d(
            model.out_dim // 32,
            classes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)

        modules_to_init = [
            self.upconv1, self.upconv2, self.upconv3, self.predict_1,
            self.conv4, self.conv5, self.conv6
        ]
        for m in modules_to_init:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _create_convblock(self, in_planes, out_planes, dilation):
        downsample = nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_planes))
        convblock = BasicBlock(
            in_planes,
            out_planes,
            dilation=(dilation, dilation),
            downsample=downsample)
        return convblock

    def forward(self, x):
        x = self.base(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)

        pr1 = self.predict_1(x)
        pr1 = nn.functional.relu(pr1)

        beta1 = pr1[:, 0, :, :]
        data1 = pr1[:, 1, :, :]
        kappa1 = pr1[:, 2, :, :]

        x = self.conv4(torch.cat((x, pr1), dim=1))
        x = self.conv5(x)
        x = self.conv6(x)

        pr2 = self.predict_2(x)
        pr2 = nn.functional.relu(pr2)

        beta2 = pr2[:, 0, :, :]
        data2 = pr2[:, 1, :, :]
        kappa2 = pr2[:, 2, :, :]

        return beta1, data1, kappa1, beta2, data2, kappa2
