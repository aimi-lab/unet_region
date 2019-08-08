import torch
from torch import nn
from torch.nn import functional as F
from unet_region.baselines.conn_net.backbone import Backbone


class ConnNet(torch.nn.Module):
    def __init__(self):
        super(ConnNet, self).__init__()

        self.backbone = Backbone()

        rates = [6, 12, 18, 24]
        self.aspp1 = Atrous_module(2048, 256, 3, rate=rates[0])
        self.aspp2 = Atrous_module(2048, 256, 3, rate=rates[1])
        self.aspp3 = Atrous_module(2048, 256, 3, rate=rates[2])
        self.aspp4 = Atrous_module(2048, 256, 3, rate=rates[3])

        self.deconv = nn.ConvTranspose2d(
            in_channels=3, out_channels=3, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, w, h, _ = x.shape
        x = self.backbone(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.sum((x1, x2, x3, x4))
        x = self.deconv(x)

        if self.training:
            return self.sigmoid(x)
        else:
            x = F.upsample_bilinear(x, (w, h))
            return self.sigmoid(x)


class Atrous_module(nn.Module):
    def __init__(self,
                 inplanes,
                 out_planes1,
                 out_planes2,
                 rate):
        super(Atrous_module, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            out_planes1,
            kernel_size=3,
            stride=1,
            padding=rate,
            dilation=rate)
        self.batch_norm = nn.BatchNorm2d(out_planes1)
        self.conv2 = nn.Conv2d(
            out_planes1, out_planes2, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.conv2(x)

        return x
