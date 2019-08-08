import torch
from torch import nn
from torch.nn import functional as F
from unet_region.baselines.rbf.backbone import Backbone


class RBFNet(torch.nn.Module):
    def __init__(self, in_shape):
        super(RBFNet, self).__init__()

        self.in_shape = in_shape

        self.backbone = Backbone()

        rates = [6, 12, 18, 24]
        self.aspp1 = Atrous_module(2048, 256, rate=rates[0])
        self.aspp2 = Atrous_module(2048, 256, rate=rates[1])
        self.aspp3 = Atrous_module(2048, 256, rate=rates[2])
        self.aspp4 = Atrous_module(2048, 256, rate=rates[3])

        # skewing branch
        self.conv_skew = nn.Conv2d(256, 64, kernel_size=3)

        # gaussian branch
        self.conv_gauss = nn.Conv2d(256, 64, kernel_size=3)
        # self.lin_gauss = nn.Linear(256*

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        import pdb; pdb.set_trace() ## DEBUG ##
        _, w, h, _ = x.shape
        
        assert(w == 224 and h == 224, "Input shape is {}, should be (224, 224)".format(
            (w, h)))
        x = self.backbone(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.sum((x1, x2, x3, x4))

        return self.sigmoid(x)


class Atrous_module(nn.Module):
    def __init__(self,
                 inplanes,
                 out_planes1,
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)

        return x
