import torch
import torch.nn as nn
import torch.nn.functional as F
from coordconv import make_addcoords


class conv2DBatchNormRelu(nn.Module):
    def __init__(self,
                 in_channels,
                 n_filters,
                 k_size,
                 stride,
                 padding,
                 bias=True,
                 dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(
                int(in_channels),
                int(n_filters),
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                bias=bias,
                dilation=dilation)

        else:
            conv_mod = nn.Conv2d(
                int(in_channels),
                int(n_filters),
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                bias=bias,
                dilation=1)

        self.cbr_unit = nn.Sequential(
            conv_mod,
            nn.BatchNorm2d(int(n_filters)),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, with_coordconv,
                 with_coordconv_r):
        super(PyramidPooling, self).__init__()

        self.with_coordconv = with_coordconv
        self.with_coordconv_r = with_coordconv_r

        self.addcoords, self.in_channels = make_addcoords(in_channels,
                                                          with_coordconv,
                                                          with_coordconv_r)

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(
                conv2DBatchNormRelu(
                    self.in_channels,
                    int(in_channels / len(pool_sizes)),
                    1,
                    1,
                    0,
                    bias=False))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        output_slices = [x]
        h, w = x.shape[2:]

        if(self.addcoords is not None):
            x = self.addcoords(x)

        for module, pool_size in zip(self.path_module_list, self.pool_sizes):
            out = F.avg_pool2d(x, int(h / pool_size))
            out = module(out)
            out = F.upsample(out, size=(h, w), mode='bilinear')
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)

