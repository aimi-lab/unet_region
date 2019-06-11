import torchvision.models as models
import torch
import numpy as np
import math
from torch.nn import functional as F


class DSAC(torch.nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_size=256,
                 num_filts=[32, 64, 128, 128, 256],
                 stack_from=0,
                 wd=0.001,
                 E_blur=2,
                 checkpoint_path=None,
                 cuda=True):

        super(DSAC, self).__init__()

        self.device = torch.device('cuda' if cuda else 'cpu')

        self.stack_from = stack_from

        # hypercolumns
        in_channels = 3
        self.resized_out = []
        self.mlp = []
        self.modules_feats = [
            DSACFeats(
                in_channels, num_filts[0], out_size=out_size, kernel_size=7)
        ]
        for i in range(1, len(num_filts)):
            if (i == 1):
                self.modules_feats.append(
                    DSACFeats(
                        num_filts[i - 1],
                        num_filts[i],
                        out_size=out_size,
                        kernel_size=5))

            else:
                self.modules_feats.append(
                    DSACFeats(
                        num_filts[i - 1],
                        num_filts[i],
                        out_size=out_size,
                        kernel_size=3))

            if (i >= stack_from):
                self.resized_out.append(
                    torch.nn.UpsamplingBilinear2d(out_size))

        self.modules_feats = torch.nn.Sequential(*self.modules_feats)

        # MLP for dimension reduction
        self.module_reduc = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=np.sum(num_filts),
                            out_channels=256,
                            kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=64,
                            kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64)
        ])

        # Predict energy
        self.module_energy = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=64, out_channels=1,
                            kernel_size=1),
            GaussianSmoothing(1, 9, E_blur),
            torch.nn.UpsamplingBilinear2d(out_size)])

        # Predict alpha
        self.module_alpha = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=64, out_channels=1,
                            kernel_size=1),
            torch.nn.AvgPool2d(out_size),
            torch.nn.UpsamplingNearest2d(out_size)])

        # Predict beta
        self.module_beta = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=64, out_channels=1,
                            kernel_size=1),
            torch.nn.UpsamplingBilinear2d(out_size)])

        # Predict kappa
        self.module_kappa = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=64, out_channels=1,
                            kernel_size=1),
            torch.nn.UpsamplingBilinear2d(out_size)])

        self.to(self.device)

    def forward(self, x):

        # get features and stack them
        stack = []
        for i, m in enumerate(self.modules_feats):
            x, out = m(x)
            if(i >= self.stack_from):
                stack.append(out)

        # concat and pass in dimension reduction
        stack = torch.cat(stack, dim=1)
        for_predict = self.module_reduc(stack)

        data = self.module_energy(for_predict)
        alpha = self.module_alpha(for_predict)
        beta = self.module_beta(for_predict)
        kappa = self.module_kappa(for_predict)

        return data, alpha, beta, kappa


class DSACFeats(torch.nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=32,
                 out_size=256,
                 kernel_size=3):

        super(DSACFeats, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.relu(feat)
        out = self.maxpool(feat)
        out = self.bn(out)

        return feat, out


class GaussianSmoothing(torch.nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, int):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.
                format(dim))

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
