import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet101
from torchvision.models import resnet50


class MyResNet(nn.Module):
    """
    ResNet101 for feature extraction
    The stride is reduced, and the last two blocks 
    have dilated convolutions to preserve resolution of output.
    Skip layers are also added:
    Outputs of each block are concatenated (after adequate resampling),
and concatenated, etc
Mode is for classification/features mode
    """

    def __init__(self,
                 in_shape=224,
                 pretrained=True):
        super(MyResNet, self).__init__()

        self.backbone = resnet50(pretrained)

        # skip-layers: find
        self.skip_layers_shape = conv_output_shape(
            (in_shape, in_shape),
            self.backbone.layer1[-1].conv3.kernel_size,
            self.backbone.layer1[-1].conv3.stride,
            self.backbone.layer1[-1].conv3.padding,
            self.backbone.layer1[-1].conv3.dilation)

        self.skip_layers_n_feats = [m[-1].bn3.num_features
                                    for m in [self.backbone.layer1,
                                              self.backbone.layer2,
                                              self.backbone.layer3,
                                              self.backbone.layer4]]

        self.skip_layer_conv1 = nn.Conv2d(self.skip_layers_n_feats[0],
                                          128,
                                          kernel_size=3,
                                          stride=1, padding=1)

        self.skip_layer_conv2 = nn.Conv2d(self.skip_layers_n_feats[1],
                                          128,
                                          kernel_size=3,
                                          stride=1, padding=1)

        self.skip_layer_conv3 = nn.Conv2d(self.skip_layers_n_feats[2],
                                          128,
                                          kernel_size=3,
                                          stride=1, padding=1)

        self.skip_layer_conv4 = nn.Conv2d(self.skip_layers_n_feats[3],
                                          128,
                                          kernel_size=3,
                                          stride=1, padding=1)

        self.feats_conv = nn.Conv2d(128*4, 128, kernel_size=3,
                                    stride=1, padding=1)

    def forward(self, x):

        upsamp = lambda x: F.interpolate(
            x, size=self.skip_layers_shape, mode='bilinear')

        skip_outputs = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        skip_outputs.append(self.skip_layer_conv1(upsamp(x)))
        x = self.backbone.layer2(x)
        skip_outputs.append(self.skip_layer_conv2(upsamp(x)))
        x = self.backbone.layer3(x)
        skip_outputs.append(self.skip_layer_conv3(upsamp(x)))
        x = self.backbone.layer4(x)
        skip_outputs.append(self.skip_layer_conv4(upsamp(x)))

        stack = torch.cat(skip_outputs, dim=1)
        features = self.feats_conv(stack)
        return features


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        dilation=dilation,
        bias=False)


def conv1x1(in_planes, out_planes, stride=1, dilation=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        dilation=dilation)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, dilation)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] + (2 * pad[0]) -
         (dilation[0] * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) -
         (dilation[1] * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w
