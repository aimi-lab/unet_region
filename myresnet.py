import torch
from torch import nn
from torch.nn import functional as F


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

    def __init__(self, num_classes=1000,
                 zero_init_residual=False,
                 in_shape=224,
                 mode='classification'):
        super(MyResNet, self).__init__()

        layers = [3, 4, 23, 3]

        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(
            Bottleneck, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            Bottleneck, 512, layers[3], stride=1, dilation=2)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        # skip-layers: find
        self.skip_layers_shapes = []
        for m in self.layers:
            shape = conv_output_shape(
                (in_shape, in_shape), self.layer1.conv3.kernel_size,
                self.layer1.conv3.stride, self.layer1.conv3.pad,
                self.layer1.conv3.dilation)
            self.skip_layers_shapes.append(shape)

        all_concat_dim = [s[-2] for s in self.skip_layers_shapes]
        self.skip_layers_module = nn.Sequential(*[
            nn.Conv2d(all_concat_dim, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn3.weight, 0)

        
        ok_modes = ['classification', 'features']
        self.mode = mode
        if mode not in ok_modes:
            raise Exception('mode must be one of {}'.format(
                ok_modes))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride,
                        dilation),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        upsamp = lambda x: F.interpolate(
            x, size=self.skip_layers_shapes[0][2:4], mode='bilinear')

        skip_outputs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        skip_outputs.append(x)
        x = self.layer2(x)
        skip_outputs.append(upsamp(x))
        x = self.layer3(x)
        skip_outputs.append(upsamp(x))
        x = self.layer4(x)
        skip_outputs.append(upsamp(x))

        if(self.mode == 'features'):
            stack = torch.cat(skip_outputs, dim=1)
            features = self.skip_layers_module(stack)
            return features

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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
         (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) -
         (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w
