import torch
import numpy as np

shape = 64
in_chans = 1
out_chans = 1
ks = 1
stride = 1

input = torch.ones(1, 1, shape, shape)

conv = torch.nn.Conv2d(in_chans, out_chans, kernel_size=ks,
                       stride=stride, bias=True)

output = conv(input)

print(np.prod(conv.weight.shape) + np.prod(conv.bias.shape))
