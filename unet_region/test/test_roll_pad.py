from os.path import join as pjoin
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import sobel, gaussian
from skimage.transform import resize
from skimage import io, draw, segmentation, color
import test
from dsac_utils import acm_inference
from acm_utils import *
from scipy import ndimage

a = torch.arange(1, 10)
import torch.nn.functional as F
a_pad = F.pad(a.unsqueeze(0), (1, 1, 0, 0), mode='constant')
print(a_pad)
ap1 = torch.roll(a_pad, -1, 1)
am1 = torch.roll(a_pad, 1, 1)
print(ap1)
print(am1)
da_dx = (ap1 - am1) / 2
da_dx = da_dx[0, 1:-1]
print(a)
print(da_dx)
