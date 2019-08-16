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
from unet_region import acm_utils as autls
from scipy import ndimage
import math
from scipy.ndimage import distance_transform_edt

def approx_heaviside(s, eps):
    return 0.5 * (1 + (2 / math.pi) * torch.atan(s / eps))

root_dir = '/home/ubelix/data/medical-labeling'

out_size = 256

# img = io.imread(pjoin(root_dir, 'Dataset20/input-frames/frame_0030.png'))
# truth = io.imread(
#     pjoin(root_dir, 'Dataset20/ground_truth-frames/frame_0030.png'))
# p_x, p_y = 143, 132

# img = io.imread(pjoin(root_dir, 'Dataset01/input-frames/frame_0150.png'))
# truth = (io.imread(
#     pjoin(root_dir, 'Dataset01/ground_truth-frames/frame_0150.png'))[..., 0] > 0).astype(float)
# p_x, p_y = 190, 100

# img = io.imread(pjoin(root_dir, 'Dataset30/input-frames/frame_0075.png'))[..., :3]
# truth = (io.imread(
#     pjoin(root_dir, 'Dataset30/ground_truth-frames/frame_0075.png'))[..., 0] > 0).astype(float)
# p_x, p_y = 150, 110

p_x, p_y = 190, 100
rr, cc = draw.ellipse(p_y, p_x, 40, 20, shape=(out_size, out_size), rotation=15)
img = np.zeros((out_size, out_size, 3))
img[rr, cc, ...] = 1
truth = np.zeros((out_size, out_size))
truth[rr, cc] = 1

img = resize(img, (out_size, out_size))
img_gray = color.rgb2gray(img)
truth = resize(truth, (out_size, out_size))
truth_contour = (segmentation.find_boundaries(truth > 0))

rr, cc = draw.circle(p_y, p_x, 5, shape=img.shape)
img[rr, cc, :] = (0, 1, 0)

rbfstore = autls.RBFstore((p_x, p_y), 20, 20, 20, truth.shape)
rbf = rbfstore[0, 0, 0]

a = torch.zeros(len(rbfstore))
rbfstore.make_phi(15, a)

fig, ax = plt.subplots(1, 2)
ax = ax.flatten()
ax[0].imshow(rbf)
ax[1].imshow(rbf > 0.1)
plt.show()

print(len(rbfstore))
