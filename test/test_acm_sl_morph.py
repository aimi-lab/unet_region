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


root_dir = '/home/ubelix/medical-labeling'

out_size = 256

# img = io.imread(pjoin(root_dir, 'Dataset20/input-frames/frame_0030.png'))
# truth = io.imread(
#     pjoin(root_dir, 'Dataset20/ground_truth-frames/frame_0030.png'))
# p_x, p_y = 143, 132

img = io.imread(pjoin(root_dir, 'Dataset01/input-frames/frame_0150.png'))
truth = (io.imread(
    pjoin(root_dir, 'Dataset01/ground_truth-frames/frame_0150.png'))[..., 0] > 0).astype(float)
p_x, p_y = 190, 100

# img = io.imread(pjoin(root_dir, 'Dataset30/input-frames/frame_0075.png'))[..., :3]
# truth = (io.imread(
#     pjoin(root_dir, 'Dataset30/ground_truth-frames/frame_0075.png'))[..., 0] > 0).astype(float)
# p_x, p_y = 150, 110

img = resize(img, (out_size, out_size))
img_gray = color.rgb2gray(img)
truth = resize(truth, (out_size, out_size))
truth_contour = (segmentation.find_boundaries(truth > 0))

V = segmentation.inverse_gaussian_gradient(truth, sigma=1)
V = V - V.min()
V = V / V.max()
V -= 0.5
# sdf = make_sdf(truth_contour, thr=30)

# plt.subplot(121)
# plt.imshow(V)
# plt.subplot(122)
# plt.imshow(truth_contour)
# plt.show()

# modify input
# input[p_y - 5: p_y + 5, :] = 1
# input += np.random.normal(0.5, 0.05, (out_size, out_size))

# gac parameters
max_iter = 50
thr = 0.9

# tweezer
balloon=3
smoothing = 1

# slitlamp
balloon = 1
smoothing = 1

# brain
balloon = 1
smoothing = 2

# make initial level set
std = 5
cone = make_init_ls_gaussian(p_x, p_y, (out_size, out_size), 1)
# init_ls = np.random.normal(scale=1, size=sdf.shape) + sdf
# init_ls[cone != 0] = cone[cone != 0]
init_ls = cone

init_ls_contour = segmentation.find_boundaries(init_ls > 0)

start = time.time()
phi = acm_ls(init_ls, V, 0.8, 1, max_iter,
             lambda_=1, mu=0.04)

end = time.time()

print('n_iter: {}, time: {}s'.format(max_iter, end - start))
segm_contour = segmentation.find_boundaries(phi[-1] < 0)
colors = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
labels = ['segm', 'init', 'truth']
img[segm_contour] = colors[0]
img[init_ls_contour] = colors[1]
# img[truth_contour] = colors[2]

patches = [
    mpatches.Patch(color=colors[i], label="{}".format(labels[i]))
    for i in range(len(labels))
]

plt.ion()
fig, ax = plt.subplots(2, 2)
ax = ax.flatten()
ax[0].imshow(img)
ax[0].set_title('img')
ax[0].legend(handles=patches,
             bbox_to_anchor=(1.05, 1),
             loc=2,
             borderaxespad=0.)
ax[1].imshow(init_ls)
ax[1].set_title('init_ls')
ax[2].imshow(V)
ax[2].set_title('V')
ax[3].imshow(phi[-1])
ax[3].set_title('phi')
fig.show()
