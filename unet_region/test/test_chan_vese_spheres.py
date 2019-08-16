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

data = torch.tensor(gaussian(truth, sigma=3)[None, ...]).float()
r0 = torch.tensor([10.])[None, ...]
N = 5
nu = 5.
step_size = 0.01
step_size_phase = 0.5
decay = 0.96
tol = 0.0001
n_iter = 400
a = torch.zeros(1, N)
b = torch.zeros(1, N)
alpha = torch.tensor([0.])[None, ...]
# alpha = torch.zeros(1, N)
# beta = torch.zeros(1, N)
center = np.array((p_x, p_y))[None, ...]
lambda_1 = 10
lambda_2 = 1
r0[0] = 20.
# a[0, 3] = 12.
# b[0, 0] = 10
phi = autls.make_phi_spheres(center, r0, a, b, alpha, truth.shape)

out = autls.acwe_sphere(center, r0, a, b,
                        alpha,
                        data,
                        step_size=step_size,
                        step_size_phase=step_size_phase,
                        n_iter=n_iter,
                        decay=decay,
                        nu=nu,
                        tol=tol)
print('r0: {}'.format(out['r0']))
print('a: {}'.format(out['a']))
print('b: {}'.format(out['b']))
print('alpha: {}'.format(out['alpha']))

phi_boundary = segmentation.find_boundaries(phi.detach().cpu().numpy() < 0)
img0 = img.copy()
img0[phi_boundary[0, ...], :] = (1, 0, 0)

img_last = img.copy()
phi_new_boundary = segmentation.find_boundaries(out['phi'].detach().cpu().numpy() < 0)
img_last[phi_new_boundary[0, ...], :] = (0, 0, 1)

fig, ax = plt.subplots(2, 2)
fig.suptitle('At last iter. {}'.format(len(out['phi_history'])))
ax = ax.flatten()
ax[0].imshow(img_last)
ax[1].imshow(out['phi'][0, ...])
ax[2].imshow(truth)
ax[3].imshow(data[0, ...])
fig.show()

best_iter = np.argmin(np.array(out['E_in']) + np.array(out['E_out']))
img_best = img.copy()
phi_best_boundary = segmentation.find_boundaries(out['phi_history'][best_iter].detach().cpu().numpy() < 0)
img_best[phi_best_boundary[0, ...], :] = (0, 0, 1)
fig, ax = plt.subplots(2, 2)
fig.suptitle('At best iter. {}'.format(best_iter))
ax = ax.flatten()
ax[0].imshow(img_best)
ax[1].imshow(out['phi_history'][best_iter][0, ...])
ax[2].imshow(truth)
ax[3].imshow(data[0, ...])
fig.show()

# test heaviside / dirac
fig, ax = plt.subplots(3, 2)
ax = ax.flatten()
ax[0].plot(out['E'], 'bo-')
ax[0].set_ylabel('Energy')
ax[0].set_xlabel('Iterations')
ax[1].plot(out['Evar'], 'bo-')
ax[1].set_ylabel('Phi var.')
ax[1].set_xlabel('Iterations')
ax[2].plot(out['E_in'], 'bo-')
ax[2].set_ylabel('E_in')
ax[2].set_xlabel('Iterations')
ax[3].plot(out['E_out'], 'bo-')
ax[3].set_ylabel('E_out')
ax[3].set_xlabel('Iterations')
ax[4].plot(np.array(out['E_out']) + np.array(out['E_in']), 'bo-')
ax[4].set_ylabel('E')
ax[4].set_xlabel('Iterations')
fig.show()
