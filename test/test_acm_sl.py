import torch

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import sobel, gaussian
from skimage import segmentation
from skimage.transform import resize
from skimage import io
import test
from dsac_utils import acm_inference

out_size = 256
img = io.imread('/home/ubelix/medical-labeling/Dataset20/input-frames/frame_0030.png')
truth = io.imread('/home/ubelix/medical-labeling/Dataset20/ground_truth-frames/frame_0030.png') / 255
img = resize(img, (out_size, out_size))
truth = resize(truth, (out_size, out_size))
contour = segmentation.find_boundaries(truth, mode='thick')
contour =  gaussian(contour, 3)
truth = gaussian(truth, 5)

gamma = 1
alpha=0.0
beta=10
kappa = 0.01

L = 20

s = np.linspace(0, 2*np.pi, L)
radius = 0.02
x = 142 + radius*img.shape[1]*np.cos(s)
y = 131 + radius*img.shape[0]*np.sin(s)

init_snake = np.array([x, y]).T

# compute edge map (data term)
# map_e = np.zeros(img.shape[:2])[np.newaxis, np.newaxis, ...]
map_e = 1 * contour[np.newaxis, np.newaxis, ...]
map_a = alpha * np.ones(map_e.shape)
map_b = beta * np.ones(map_e.shape)
map_k = kappa * (1 - contour)[np.newaxis, np.newaxis, ...]
# map_k = kappa * truth[np.newaxis, np.newaxis, ...]
# map_k = kappa * np.ones(map_e.shape)
max_px_move = 1
niter = 50

delta_s = np.max(map_e.shape) / L

snakes = acm_inference(map_e,
                       map_a, map_b, map_k, init_snake,
                       gamma,
                       delta_s,
                       max_px_move, niter,
                       verbose=True)

fig, ax = plt.subplots(2, 2)

n_snakes_to_draw = 2
ax[0, 0].imshow(img, cmap=plt.cm.gray)
ax[0, 0].set_title('with balloon term')
ax[0, 0].plot(init_snake[:, 0], init_snake[:, 1], '--r', lw=3)
snake = snakes[-1]
ax[0, 0].plot(snake[:, 0], snake[:, 1], 'bx-', lw=3)
# for n in np.linspace(0, niter - 1, n_snakes_to_draw, dtype=int):
#     snake = snake_hist[-1][n].numpy()
#     ax[1].plot(snake[:, 0], snake[:, 1], '--')

ax[0, 1].imshow(map_e[0, 0, ...])
ax[0, 1].set_title('map_e')
ax[1, 0].imshow(map_b[0, 0, ...])
ax[1, 0].set_title('map_b')
ax[1, 1].imshow(map_k[0, 0, ...])
ax[1, 1].set_title('map_k')

fig.show()
