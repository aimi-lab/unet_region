import torch

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import test
from dsac_utils import acm_inference


img = data.astronaut()
img = rgb2gray(img)

s = np.linspace(0, 2*np.pi, 300)
x = 400 + 30*np.cos(s)
y = 400 + 30*np.sin(s)
init = np.array([x, y]).T

snake = active_contour(gaussian(img, 3),
                       init, alpha=0.015, beta=10, gamma=0.001)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 7))
ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].plot(init[:, 0], init[:, 1], '--r', lw=3)
ax[0].plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].axis([0, img.shape[1], img.shape[0], 0])
ax[0].set_title('no balloon term')

init_snake = np.array([x, y]).T

map_e = img[np.newaxis, np.newaxis, ...]
map_a = 0.015 * np.ones(map_e.shape)
map_b = 10 * np.ones(map_e.shape)
map_k = 10 * np.ones(map_e.shape)
gamma = 0.001
max_px_move = 1
delta_s = 1
niter = 100

snake_hist = acm_inference(map_e, map_a, map_b, map_k, init_snake,
                           gamma, max_px_move, delta_s, niter, 3)

n_snakes_to_draw = 2
ax[1].imshow(img, cmap=plt.cm.gray)
ax[1].set_title('with balloon term')
ax[1].plot(init_snake[:, 0], init_snake[:, 1], '--r', lw=3)
snake = snake_hist[-1][-1]
ax[1].plot(snake[:, 0], snake[:, 1], '-b', lw=3)
# for n in np.linspace(0, niter - 1, n_snakes_to_draw, dtype=int):
#     snake = snake_hist[-1][n].numpy()
#     ax[1].plot(snake[:, 0], snake[:, 1], '--')

fig.show()
