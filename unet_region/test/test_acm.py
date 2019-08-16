import torch

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import sobel, gaussian
from skimage.segmentation import active_contour
from skimage.transform import resize
import test
from dsac_utils import acm_inference


img = data.astronaut()
img = rgb2gray(img)
img = resize(img, (256, 256))


s = np.linspace(0, 2*np.pi, 300)

gamma = 0.01

alpha=0.015
beta=10

# contract to head
radius = 0.2
x = 110 + radius*img.shape[1]*np.cos(s)
y = 60 + radius*img.shape[0]*np.sin(s)
kappa = 30

# expand on helmet
# radius = 0.02
# x = 194 + radius*img.shape[1]*np.cos(s)
# y = 199 + radius*img.shape[0]*np.sin(s)
# kappa = 3

init = np.array([x, y]).T

snake = active_contour(gaussian(img, 3),
                       init, alpha=alpha,
                       beta=beta, gamma=gamma)

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))
ax[0, 0].imshow(img, cmap=plt.cm.gray)
ax[0, 0].plot(init[:, 0], init[:, 1], '--r', lw=3)
ax[0, 0].plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].axis([0, img.shape[1], img.shape[0], 0])
ax[0, 0].set_title('no balloon term')

init_snake = np.array([x, y]).T

# compute edge map (data term)
# map_e = np.zeros(img.shape[:2])[np.newaxis, np.newaxis, ...]
map_e = sobel(gaussian(img, 3))[np.newaxis, np.newaxis, ...]
map_a = alpha * np.ones(map_e.shape)
map_b = beta * np.ones(map_e.shape)
map_k = kappa * np.ones(map_e.shape)
max_px_move = 1
delta_s = 1
niter = 200

snakes = acm_inference(map_e,
                       map_a, map_b, map_k, init_snake,
                       gamma, max_px_move, niter,
                       verbose=True)



n_snakes_to_draw = 2
ax[0, 1].imshow(img, cmap=plt.cm.gray)
ax[0, 1].set_title('with balloon term')
ax[0, 1].plot(init_snake[:, 0], init_snake[:, 1], '--r', lw=3)
snake = snakes[-1]
ax[0, 1].plot(snake[:, 0], snake[:, 1], '-b', lw=3)
# for n in np.linspace(0, niter - 1, n_snakes_to_draw, dtype=int):
#     snake = snake_hist[-1][n].numpy()
#     ax[1].plot(snake[:, 0], snake[:, 1], '--')

ax[1, 0].imshow(map_e[0, 0, ...])
ax[1, 0].set_title('map_e')
ax[1, 1].imshow(map_b[0, 0, ...])
ax[1, 1].set_title('map_b')
ax[2, 0].imshow(map_k[0, 0, ...])
ax[2, 0].set_title('map_k')

fig.show()
