import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import sobel, gaussian
from skimage import segmentation
from skimage.transform import resize
from skimage import io, draw
import test
from dsac_utils import acm_inference

out_size = 256
img = io.imread('/home/ubelix/medical-labeling/Dataset01/input-frames/frame_0140.png')
truth = (io.imread('/home/ubelix/medical-labeling/Dataset01/ground_truth-frames/frame_0140.png').sum(axis=-1) > 0) / 255
img = resize(img, (out_size, out_size))
truth = resize(truth, (out_size, out_size))
contour = segmentation.find_boundaries(truth)
contour_smooth =  gaussian(contour, 6)[..., np.newaxis]

gamma = 0.1
alpha=0.0
beta=10
kappa = 4

L = 300

s = np.linspace(0, 2*np.pi, L)
radius = 0.01

rr_ls, cc_ls = draw.circle(114, 166, radius*out_size, shape=(out_size, out_size))
init_ls = np.zeros((out_size, out_size), dtype=bool)
init_ls[rr_ls, cc_ls] = True

# compute edge map (data term)
map_e = (1 - contour_smooth)[..., 0]
# map_e = segmentation.inverse_gaussian_gradient(img)[..., 0]
map_a = alpha * np.ones(map_e.shape)
map_b = beta * np.ones(map_e.shape)
map_k = kappa * np.ones(map_e.shape)
max_px_move = 1
niter = 100


start = time.time()
snake_arr = segmentation.morphological_geodesic_active_contour(map_e/2, niter,
                                                           init_level_set=init_ls,
                                                               threshold=0.8,
                                                           balloon=kappa)
end = time.time()

print('n_iter: {}, time: {}s'.format(niter, end-start))

init_ls_contour = segmentation.find_boundaries(init_ls, mode='thick')
rr, cc = np.where(init_ls_contour)
img[rr, cc, ...] = (1, 0, 0)

ls_contour = segmentation.find_boundaries(snake_arr, mode='thick')
rr, cc = np.where(ls_contour)
img[rr, cc, ...] = np.array((0, 0, 1))

# rr, cc = np.where(contour)
# img[rr, cc, ...] = np.array((1, 0, 0))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[1].imshow(map_e)
fig.show()
