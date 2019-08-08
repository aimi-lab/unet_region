from os.path import join as pjoin
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io, draw, segmentation, color
import test
import unet_region.baselines.darnet_.utils as utls
from scipy.ndimage import distance_transform_edt
import copy


root_dir = '/home/ubelix/data/data/medical-labeling'

out_size = 256

# img = io.imread(pjoin(root_dir, 'Dataset20/input-frames/frame_0030.png'))
# truth = io.imread(
#     pjoin(root_dir, 'Dataset20/ground_truth-frames/frame_0030.png'))
# p_x, p_y = 143, 132

img = io.imread(pjoin(root_dir, 'Dataset00/input-frames/frame_0400.png'))
truth = (io.imread(
    pjoin(root_dir, 'Dataset00/ground_truth-frames/frame_0400.png'))[..., 0] > 0).astype(float)
p_x, p_y = 180, 100

# img = io.imread(pjoin(root_dir, 'Dataset30/input-frames/frame_0075.png'))[..., :3]
# truth = (io.imread(
#     pjoin(root_dir, 'Dataset30/ground_truth-frames/frame_0075.png'))[..., 0] > 0).astype(float)
# p_x, p_y = 150, 110

img = resize(img, (out_size, out_size))
img_gray = color.rgb2gray(img)
truth = resize(truth, (out_size, out_size))
truth_contour = (segmentation.find_boundaries(truth > 0))

L = 100
init_radius = 6
n_iter = 100
max_px_move = 1

init_snake = utls.RaySnake((p_x,
                            p_y), L, init_radius,
                           truth.shape)
r = init_snake.cart_coords

data = distance_transform_edt(np.logical_not(truth_contour))

beta = data.copy()
beta[truth > 0] = 0
beta /= 0.005

kappa = 10*data.copy()
kappa[truth == 0] = 0
kappa /= 0.1

snake_truth = utls.RaySnake((p_x, p_y), L, arr=truth)

snake = utls.active_contour_steps(data,
                                  copy.copy(init_snake),
                                  beta,
                                  kappa,
                                  max_px_move=max_px_move,
                                  max_iterations=n_iter,
                                  verbose=True)

fig, ax = plt.subplots(2, 2)
ax = ax.flatten()
img[truth_contour, ...] = (0, 0, 1)
ax[0].imshow(img)
ax[0].plot(init_snake.cart_coords[:, 0],
           init_snake.cart_coords[:, 1],
           'bx-')
ax[0].plot(p_x,
           p_y,
           'go')
ax[0].plot(snake.cart_coords[:, 0], snake.cart_coords[:, 1],
           'rx-')
ax[0].set_title('image')
ax[1].imshow(data)
ax[1].set_title('data')
ax[2].imshow(kappa)
ax[2].set_title('kappa')
ax[3].imshow(beta)
ax[3].set_title('beta')
fig.show()
