import torch
from skimage import io, draw, segmentation, color, transform
from os.path import join as pjoin
import numpy as np
import unet_region.fourier.fourier_utils as utls
import matplotlib.pyplot as plt


root_dir = '/home/ubelix/data/medical-labeling'

out_size = 256

# img = io.imread(pjoin(root_dir, 'Dataset20/input-frames/frame_0030.png'))
# truth = io.imread(
#     pjoin(root_dir, 'Dataset20/ground_truth-frames/frame_0030.png'))
# px, py = 143, 132

img = io.imread(pjoin(root_dir, 'Dataset01/input-frames/frame_0150.png'))
truth = (io.imread(
    pjoin(root_dir, 'Dataset01/ground_truth-frames/frame_0150.png'))[..., 0] > 0).astype(float)
px, py = 190, 100


# img = io.imread(pjoin(root_dir, 'Dataset30/input-frames/frame_0075.png'))[..., :3]
# truth = (io.imread(
#     pjoin(root_dir, 'Dataset30/ground_truth-frames/frame_0075.png'))[..., 0] > 0).astype(float)
# px, py = 150, 110

img = transform.resize(img, (out_size, out_size))
img_gray = color.rgb2gray(img)
truth = transform.resize(truth, (out_size, out_size))
truth_contour = (segmentation.find_boundaries(truth > 0))

init_radius = 0.05 * np.max(img.shape)

n_q = 51
n = 50
Q = torch.zeros(1, n_q, 1)
Q[:, 0] = init_radius
# Q[2] = init_radius / 2

rc = torch.tensor((px, py)).type(torch.float)[None, ..., None]

# r = utls.get_cart_coords(Q, torch.tensor(0.), torch.tensor((px, py)).type(torch.float))
c = utls.get_contour(Q, rc, n)

utls.build_b(Q, torch.tensor(0.))

c = c.detach().cpu().numpy()
M_in = utls.get_mask(Q, rc, n, truth.shape)

b = 0

fig, ax = plt.subplots(2, 2)
ax = ax.flatten()
ax[0].imshow(img)
ax[0].plot(c[b, :, 0], c[b, :, 1], 'ro')
ax[1].imshow(truth)
ax[2].imshow(M_in[b, ...])
plt.show()
