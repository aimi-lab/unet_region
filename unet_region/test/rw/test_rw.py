import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io, draw, segmentation, color
from os.path import join as pjoin
from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage

root_dir = '/home/ubelix/data/medical-labeling'

out_size = 256

img = io.imread(pjoin(root_dir, 'Dataset20/input-frames/frame_0030.png'))
truth = io.imread(
    pjoin(root_dir, 'Dataset20/ground_truth-frames/frame_0030.png'))
p_x, p_y = 143, 132

# img = io.imread(pjoin(root_dir, 'Dataset01/input-frames/frame_0150.png'))
# truth = (io.imread(
#     pjoin(root_dir, 'Dataset01/ground_truth-frames/frame_0150.png'))[..., 0] > 0).astype(float)
# p_x, p_y = 190, 100

# img = io.imread(pjoin(root_dir, 'Dataset30/input-frames/frame_0075.png'))[..., :3]
# truth = (io.imread(
#     pjoin(root_dir, 'Dataset30/ground_truth-frames/frame_0075.png'))[..., 0] > 0).astype(float)
# p_x, p_y = 150, 110

# out_size = 22
# p_x, p_y = 9, 9
# rr, cc = draw.ellipse(p_y, p_x, 8, 8, shape=(out_size, out_size), rotation=15)
# img = np.zeros((out_size, out_size, 3))
# truth = np.zeros((out_size, out_size))
# truth[rr, cc] = 1
# img[rr, cc, :] = (0, 1, 0)

img = resize(img, (out_size, out_size))
img_gray = color.rgb2gray(img)
width_bg = 2

truth_ = resize(truth, (out_size, out_size))
truth = np.zeros((truth_.shape[0] + 2*width_bg, truth_.shape[1] + 2*width_bg))
truth[width_bg: -width_bg, width_bg:-width_bg] = truth_
truth_contour = (segmentation.find_boundaries(truth > 0))

data = torch.tensor(truth)[None, None, ...]

markers = np.zeros(truth.shape, dtype=np.uint)
rr, cc = draw.circle(p_y, p_x, 4, shape=markers.shape)
markers[rr, cc] = 2
markers[0:width_bg, :] = 1
markers[-width_bg:, :] = 1
markers[:, 0:width_bg] = 1
markers[:, -width_bg:] = 1

# Run random walker algorithm
labels = random_walker(truth, markers, beta=400, mode='bf',
                       return_full_prob=True)

# Plot results
fig, ax = plt.subplots(1, 5, figsize=(8, 3.2),
                                    sharex=True, sharey=True)
ax = ax.flatten()
ax[0].imshow(img)
ax[0].axis('off')
ax[0].set_title('image')
ax[1].imshow(truth, cmap='gray')
ax[1].axis('off')
ax[1].set_title('Data')
ax[2].imshow(markers, cmap='magma')
ax[2].axis('off')
ax[2].set_title('Markers')
ax[3].imshow(labels[0])
ax[3].axis('off')
ax[3].set_title('Labels 0')
ax[4].imshow(labels[1])
ax[4].axis('off')
ax[4].set_title('Labels 1')

fig.tight_layout()
plt.show()
