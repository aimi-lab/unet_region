from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.transform import resize
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import io, draw, segmentation, color
from acm_utils import *


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
truth = resize(truth, (out_size, out_size))

energy = make_energy_ws(truth > 0)
markers = np.zeros_like(energy).astype(np.uint8)
markers[p_y, p_x] = 1
labels = watershed(-energy, markers)

img[p_y, p_x] = (1, 0, 0)

fig, ax = plt.subplots(2, 2)
ax = ax.flatten()
ax[0].imshow(img)
ax[0].set_title('img')
ax[1].imshow(energy)
ax[1].set_title('energy')
ax[2].imshow(labels)
ax[2].set_title('labels')
ax[3].imshow(markers)
ax[3].set_title('markers')
plt.show()
