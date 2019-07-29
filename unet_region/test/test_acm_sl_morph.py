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
import acm_utils as autls
from scipy import ndimage
import math
from scipy.ndimage import distance_transform_edt

def approx_heaviside(s, eps):
    return 0.5 * (1 + (2 / math.pi) * torch.atan(s / eps))

x = torch.linspace(-20, 20, 100)
y = approx_heaviside(x, 1)
# plt.plot(x)
plt.plot(x.cpu().numpy(), y.cpu().numpy())
plt.grid()
plt.show()

root_dir = '/home/ubelix/data_mystique/medical-labeling'

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


dist = distance_transform_edt(np.logical_not(truth_contour))
dy, dx = np.gradient(dist)
norm = np.linalg.norm(np.array((dy, dx)), axis=0)
U = np.array((dy, dx)) / (norm + 1e-7)
U = -U

# gac parameters
max_iter = 5
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
# std = 5
# cone = make_init_ls_gaussian(p_x, p_y, (out_size, out_size), 3)
# init_ls = np.random.normal(scale=1, size=sdf.shape) + sdf
# init_ls[cone != 0] = cone[cone != 0]
init_ls, init_ls_contour = autls.make_sdf(truth > 0,
                                          thr=30,
                                          return_contour=True)
init_ls -= 5

start = time.time()
phi = autls.acm_ls(torch.from_numpy(init_ls),
             # torch.from_numpy(np.concatenate((V[np.newaxis, ...],
             #                                  V[np.newaxis, ...]))),
             torch.from_numpy(U),
             torch.tensor(80),
             1,
             max_iter,
             lambda_=torch.tensor(1),
             # mu=torch.tensor(0.04),
             mu=torch.tensor(0.04),
             vec_field=True)

end = time.time()

print('n_iter: {}, time: {}s'.format(max_iter, end - start))
segm_contour = segmentation.find_boundaries(phi[-1].cpu().numpy() < 0)
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
ax[2].imshow(np.linalg.norm(U, axis=0))
ax[2].set_title('V')
ax[3].imshow(phi[-1].cpu().numpy() < 0)
ax[3].set_title('phi')
fig.show()
