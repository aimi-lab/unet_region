import numpy as np
import matplotlib.pyplot as plt
from patch_loader import PatchLoader
import math
from dsac_utils import make_spline_contour

loader = PatchLoader(
    '/home/ubelix/medical-labeling/Dataset01/',
    'hand',
    fake_len=100,
    fix_frames=[30])

np.random.seed(14)

sample = loader[0]
truth = sample['segmentation'][..., 0]
im = sample['image']

L = 100
s = 2
k = 1
per = 1
ds_contour_rate = 0.1

nodes = make_spline_contour(truth, L, s, k, per, ds_contour_rate)

plt.imshow(im)
plt.plot(nodes[:, 0], nodes[:, 1], 'bo--')
for i in range(nodes.shape[0]):
    plt.annotate(str(i), (nodes[i, 0], nodes[i, 1]))
plt.show()
