import numpy as np
import matplotlib.pyplot as plt
from unet_region.rw_net import rw_utils as rwu
from unet_region.rw_net.dummy_sum import DummySum
from unet_region.rw_net.edge_softmax import EdgeSoftmax
from unet_region.rw_net.affinity_branch import AffinityBranch
from unet_region.rw_net.rw_layer import RandomWalk
from os.path import join as pjoin
from skimage import io, draw, segmentation, color, transform
from torch_geometric import nn as gnn
from torch_geometric.data import Data, Batch
import torch

root_dir = '/home/ubelix/data/medical-labeling'

out_size = 100
radius_pw = 5
n_workers = 4

img = io.imread(pjoin(root_dir, 'Dataset20/input-frames/frame_0030.png'))
truth = io.imread(
    pjoin(root_dir, 'Dataset20/ground_truth-frames/frame_0030.png'))
p_x, p_y = 143, 132

img = transform.resize(img, (out_size, out_size))
img_gray = color.rgb2gray(img)
width_bg = 2

truth = transform.resize(truth, (out_size, out_size))
truth[0, :] = 0
truth[-1, :] = 0
truth[:, 0] = 0
truth[:, -1] = 0

truth_contour = (segmentation.find_boundaries(truth > 0))

edge_softmax = EdgeSoftmax()
dummy_sum = DummySum()
aff_branch = AffinityBranch(2, 1)
rw_branch = RandomWalk()

truth = torch.tensor(truth)[None, None, ...].float()

N = 2
C = 1
truth = torch.repeat_interleave(truth, N, 0)
truth = torch.repeat_interleave(truth, C, 1)
W_truth = rwu.make_sparse_pairwise(truth, 1, radius_pw, 1)

# make similarity
A_truth = edge_softmax(W_truth)

# test decimate inversion
A_test = rwu.graph_to_tensors(A_truth.to_data_list()[0]).coalesce()
A_inv = rwu.eigendecomposition_downsample(A_test, 5)

f0 = torch.zeros((N, 2, out_size, out_size))

# foreground seed
f0[:, 1, p_y, p_x] = 1

# background seeds on frame
f0[:, 0, 0, :] = 1
f0[:, 0, -1, :] = 1
f0[:, 0, :, 0] = 1
f0[:, 0, :, -1] = 1

rw_out = rw_branch(A_truth, f0)

fig, ax = plt.subplots(2, 3)
ax = ax.flatten()
ax[0].imshow(f0[0, 0, ...])
ax[1].imshow(f0[0, 1, ...])
ax[2].imshow(rw_out[0, ...].argmax(0))
ax[3].imshow(rw_out[0, 0, ...])
ax[4].imshow(rw_out[0, 1, ...])
ax[5].imshow(truth[0, 0, ...])
fig.show()

# n_iter = 1
# for _ in range(n_iter):

# sim_truth = aff_branch(sim)
# sim_truth = edge_softmax(sim)

# print('is_coalesced {}'.format(sim.is_coalesced()))
# A = rwu.sparse_softmax(sim)
