import sys
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
from unet_region.test import maxflow_utils as mfutls
from scipy import ndimage
import math
import maxflow
from scipy.ndimage import distance_transform_edt

def approx_heaviside(s, eps):
    return 0.5 * (1 + (2 / math.pi) * torch.atan(s / eps))

root_dir = '/home/ubelix/data/medical-labeling'

out_size = 256

# img = io.imread(pjoin(root_dir, 'Dataset20/input-frames/frame_0030.png'))
# truth = io.imread(
#     pjoin(root_dir, 'Dataset20/ground_truth-frames/frame_0030.png'))
# p_x, p_y = 143, 132

# img = io.imread(pjoin(root_dir, 'Dataset01/input-frames/frame_0150.png'))
# truth = (io.imread(
#     pjoin(root_dir, 'Dataset01/ground_truth-frames/frame_0150.png'))[..., 0] > 0).astype(float)
# p_x, p_y = 190, 100

# img = io.imread(pjoin(root_dir, 'Dataset30/input-frames/frame_0075.png'))[..., :3]
# truth = (io.imread(
#     pjoin(root_dir, 'Dataset30/ground_truth-frames/frame_0075.png'))[..., 0] > 0).astype(float)
# p_x, p_y = 150, 110

out_size = 9
p_x, p_y = 3, 3
rr, cc = draw.ellipse(p_y, p_x, 8, 8, shape=(out_size, out_size), rotation=15)
img = np.zeros((out_size, out_size, 3))
truth = np.zeros((out_size, out_size))
truth[rr, cc] = 1
img[rr, cc, :] = (0, 1, 0)

img = resize(img, (out_size, out_size))
img_gray = color.rgb2gray(img)
truth = resize(truth, (out_size, out_size))
truth_contour = (segmentation.find_boundaries(truth > 0))

# Create the graph.
g = maxflow.Graph[float]()
# Add the nodes. nodeids has the identifiers of the nodes in the grid.
nodeids = g.add_grid_nodes(truth.shape)

# top right edges on top right quadrant
structure = np.array([[0, 0, 1],
                      [0, 0, 0],
                      [0, 0, 0]])
C_topright = truth * np.roll(np.roll(truth, 1, axis=0), -1, axis=1)
g.add_grid_edges(nodeids[:p_y+1, p_x:],
                 structure=structure,
                 weights=C_topright[:p_y+1, p_x:])

# top left edges on top left quadrant
structure = np.array([[1, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])
C_topleft = truth * np.roll(np.roll(truth, 1, axis=0), 1, axis=1)
g.add_grid_edges(nodeids[:p_y+1, :p_x+1],
                 structure=structure,
                 weights=C_topleft[:p_y+1, :p_x+1],
                 symmetric=False)

# bottom left edges on bottom left quadrant
structure = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [1, 0, 0]])
C_bottomleft = truth * np.roll(np.roll(truth, -1, axis=0), 1, axis=1)
g.add_grid_edges(nodeids[p_y:, :p_x+1],
                 structure=structure,
                 weights=C_bottomleft[p_y:, :p_x+1],
                 symmetric=False)

# bottom right edges on bottom right quadrant
structure = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 1]])
C_bottomleft = truth * np.roll(np.roll(truth, -1, axis=0), -1, axis=1)
g.add_grid_edges(nodeids[p_y:, p_x:],
                 structure=structure,
                 weights=C_bottomleft[p_y:, p_x:],
                 symmetric=False)

# top edges on top half
structure = np.array([[0, 1, 0],
                      [0, 0, 0],
                      [0, 0, 0]])
C_top = truth * np.roll(truth, 1, axis=0)
g.add_grid_edges(nodeids[:p_y+1, :],
                 structure=structure,
                 weights=C_top[:p_y+1, :],
                 symmetric=False)

# bottom edges on bottom half
structure = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 0]])
C_bottom = truth * np.roll(truth, -1, axis=0)
g.add_grid_edges(nodeids[p_y:, :],
                 structure=structure,
                 weights=C_bottom[p_y:, :],
                 symmetric=False)

# right edges on right half
structure = np.array([[0, 0, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
C_right = truth * np.roll(truth, -1, axis=1)
g.add_grid_edges(nodeids[:, p_x:],
                 structure=structure,
                 weights=C_right[:, p_x:],
                 symmetric=False)
# left edges on left half
structure = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 0, 0]])
C_left = truth * np.roll(truth, 1, axis=1)
g.add_grid_edges(nodeids[:, :p_x+1],
                 structure=structure,
                 weights=C_left[:, :p_x+1],
                 symmetric=False)

mfutls.plot_graph_2d(g, nodeids.shape)

# connect source and sink edges
# - from source: all have null cap except "center" node
# - to sink: all have large capacities except "center" node
sink_caps = 1 - truth
sink_caps[p_y, p_x] = 0

# source on truth
src_caps = truth.copy()
src_caps[p_y, p_x] = 1

# source on center pixel
# src_caps = np.zeros(truth.shape)
# src_caps[p_y, p_x] = truth.size

# source on circle around center
# src_caps = np.zeros(truth.shape)
# rr, cc = draw.circle(p_y, p_x, 4, shape=(out_size, out_size))
# src_caps[rr, cc] = 1
# sink_caps[rr, cc] = 0

g.add_grid_tedges(nodeids, src_caps, sink_caps)

# Find the maximum flow.
flow = g.maxflow()
print('max flow: {}'.format(flow))
# Get the segments of the nodes in the grid.
sgm = np.logical_not(g.get_grid_segments(nodeids))

fig, ax = plt.subplots(2, 2)
ax = ax.flatten()
ax[0].imshow(img)
ax[1].imshow(truth)
ax[2].imshow(sgm)
fig.show()
