import torch
from torch import nn
from os.path import join as pjoin
from skimage import io, draw, segmentation, color
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import scipy

def make_circle_masks(radius, shape):
    def make_circle_mask(i, j, radius, shape):
        rr, cc = draw.circle(i, j, radius, shape)
        mask = sparse.csr_matrix((np.ones(len(rr), dtype=bool),
                                 (rr, cc)),
                                 shape=shape, dtype=bool).reshape((1, -1))
        return mask
    
    masks = sparse.vstack([make_circle_mask(i, j, radius, shape)
                           for i in range(shape[0])
                           for j in range(shape[1])], format='csr')
    return masks

def scipy_to_torch_coo(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

root_dir = '/home/ubelix/data/medical-labeling'

out_size = 256
radius = 5
p_x, p_y = 34, 50

truth = np.zeros((out_size, out_size))
rr, cc = draw.circle(p_y, p_x, radius, truth.shape)
truth[rr, cc] = 1

# pd = scipy.spatial.distance.pdist(truth.reshape(-1, 1))
# pd = scipy.spatial.distance.squareform(pd)

c = 128+3
b = 4
# masks = make_circle_masks(radius, (out_size, out_size))
truth = truth[None, None, ...]
truth = np.repeat(truth, b, axis=0)
truth = np.repeat(truth, c, axis=1)
b, c, w, h = truth.shape

arr = truth

import pdb; pdb.set_trace() ## DEBUG ##
test = torch.sparse.FloatTensor(b, c, out_size**2, out_size**2)

# test = masks[0, ...].multiply(arr[0][0].reshape(1, w*h))
arr = [[sparse.vstack([masks[i, ...].multiply(x_[c_].reshape(1, w*h))
        for i in range(masks.shape[0])])
        for c_ in range(c)]
       for x_ in arr]
import pdb; pdb.set_trace() ## DEBUG ##

arr = [[sparse.coo_matrix(abs(a_[c_] - a_[c_].T))
             for c_ in range(c)]
             for a_ in arr]

arr = torch.cat([torch.cat([scipy_to_torch_coo(a_[c_])[None, ...]
                            for c_ in range(c)])[None, ...]
             for a_ in arr])


