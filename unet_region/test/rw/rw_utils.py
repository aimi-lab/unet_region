import scipy
from scipy import ndimage as ndi
import numpy as np
import torch
from torch.nn import functional as F
import networkx as nx
from numpy.lib.stride_tricks import as_strided
from unet_region.test.rw.rag_rw import RAG

def make_affinity_l1(arr, radius):
    # arr is tensor of shape (N, C, W, H)
    n, c, w, h = arr.shape

    A = []

    for n_ in range(n):
        A.append([])
        for c_ in range(c):
            print('N: {}/{} ; C: {}/{}'.format(n_+1, n, c_+1, c))
            im = arr[n_, c_, ...]
            print('making RAG')
            rag = RAG(im.shape, radius=radius)
            print('making adj matrix')
            A_ = rag.make_dist_adjacency(im)
            # A[-1] += A_
            # k_ = torch.zeros((n, c, 2 * radius + 1, 2 * radius + 1))
            # x = int(i / (2 * radius + 1))
            # y = int(i % (2 * radius + 1))
            # k_[:, :, y, x] = -1
            # k_[:, :, radius, radius] = 1

            # # calculate padding
            # center = np.array((radius, radius))
            # padding = np.array((x, y)).tolist()
            # diag = F.conv2d(arr, k_, padding=padding)

            # # calculate diagonal indices
            # lin_index_neighb = np.ravel_multi_index((y, x), k_[0, 0, ...].shape)
            # lin_index_center = np.ravel_multi_index((radius, radius), k_[0, 0, ...].shape)
            # diag_k = 
            # indices = kth_diag_indices((w, h), )
            # diags.append((diag))

n, c, w, h = 1, 1, 256, 256
arr = torch.randn((n, c, w, h))
rag = RAG((w, h), arr[0, 0, ...], radius=3)
# rag.draw()
# A = make_affinity_l1(arr, 3)
