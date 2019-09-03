from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
from functools import partial
import torch_geometric.transforms as T 
import torch.nn.functional as F
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

def scipy_coo_to_torch_iv(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return i, v, torch.Size(shape)

def scipy_coo_to_torch(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def graph_to_tensors(data):
    edge_index, attr = data.edge_index, data.edge_attr

    shape_sparse = torch.Size((data.shape[0]**2, data.shape[1]**2))
    tnsr = torch.sparse.FloatTensor(edge_index,
                                    attr.reshape(-1),
                                    shape_sparse)

    return tnsr

def compute_distance(data, p=1):

    (row, col), pos, pseudo, x = data.edge_index, data.pos, data.edge_attr, data.x

    dist = torch.norm(x[col] - x[row], p=p, dim=-1).view(-1, 1)

    dist = dist / dist.max()
    data.edge_attr = dist

    return data

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def run_transforms(x, p, radius):

    w, h = x.shape
    xv, yv = torch.meshgrid([torch.arange(w), torch.arange(h)])
    data = Data(x=x.reshape(-1, 1),
                pos=torch.cat((xv.reshape(-1, 1),
                               yv.reshape(-1, 1)), dim=1),
                shape=torch.tensor([w, h]))
    rad_graph = T.RadiusGraph(radius, loop=True)
    data = rad_graph(data)
    data = compute_distance(data, p)
    

    return data

def merge_edge_attr(data_list):
    edge_attr = torch.cat([d.edge_attr for d in data_list], dim=-1)
    node_attr = torch.cat([d.x for d in data_list], dim=-1)
    pos = data_list[0].pos
    edge_index = data_list[0].edge_index

    data = Data(x=node_attr, edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                shape=data_list[0].shape)
    return data
    

def make_sparse_pairwise(batch, p, radius, n_workers):

    n, c, w, h = batch.shape
    data_list = [batch[n_, c_, ...]
                 for c_ in range(c) for n_ in range(n)]

    fun = partial(run_transforms, p=p, radius=radius)
    # if(n_workers > 1 or n*c > 1):
    if(n_workers > 1):
        pool = mp.Pool(processes=n_workers)
        sparse_pw = pool.map(fun, data_list)
    else:
        sparse_pw = [fun(d) for d in data_list]

    # merge node attributes
    sparse_pw = list(chunks([g for g in sparse_pw], c))
    sparse_pw = [merge_edge_attr(d) for d in sparse_pw]

    # construct batch
    batch = Batch.from_data_list(sparse_pw)
    return batch

def sparse_softmax(x):
    n, c, w, h = x.shape
    exp = torch.exp(x.values())
    rows = x.indices()[2, :]
    cols = x.indices()[3, :]
    # x.values() = F.softmax(x.values()[]


if __name__ == "__main__":
    n, c, w, h = 1, 2, 64, 64
    arr = torch.randn((n, c, w, h))

    res = make_sparse_pairwise(arr, p=1, radius=5, n_workers=4)

    res = res.to_dense()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(res[0, 0, ...])
    ax[1].imshow(res[0, 1, ...])
    fig.show()
