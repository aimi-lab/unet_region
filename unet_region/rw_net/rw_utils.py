from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy import sparse
from functools import partial
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt


def torch_to_scipy_csr(arr):
    values = arr.values()
    indices = arr.indices()
    shape = arr.shape

    return sparse.csr_matrix((values, indices), shape=shape)


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
    tnsr = torch.sparse.FloatTensor(edge_index, attr.reshape(-1), shape_sparse)

    return tnsr


def eigendecomposition_downsample(A, n_downsample, size=None, decimate=2, n_vec=16):
    # A = affinity matrix
    # n_downsample = number of downsampling operations (2 seems okay)
    # decimate = amount of decimation for each downsampling operation (set to 2)
    # size = size of the image corresponding to A

    # transform to scipy format
    if (isinstance(A, torch.sparse.FloatTensor)):
        A = torch_to_scipy_csr(A)

    if (size is None):
        size = A.shape

    A_down = A
    size_down = np.array(size)
    Cs = {}

    for di in range(n_downsample):
        (j, i) = np.unravel_index(range(A_down.shape[0]), size_down)
        do_keep = np.logical_and((i % decimate == 0), (j % decimate == 0))
        do_keep_idx = np.argwhere(do_keep).flatten()
        import pdb; pdb.set_trace() ## DEBUG ##
        B = A_down[:, do_keep_idx]
        C = 1 / (B.dot(np.ones((B.shape[1], 1))) + (np.finfo(float).eps))
        C = B * C
        A_down = C.T.dot(B)

        size_down = np.floor(size_down / 2).astype(int)
        Cs[di] = B

    lambda_, Q = sparse.linalg.eigsh(A_down, k=n_vec, which='LM')
    lambda_ = (2 ** -n_downsample) * lambda_

    # this part "upsamples" the eigen-vectors to original size
    for di in range(n_downsample - 1, -1, -1):
        Q = Cs[di].dot(Q)

    # convert Q to sparse matrix by adding "zero" eigen vectors on right
    # Q = sparse.hstack((Q, sparse.csr_matrix((Q.shape[0], A.shape[1] - Q.shape[1]))))
    # lambda_ = sparse.hstack((lambda_, sparse.csr_matrix((1, A.shape[1] - Q.shape[1]))))

    return lambda_, Q

def apply_rw_inf(A, alpha, f, n_downsample=3):
    """
    Compute final segmentation by applying an "infinite" amount of random walk
    steps
    """
    if (isinstance(A, torch.sparse.FloatTensor)):
        A = torch_to_scipy_csr(A)

    E = (sparse.eye(A.shape[0]) - alpha*A)
    E_lambda, E_Q = eigendecomposition_downsample(E, n_downsample)

    # eigen-decomposition: E = Q*lamba*Q^-1
    # from: https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Useful_facts_regarding_eigenvalues
    # eigen-vectors of A = eigen-vectors of A^-1
    # if lambda_i are eigen-values of A, then 1/lambda_i are eigen-values of A^(-1)
    # from: https://en.wikipedia.org/wiki/Orthogonal_matrix
    # A is symmetric => Q.T = Q^-1

    # compute E^-1 as Q^T * E_lambda^-1 * Q

def decimate_columns_sparse(A, decim):
    """
    Do column decimation as in https://arxiv.org/abs/1503.00848
    diag_: diagonal sparse matrix
    B: sparse matrix
    """
    values = A.data
    indices = A.indices()

    idx_decim = np.mod(indices[1, :] + 1, decim)
    new_cols = indices[1, idx_decim] // decim
    new_rows = indices[0, idx_decim]
    new_indices = np.stack((new_rows, new_cols))
    new_values = values[idx_decim]
    new_shape = (A.shape[0], A.shape[1] // decim)

    return sparse.csr_matrix(new_indices, new_values, new_shape)
    return sparse.csr_matrix((new_values, new_indices), shape=shape)


def sparse_diag_mm(diag_, B):
    """
    Make matrix multiplication: diag * B
    diag_: diagonal sparse matrix
    B: sparse matrix
    """

    B = B.coalesce()
    diag_ = diag_.coalesce()
    rows = B.indices()[0, :]
    diag_values = diag_.values()
    new_values = diag_values[rows]

    return torch.sparse.FloatTensor(B.indices(), new_values, B.shape)



def compute_distance(data, p=1):

    (row,
     col), pos, pseudo, x = data.edge_index, data.pos, data.edge_attr, data.x

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
                pos=torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), dim=1),
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

    data = Data(x=node_attr,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                shape=data_list[0].shape)
    return data


def make_sparse_pairwise(batch, p, radius, n_workers):

    n, c, w, h = batch.shape
    data_list = [batch[n_, c_, ...] for c_ in range(c) for n_ in range(n)]

    fun = partial(run_transforms, p=p, radius=radius)
    # if(n_workers > 1 or n*c > 1):
    if (n_workers > 1):
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
