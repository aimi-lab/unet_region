import networkx as nx
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import ndimage as ndi
from scipy import sparse
import math
from skimage import measure, segmentation, util, color, draw
import matplotlib.pyplot as plt


def _add_edge_filter(labels, graph, im):
    """Create edge in `graph` between central element of `values` and the rest.

    Add an edge between the middle element in `values` and
    all other elements of `values` into `graph`.  ``values[len(values) // 2]``
    is expected to be the central value of the footprint used.

    Parameters
    ----------
    values : array
        The array to process.
    graph : RAG
        The graph to add edges in.

    Returns
    -------
    0 : float
        Always returns 0. The return value is required so that `generic_filter`
        can put it in the output array, but it is ignored by this filter.
    """
    labels = labels.astype(int)
    center = labels[len(labels) // 2]
    for label in labels:
        if not graph.has_edge(center, label):
            graph.add_edge(center, label, weight=np.abs(im[center] - im[label]))
    return 0.


class RAG(nx.Graph):

    """
    The Region Adjacency Graph (RAG) of an image, subclasses
    `networx.Graph <http://networkx.github.io/documentation/latest/reference/classes/graph.html>`_

    Parameters
    ----------
    label_image : array of int
        An initial segmentation, with each region labeled as a different
        integer. Every unique value in ``label_image`` will correspond to
        a node in the graph.
    connectivity : int in {1, ..., ``label_image.ndim``}, optional
        The connectivity between pixels in ``label_image``. For a 2D image,
        a connectivity of 1 corresponds to immediate neighbors up, down,
        left, and right, while a connectivity of 2 also includes diagonal
        neighbors. See `scipy.ndimage.generate_binary_structure`.
    data : networkx Graph specification, optional
        Initial or additional edges to pass to the NetworkX Graph
        constructor. See `networkx.Graph`. Valid edge specifications
        include edge list (list of tuples), NumPy arrays, and SciPy
        sparse matrices.
    **attr : keyword arguments, optional
        Additional attributes to add to the graph.
    """

    def __init__(self, shape, image, radius=1, data=None, **attr):

        super(RAG, self).__init__(data, **attr)
        if self.number_of_nodes() == 0:
            self.max_id = 0
        else:
            self.max_id = max(self.nodes())

        self.label_image = np.arange(np.prod(shape)).reshape(shape)
        w = 2*radius + 1
        struct = np.zeros((w, w), dtype=bool)
        rr, cc = draw.circle(w // 2, w // 2, radius, shape=(w, w))
        struct[rr, cc] = True
        # In the next ``ndi.generic_filter`` function, the kwarg
        # ``output`` is used to provide a strided array with a single
        # 64-bit floating point number, to which the function repeatedly
        # writes. This is done because even if we don't care about the
        # output, without this, a float array of the same shape as the
        # input image will be created and that could be expensive in
        # memory consumption.
        ndi.generic_filter(
            self.label_image,
            function=_add_edge_filter,
            footprint=struct,
            mode='nearest',
            output=as_strided(np.empty((1,), dtype=np.float_),
                                shape=shape,
                                strides=((0,) * self.label_image.ndim)),
            extra_arguments=(self, image.flatten(),))

    def draw(self):
        
        w, h = self.label_image.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = x.flatten()
        y = y.flatten()
        pos = {i: (x_, y_) for i, x_, y_ in zip(range(w*h), x, y)}
        nx.draw(self, pos, with_labels=True)

    def make_dist_adjacency(self, im):

        edges = list(self.edges)
        im_flat = im.flatten()
        import pdb; pdb.set_trace() ## DEBUG ##
        dists = {e: np.abs(im_flat[e[0]] - im_flat[e[1]])
                 for e in edges}
        nx.set_edge_attributes(self, dists, name='dist')
        A = nx.adjacency_matrix(self, weight='dist')
        return A
            
