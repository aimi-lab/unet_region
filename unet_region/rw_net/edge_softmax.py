from collections import namedtuple

from torch_geometric.data import Data, Batch
import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.utils import softmax, scatter_


class EdgeSoftmax(torch.nn.Module):
    r"""    Args:
        in_channels (int): Size of each input sample.
    """

    def __init__(self):
        super(EdgeSoftmax, self).__init__()
        self.eps = 1e-12

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index):
        return softmax(raw_edge_score, edge_index[1], raw_edge_score.size(0))

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index):
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index):
        return torch.sigmoid(raw_edge_score)

    def forward(self, batch):
        r"""Forward computation which computes the raw edge score, normalizes
        it
        """

        data_list = Batch.to_data_list(batch)
        data_list_out = []
        for data in data_list:
            new_edge_attr = softmax(data.edge_attr, data.edge_index[0])
            data.edge_attr = new_edge_attr
            data_list_out.append(data)
        batch = Batch.from_data_list(data_list_out)
        return batch
            
            
