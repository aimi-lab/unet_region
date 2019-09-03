from torch_geometric.data import Data, Batch
import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.utils import softmax, scatter_


class DummySum(torch.nn.Module):
    r"""    Args:
        in_channels (int): Size of each input sample.
    """

    def __init__(self):
        super(DummySum, self).__init__()

    def forward(self, batch):
        data_list = Batch.to_data_list(batch)
        data_list = [Data(x=d.x,
                          edge_index=d.edge_index,
                          edge_attr=torch.sum(d.edge_attr, dim=1)[..., None],
                          pos=d.pos,
                          shape=d.shape)
                     for d in data_list]
        batch = Batch.from_data_list(data_list)
        return batch
