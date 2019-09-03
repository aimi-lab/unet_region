from torch_geometric.data import Data, Batch
import torch
import torch.nn.functional as F


class AffinityBranch(torch.nn.Module):
    r"""
    Applies / learns filters on each edge. Applies exp to get similarity graph

    Args:
        in_channels (int): Size of each input sample.
    """

    def __init__(self, in_channels, out_channels):
        super(AffinityBranch, self).__init__()
        self.eps = 1e-12
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, batch):

        batch = Batch.to_data_list(batch)
        batch_size = len(batch)
        n_chans = batch[0].x.shape[-1]
        edge_attrs = torch.stack([d.edge_attr.t() for d in batch])
        edge_attrs_out = self.conv(edge_attrs)

        edge_attrs_out = torch.exp(-edge_attrs_out)

        # put new attributes in graphs
        for i in range(batch_size):
            batch[i].edge_attr = edge_attrs_out[i, ...].t()

        return Batch.from_data_list(batch)
            
        # get sparse adjacency matrices
        # adj = [make_sparse_adj(d) for d in data_list]
