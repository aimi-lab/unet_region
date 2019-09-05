from torch_geometric.data import Data, Batch
import torch
import torch.nn.functional as F
from unet_region.rw_net.rw_utils import graph_to_tensors


class RandomWalk(torch.nn.Module):
    r"""

    Args:
    """
    def __init__(self, alpha=0.5, mode='single'):
        super(RandomWalk, self).__init__()
        assert (mode == 'single'
                or mode == 'all'), 'mode should be single or all'
        self.mode = mode
        self.alpha = alpha

    def forward(self, batch_aff, batch_activ):
        """
        batch_activ is a tensor of shape (N,C,W,H) with C the class index (0: bg, 1: fg)
        batch_aff are graphs where edge attributes are probability of being neighbors
        """
        ba_n, ba_c, ba_w, ba_h = batch_activ.shape
        batch_activ = batch_activ.reshape(ba_n, ba_c, ba_w * ba_h)

        # get for each batch element the affinity matrix A
        batch_aff = Batch.to_data_list(batch_aff)
        batch_aff = [graph_to_tensors(b) for b in batch_aff]

        # atm, no batch matmul with sparse tensors... iterate on batch elements
        y_tilda = []
        for A, f in zip(batch_aff, batch_activ):
            y_tilda_ = A.mm(f.t())
            y_tilda.append(y_tilda_.t().reshape(ba_c, ba_h, ba_w))

        return torch.stack(y_tilda)
