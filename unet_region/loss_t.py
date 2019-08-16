import torch
from skimage.draw import circle
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


class LossT(torch.nn.Module):
    """
    Computes bce with heaviside
    """

    def __init__(self):
        """
        """
        super(LossT, self).__init__()
        self.eps = 1


    def forward(self, input, target):

        input = approx_heaviside(input, self.eps)
        input = torch.clamp(input, min=1e-7)
        log_input = torch.log(input)

        weight_pos = torch.sum(target) / target.numel()
        weight_neg = 1 - weight_pos

        weight_ = torch.tensor(
            (weight_pos,
             weight_neg), device=input.device)

        loss = F.nll_loss(torch.stack((input.view(-1),
                                            1 - input.view(-1)), dim=-1),
                               target.view(-1).long(),
                               weight=weight_)

        return loss


def approx_heaviside(s, eps):
    return 0.5 * (1 + (2 / math.pi) * torch.atan(s / eps))
