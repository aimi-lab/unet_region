import torch
from skimage.draw import circle
import numpy as np
import torch.nn.functional as F
from torch.nn import MSELoss
import matplotlib.pyplot as plt
from scipy import interpolate
from skimage import draw


class LossDirect(torch.nn.Module):
    """
    Computes MSE in the angular domain
    """

    def __init__(self):
        """
        """
        super(LossDirect, self).__init__()


    def forward(self, input, target):

        norm_in = (torch.norm(input, dim=1) + 1e-7).unsqueeze(1)
        input = input / torch.cat((norm_in, norm_in), dim=1)

        dot = (input * target).sum(1)
        acos = torch.acos(dot)
        acos2 = acos**2
        loss = acos2.mean()

        return loss
