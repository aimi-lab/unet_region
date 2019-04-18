import torch
from skimage.draw import circle
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


class BCERegionLoss(torch.nn.Module):
    """
    Computes BCE on a circular region defined by radius_rel
    """

    def __init__(self, region_rel_size, size):
        super(BCERegionLoss,self).__init__()

        self.size = int(region_rel_size*size)
        self.size += 0 if self.size % 2 else 1


    def forward(self, input, target):

        size_in = input.shape[-1]
        input = input[...,
                      size_in//2 - self.size//2: size_in//2 + self.size//2 + 1,
                      size_in//2 - self.size//2: size_in//2 + self.size//2 + 1]
        target = target[...,
                        size_in//2 - self.size//2: size_in//2 + self.size//2 + 1,
                        size_in//2 - self.size//2: size_in//2 + self.size//2 + 1]
        return F.binary_cross_entropy_with_logits(input,
                                                  target)
