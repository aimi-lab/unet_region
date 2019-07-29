import torch
from skimage.draw import circle
import numpy as np
import torch.nn.functional as F
from torch.nn import SmoothL1Loss
import matplotlib.pyplot as plt
import shapely.geometry as geom
from scipy import interpolate
from skimage import draw
from PIL import Image, ImageDraw, ImageMath

class LossDSAC(torch.nn.Module):
    def __init__(self):
        super(LossDSAC, self).__init__()

    def forward(self, edges, alpha, beta, kappa, snakes, target_mask,
                target_snake):
        
        M = edges.shape[-2]
        N = edges.shape[-1]

        grads_edges = torch.zeros(edges.shape, device=edges.device)
        grads_alpha = torch.zeros(edges.shape, device=edges.device)
        grads_beta = torch.zeros(edges.shape, device=edges.device)
        grads_kappa = torch.zeros(edges.shape, device=edges.device)

        intersection = []
        union = []
        iou = []
        area_gt = []
        area_snake = []

        for i in range(len(snakes)):
            # compute mask of inferred snake
            snake_mask = torch.zeros((M, N), device=edges.device)
            tgt_mask = target_mask[i, 0, ...]

            rr, cc = draw.polygon(
                snakes[i][:, 0],
                snakes[i][:, 1],
                shape=(M, N))
            snake_mask[rr, cc] = 1

            # compute first and second order derivatives
            d_y, d_y2 = derivatives_poly(snakes[i])
            d_y_tgt, d_y_tgt2 = derivatives_poly(target_snake[i])

            # -----------------
            # compute gradients
            # -----------------

            # data (edge) term
            grads_edges[i, 0, ...] += torch.from_numpy(draw_poly(snakes[i],
                                                1, [M, N], 4) - \
                                      draw_poly(
                                          target_snake[i],
                                          1, [M, N], 4)).to(edges.device)

            # alpha term
            grads_alpha[i, 0, ...] += (np.mean(d_y) - np.mean(d_y_tgt))

            # beta term
            grads_beta[i, 0, ...] += torch.from_numpy(draw_poly(snakes[i],
                                                d_y2, [M, N], 4) - \
                                      draw_poly(target_snake[i],
                                                d_y_tgt2,
                                                [M, N], 4)).to(beta.device)

            # kappa term
            grads_kappa[i, 0, ...] += tgt_mask - snake_mask

            tgt_mask = tgt_mask.cpu().numpy()
            snake_mask = snake_mask.cpu().numpy()
            intersection.append((tgt_mask + snake_mask) == 2)
            union.append((tgt_mask + snake_mask) >= 1)
            iou.append(intersection[-1].sum() / union[-1].sum())

            intersection[-1] = intersection[-1].sum() / intersection[-1].size
            union[-1] = union[-1].sum() / union[-1].size

            area_gt.append(np.sum(tgt_mask > 0))
            area_snake.append(np.sum(snake_mask > 0))

        # this is for monitoring/plotting
        data = {'iou': iou,
                'intersection': intersection,
                'union': union,
                'area_gt': area_gt,
                'area_snake': area_snake}

        gradients = {'edges': grads_edges,
                     'alpha': grads_alpha,
                     'beta': grads_beta,
                     'kappa': grads_kappa}

        return data, gradients


def derivatives_poly(poly):
    """
    :param poly: the Lx2 polygon array [u,v]
    :return: der1, der1, Lx2 derivatives arrays
    """
    u = poly[:, 0]
    v = poly[:, 1]
    L = len(u)

    der1_mat = -np.roll(np.eye(L), -1, axis=1) + \
               np.roll(np.eye(L), -1, axis=0)  # first order derivative, central difference
    der2_mat = np.roll(np.eye(L), -1, axis=0) + \
               np.roll(np.eye(L), -1, axis=1) - \
               2 * np.eye(L)  # second order derivative, central difference
    der1 = np.sqrt(np.power(np.matmul(der1_mat, u), 2) + \
                   np.power(np.matmul(der1_mat, v), 2))
    der2 = np.sqrt(np.power(np.matmul(der2_mat, u), 2) + \
                   np.power(np.matmul(der2_mat, v), 2))

    return der1, der2


def draw_poly(poly, values, im_shape, brush_size):
    """ 
    Returns a MxN (im_shape) array with values in the pixels crossed
    by the edges of the polygon (poly). total_points is the maximum number
    of pixels used for the linear interpolation.
    """

    u = poly[:, 0]
    v = poly[:, 1]
    b = np.round(brush_size / 2)
    image = Image.fromarray(np.zeros(im_shape))
    image2 = Image.fromarray(np.zeros(im_shape))
    d = ImageDraw.Draw(image)
    if type(values) is int:
        values = np.ones(np.shape(u)) * values
    for n in range(len(poly)):
        d.ellipse([(v[n] - b, u[n] - b), (v[n] + b, u[n] + b)], fill=values[n])
        image2 = ImageMath.eval("convert(max(a, b), 'F')", a=image, b=image2)

    return np.array(image2)
