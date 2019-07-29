import numpy as np
from scipy import ndimage
import skfmm
import torch
import matplotlib.pyplot as plt
from skimage import segmentation
import torch.nn.functional as F
from typing import Optional
import math


def make_1d_gauss(length, std, x0):

    x = np.arange(length)
    y = np.exp(-0.5 * ((x - x0) / std)**2)

    return y / np.sum(y)


def make_2d_gauss(shape, std, center):
    """
    Make object prior (gaussians) on center
    """

    g = np.zeros(shape)
    g_x = make_1d_gauss(shape[1], std, center[1])
    g_x = np.tile(g_x, (shape[0], 1))
    g_y = make_1d_gauss(shape[0], std, center[0])
    g_y = np.tile(g_y.reshape(-1, 1), (1, shape[1]))

    g = g_x * g_y

    return g / np.sum(g)

def make_energy_ws(a):
    """
    Make energy for watershed
    
    """

    a_dist = np.ones(a.shape)
    contour = (segmentation.find_boundaries(a > 0))
    a_dist[contour] = -1

    e = skfmm.distance(a_dist)
    e[np.logical_not(a)] = 0

    return -e

def make_sdf(a, thr=None, return_contour=False):
    """
    Make signed distance array
    
    """
    contour = segmentation.find_boundaries(a)
    if(contour.sum() == 0):
        contour[0, :]  = 1
        contour[-1, :]  = 1
        contour[:, 0]  = 1
        contour[:, -1]  = 1

    a_ls = np.ones(contour.shape)
    a_ls[contour] = -1

    sdf = skfmm.distance(a_ls)
    sdf += 0.5

    # make values outside positive
    sdf[a] = -sdf[a]

    if (thr is not None):
        if(return_contour):
            return np.clip(sdf, -thr, thr), contour
        else:
            return np.clip(sdf, -thr, thr)

    else:
        if(return_contour):
            return sdf, contour
        else:
            return sdf

def make_init_ls_gaussian(x, y, shape, radius):

    g = make_2d_gauss(shape, radius, (y, x))

    g -= g.min()
    g /= g.max()
    g = 2*(1 - g)
    g -= 1
    # g = 10*(1 - g)
    # g -= 5
    return g


def div(fx, fy):

    fyx, fyy = grad(fy)
    fxx, fxy = grad(fx)

    return fxx + fyy

def double_well_potential(s, deriv=0):
    p = torch.zeros_like(s)

    if(deriv == 0):
        p[s <= 1] = (1 / (2 * np.pi)**2) * (1 - torch.cos(2 * np.pi * s[s <= 1]))
        p[s >= 1] = (1 / 2) * (s[s >= 1] - 1)**2
    elif(deriv == 1):
        p[s <= 1] = (1 / (2 * np.pi)) * (torch.sin(2 * np.pi * s[s <= 1]))
        p[s >= 1] = s[s >= 1] - 1
        
    return p

def curvature(f):
    fx_fy = grad(f)
    fx, fy = fx_fy
    norm = torch.norm(fx_fy, dim=0)
    Nx = fx / (norm + 1e-7)
    Ny = fy / (norm + 1e-7)
    return div(Nx, Ny)

def regularize(f):
    fx_fy = grad(f)
    fx, fy = fx_fy
    norm = torch.norm(fx_fy, dim=0)
    Nx = fx / (norm + 1e-7)
    Ny = fy / (norm + 1e-7)

    dw = double_well_potential(norm, deriv=1)
    # dw = 1

    return div(dw * Nx, dw * Ny)

def grad(a):
    a_pad = F.pad(a.unsqueeze(0), (1, 1, 1, 1), mode='constant')
    da_dx = (torch.roll(a_pad, -1, 1) - torch.roll(a_pad, 1, 1))
    da_dx = da_dx[:, 1:-1, 1:-1]
    # da_dy = (roll(a_pad, -1, 0) - roll(a_pad, 1, 0)) / 2
    da_dy = (torch.roll(a_pad, -1, 0) - torch.roll(a_pad, 1, 0))
    da_dy = da_dy[:, 1:-1, 1:-1]

    return torch.cat((da_dx, da_dy))

def acm_ls(phi, V, m, dt, n_iters,
           vec_field=False,
           lambda_=torch.tensor(1),
           mu=torch.tensor(0.04)):

    phis = []
    for i in range(n_iters):
        dphi = grad(phi)
        dphi_norm = torch.norm(dphi, dim=0)

        # motion term. 
        if(vec_field):
            attachment = -(V * dphi).sum(0)
        else:
            attachment = -V * dphi_norm

        # curvature term
        # div_ = div(dphi_over_norm[0], dphi_over_norm[1])
        smoothing = m * curvature(phi)
        # smoothing = torch.tensor(0)

        # additional regularization
        regu = regularize(phi)
        # regu = torch.tensor(0)

        # print('smoothing in [{}, {}]'.format(smoothing.min(),
        #                                      smoothing.max()))
        # print('attachment in [{}, {}]'.format(attachment.min(),
        #                                       attachment.max()))

        dphi_t = attachment + lambda_*smoothing + mu*regu
        # dphi_t = smoothing + attachment
        # dphi_t = attachment
        phi = phi + dt * dphi_t
        phis.append(phi)

    return phis
