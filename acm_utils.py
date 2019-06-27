import numpy as np
from scipy import ndimage
import skfmm
import matplotlib.pyplot as plt
from skimage import segmentation


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

def make_sdf(a, thr=None):
    """
    Make signed distance array
    
    """

    a_ls = np.ones(a.shape)
    a_ls[a] = -1

    sdf = skfmm.distance(a_ls)

    if (thr is not None):
        return np.sign(sdf) * np.min(
            np.stack((np.abs(sdf), thr * np.ones_like(a))), axis=0)

    else:
        return sdf


def make_init_ls_gaussian(x, y, shape, radius):

    g = make_2d_gauss(shape, radius, (y, x))

    g -= g.min()
    g /= g.max()
    g = 1 - g
    g -= 0.5
    return g


def div(fx, fy):

    fyy, fyx = np.gradient(fy)
    fxy, fxx = np.gradient(fx)

    return fxx + fyy


def double_well_potential(s):
    s[s <= 1] = (1 / (2 * np.pi)**2) * (1 - np.cos(2 * np.pi * s[s <= 1]))
    s[s >= 1] = (1 / 2) * (s[s >= 1] - 1)**2
    return s

def curvature(f):
    fy, fx = np.gradient(f)
    norm = np.sqrt(fx**2 + fy**2)
    Nx = fx / (norm + 1e-8)
    Ny = fy / (norm + 1e-8)
    return div(Nx, Ny)

def regularize(f):
    fy, fx = np.gradient(f)
    norm = np.sqrt(fx**2 + fy**2)
    Nx = fx / (norm + 1e-8)
    Ny = fy / (norm + 1e-8)

    dw = double_well_potential(norm)
    # dw = 1

    return div(dw * Nx, dw * Ny)
    

def acm_ls(phi, V, m, dt, n_iters,
           lambda_=1,
           mu=0.04):

    phis = []
    for i in range(n_iters):
        dphi = np.array(np.gradient(phi))
        # dphi_norm = np.linalg.norm(dphi, axis=0)
        dphi_norm = np.sum(np.abs(dphi), axis=0)

        dphi_over_norm = dphi / (dphi_norm + 1e-2)

        # motion term. This should be a vector field to get -<V, dphi>
        attachment = -V * dphi_norm

        # curvature term
        # div_ = div(dphi_over_norm[0], dphi_over_norm[1])
        smoothing = m * curvature(phi)

        # additional regularization
        regu = regularize(phi)

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
