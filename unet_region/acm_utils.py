import numpy as np
import math
from scipy import ndimage
import skfmm
import torch
import matplotlib.pyplot as plt
from skimage import segmentation
import torch.nn.functional as F
from typing import Optional
import math
from scipy.stats import multivariate_normal
import tqdm


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
    if (contour.sum() == 0):
        contour[0, :] = 1
        contour[-1, :] = 1
        contour[:, 0] = 1
        contour[:, -1] = 1

    a_ls = np.ones(contour.shape)
    a_ls[contour] = -1

    sdf = skfmm.distance(a_ls)
    sdf += 0.5

    # make values outside positive
    sdf[a] = -sdf[a]

    if (thr is not None):
        if (return_contour):
            return np.clip(sdf, -thr, thr), contour
        else:
            return np.clip(sdf, -thr, thr)

    else:
        if (return_contour):
            return sdf, contour
        else:
            return sdf


def make_phi_spheres(center, r0, a, b, alpha, shape):
    x, y = torch.meshgrid([
        torch.arange(shape[1], dtype=torch.float),
        torch.arange(shape[0], dtype=torch.float)
    ])
    x = torch.stack([x - center_[1] for center_ in center])
    y = torch.stack([y - center_[0] for center_ in center])

    theta = torch.atan2(x, y)
    r = torch.stack([
        torch.stack([
            al * torch.sin((l + 1) * theta_ + alpha)
            for l, al in enumerate(a_)
        ]) for a_, theta_ in zip(a, theta)
    ])

    r += torch.stack([
        torch.stack([
            bl * torch.cos((l + 1) * theta_ + alpha)
            for l, bl in enumerate(b_)
        ]) for b_, theta_ in zip(b, theta)
    ])
    r = torch.sum(r, 1)
    r += r0

    phi = torch.norm(torch.stack((x, y)), dim=0) - r

    return phi


def approx_heaviside(s, eps):
    return 0.5 * (1 + (2 / math.pi) * torch.atan(s / eps))


def approx_dirac(s, eps):
    return (1 / (np.pi * eps)) * (1 / (1 + (s / eps)**2))

def grad_l1(a):
    grad = torch.zeros_like(a)
    grad[a > 0] = 1
    grad[a < 0] = -1
    return grad

def acwe_sphere(center,
                r0,
                a,
                b,
                alpha,
                data,
                step_size=0.001,
                step_size_phase=0.01,
                n_iter=100,
                eps=1,
                lambda1=1,
                lambda2=1,
                nu=0.1,
                decay=0.8,
                tol=10e-8):

    phi_history = []
    energies = []
    energies_in = []
    energies_out = []
    phivar = []
    energyvar = []

    batch_size = data.shape[0]
    w, h = data.shape[-2:]

    n_comps = a.shape[-1]

    dphi_dr0 = -1

    x, y = torch.meshgrid([
        torch.arange(w, dtype=torch.float),
        torch.arange(h, dtype=torch.float)
    ])
    x = torch.stack([x - center_[1] for center_ in center])
    y = torch.stack([y - center_[0] for center_ in center])

    theta = torch.atan2(x, y)

    dphi_da = torch.stack([
        torch.stack(
            [1 * torch.sin((l + 1) * theta_) for l, al in enumerate(a_)])
        for a_, theta_ in zip(a, theta)
    ])

    dphi_db = torch.stack([
        torch.stack(
            [-1 * torch.cos((l + 1) * theta_) for l, bl in enumerate(b_)])
        for b_, theta_ in zip(b, theta)
    ])

    for i in range(n_iter):
        if (i > 1):
            phivar_ = ((phi_new - phi)**2).sqrt().mean()
            energyvar_ = energies[-2] - energies[-1]
            energyvar.append(energyvar_)
            phivar.append(phivar_)
            print('[{}/{}] dE: {}, dphi: {}'.format(i + 1, n_iter, energyvar_,
                                                    phivar_))
            if (phivar_ < tol):
                print('hit tolerance')
                break
            phi = phi_new
        else:
            phi = make_phi_spheres(center, r0, a, b, alpha,
                                   data.shape[1:])

        phi_history.append(phi.clone().detach().cpu())

        # inside
        c1_nom = torch.sum(
            (data * (1 - approx_heaviside(phi, eps))).view(batch_size, -1),
            dim=-1)
        c1_denom = torch.sum(
            (1 - approx_heaviside(phi, eps)).view(batch_size, -1), dim=-1)
        c1 = c1_nom / c1_denom

        # outside
        c2_nom = torch.sum(
            (data * (approx_heaviside(phi, eps))).view(batch_size, -1), dim=-1)
        c2_denom = torch.sum((approx_heaviside(phi, eps)).view(batch_size, -1),
                             dim=-1)
        c2 = c2_nom / c2_denom

        term_1 = lambda1 * (data - c1)**2
        term_2 = lambda2 * (data - c2)**2
        energy_in = torch.sum((1 - approx_heaviside(phi, eps)) * term_1,
                              dim=(1, 2))
        energy_out = torch.sum((approx_heaviside(phi, eps)) * term_2,
                               dim=(1, 2))
        energy = energy_in + energy_out

        energies.append(energy.cpu().detach().numpy())
        energies_in.append(energy_in.cpu().detach().numpy())
        energies_out.append(energy_out.cpu().detach().numpy())

        dE_dphi = approx_dirac(phi, eps) * term_1 - \
            approx_dirac(phi, eps) * term_2

        # derivative wrt r0
        r0 = r0 + step_size * torch.sum(
            dE_dphi * dphi_dr0, dim=(1, 2))

        # derivative wrt a
        dE_da = torch.sum(torch.stack(n_comps * [dE_dphi], dim=1) * dphi_da, dim=(2, 3))
        dE_da += nu*grad_l1(a)
        a = a + step_size * dE_da

        # derivative wrt b
        dE_db = torch.sum(torch.stack(n_comps * [dE_dphi], dim=1) * dphi_db, dim=(2, 3))
        dE_db += nu*grad_l1(b)
        b = b + step_size * dE_db

        # derivative wrt alpha
        # if(i % 30 == 0):
        dphi_dalpha = torch.stack([
            torch.stack([
                al * torch.sin((l + 1) * theta_ + alpha_) - 
                bl * torch.cos((l + 1) * theta_ + alpha_)
                for l, (al, bl) in enumerate(zip(a_, b_))
            ]) for a_, b_, alpha_, theta_ in zip(a, b, alpha, theta)
        ])
        dphi_dalpha = torch.sum(dphi_dalpha, dim=1)
        dE_dalpha = torch.sum(dphi_dalpha * dE_dphi,
                            dim=(1, 2))
        alpha = alpha + step_size_phase * dE_dalpha

        # update step size
        step_size = step_size * decay
        step_size_phase = step_size_phase * decay

        phi_new = make_phi_spheres(center, r0, a, b, alpha, 
                                   data.shape[1:])

    return {
        'phi': phi_new,
        'phi_history': phi_history,
        'r0': r0,
        'a': a,
        'b': b,
        'alpha': alpha,
        'E': energies,
        'E_in': energies_in,
        'E_out': energies_out,
        'Evar': energyvar,
        'iters': i
    }


def make_init_ls_gaussian(x, y, shape, radius):

    g = make_2d_gauss(shape, radius, (y, x))

    g -= g.min()
    g /= g.max()
    g = 2 * (1 - g)
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

    if (deriv == 0):
        p[s <= 1] = (1 /
                     (2 * np.pi)**2) * (1 - torch.cos(2 * np.pi * s[s <= 1]))
        p[s >= 1] = (1 / 2) * (s[s >= 1] - 1)**2
    elif (deriv == 1):
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


def acm_ls(phi,
           V,
           m,
           dt,
           n_iters,
           vec_field=False,
           lambda_=torch.tensor(1),
           mu=torch.tensor(0.04)):

    phis = []
    for i in range(n_iters):
        dphi = grad(phi)
        dphi_norm = torch.norm(dphi, dim=0)

        # motion term.
        if (vec_field):
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

        dphi_t = attachment + lambda_ * smoothing + mu * regu
        # dphi_t = smoothing + attachment
        # dphi_t = attachment
        phi = phi + dt * dphi_t
        phis.append(phi)

    return phis


class RBFstore:
    def __init__(self,
                 center,
                 n_angles,
                 n_scales,
                 n_skews,
                 shape):
        self.center = center
        self.n_angles = n_angles
        self.n_scales = n_scales
        self.n_skews = n_skews
        self.shape = shape

        self.buffer_psi = self.build_psi()
        self.buffer_z = self.build_z()

    def __len__(self):

        return self.n_angles * self.n_scales * self.n_skews

    def build_z(self):
        skews_x, skews_y = torch.meshgrid([torch.arange(self.n_skews),
                                         torch.arange(self.n_skews)])
        skews_x = skews_x.flatten()
        skews_y = skews_y.flatten()
        idx = torch.empty(skews_x.shape + (2,))
        idx[..., 0] = skews_x
        idx[..., 1] = skews_y

        buffer = torch.stack([self.get_z(skews_x_idx, skews_y_idx)
                              for skews_x_idx, skews_y_idx in idx])

        return buffer

    def build_psi(self):
        angles, scales = torch.meshgrid([torch.arange(self.n_angles),
                                         torch.arange(self.n_scales)])
        angles = angles.flatten()
        scales = scales.flatten()
        idx = torch.empty(angles.shape + (2,))
        idx[..., 0] = angles
        idx[..., 1] = scales

        buffer = torch.stack([self.get_psi(angle_idx, scale_idx)
                              for angle_idx, scale_idx in idx])

        return buffer

    def get_z(self, skew_idx_x, skew_idx_y):
        skew = torch.tensor((1 / self.n_skews * skew_idx_x,
                             1 / self.n_skews * skew_idx_y))
        # apply skew function
        xc = torch.tensor((self.center[1], self.center[0])).float()
        x, y = torch.meshgrid([torch.arange(self.shape[1]), torch.arange(self.shape[0])])
        pos = torch.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        z = (1 / math.pi) * torch.atan((pos - xc) @ skew) + \
            (1 / 2)

        return z

    def get_psi(self, angle_idx, scale_idx):

        angle = math.pi / (self.n_angles) * angle_idx
        scale = np.max(self.shape) / (self.n_scales) * (scale_idx + 1)

        rot = torch.tensor([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
        cov = torch.tensor(np.diag((np.max(self.shape), scale))).float()
        cov = rot @ cov @ rot.t()
        rv = multivariate_normal(mean=(self.center[1], self.center[0]),
                                 cov=cov)
        x, y = torch.meshgrid([torch.arange(self.shape[1]), torch.arange(self.shape[0])])
        pos = torch.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y

        psi = torch.tensor(rv.pdf(pos)).float()
        psi = psi / psi.max()

        return psi

    def make_phi(self, r0, a):
        angles, scales, skews = torch.meshgrid([torch.arange(self.n_angles),
                                                torch.arange(self.n_scales),
                                                torch.arange(self.n_skews)])
        angles = angles.flatten()
        scales = scales.flatten()
        skews = skews.flatten()

        idx = torch.empty(angles.shape + (3,))
        idx[..., 0] = angles
        idx[..., 1] = scales
        idx[..., 2] = skews
        import pdb; pdb.set_trace() ## DEBUG ##
        r = torch.stack([a_ * self[idx[i, 0], idx[i, 1], idx[i, 2]]
                         for i, a_ in enumerate(a)])

        x, y = torch.meshgrid([
            torch.arange(self.center[0], dtype=torch.float),
            torch.arange(self.center[1], dtype=torch.float)
        ])
        x, y = torch.meshgrid([
            torch.arange(self.shape[1], dtype=torch.float),
            torch.arange(self.shape[0], dtype=torch.float)
        ])
        x = x - self.center[0]
        y = y - self.center[1]

        phi = torch.norm(torch.stack((x, y)), dim=0) - r


def make_phi_rbf(center, r0, a, shape):
    x, y = torch.meshgrid([
        torch.arange(shape[1], dtype=torch.float),
        torch.arange(shape[0], dtype=torch.float)
    ])
    x = torch.stack([x - center_[1] for center_ in center])
    y = torch.stack([y - center_[0] for center_ in center])

    theta = torch.atan2(x, y)
    r = torch.stack([
        torch.stack([
            al * torch.sin((l + 1) * theta_ + alpha)
            for l, al in enumerate(a_)
        ]) for a_, theta_ in zip(a, theta)
    ])

    r += torch.stack([
        torch.stack([
            bl * torch.cos((l + 1) * theta_ + alpha)
            for l, bl in enumerate(b_)
        ]) for b_, theta_ in zip(b, theta)
    ])
    r = torch.sum(r, 1)
    r += r0

    phi = torch.norm(torch.stack((x, y)), dim=0) - r

    return phi
