import torch
import numpy as np
from skimage import segmentation, draw


def get_contour(Q, rc, n):

    c = [get_cart_coords(Q, u, rc) for u in torch.linspace(0, 2 * np.pi, n)]
    c = torch.cat(c, dim=1)
    c = c.to(Q.device)
    return c


def make_phi(Q, u):

    h = (Q.shape[1] - 1) // 2
    phi = torch.zeros_like(Q)
    phi[:, 0, :] = 1
    phi[:, 1:h + 1, :] = torch.tensor(
        [torch.cos(l * u) for l in torch.arange(1, h + 1, dtype=torch.float)])[..., None]
    phi[:, h + 1:, :] = torch.tensor([
        torch.sin(l * u)
        for l in torch.arange(h + 1, 2 * h + 1, dtype=torch.float)
    ])[..., None]

    return phi

def get_mask(Q, rc, n, shape):
    c = get_contour(Q, rc, n)
    p = [draw.polygon(c_[:, 1], c_[:, 0], shape=shape) for c_ in c]
    M = torch.zeros(Q.shape[0], *shape)

    for b, p_ in enumerate(p):
        M[b, p_[0], p_[1]] = 1

    return M

def get_cart_coords(Q, u, rc):

    phi = make_phi(Q, u)

    rho = torch.bmm(torch.transpose(Q, 1, 2), phi)

    v = torch.tensor([torch.cos(u), torch.sin(u)]).to(Q.device)[None, ..., None]
    v = torch.repeat_interleave(v, Q.shape[0], 0)

    r = rc + rho * v

    return torch.transpose(r, 1, 2)

def build_b(Q, u):

    phi = make_phi(Q, u)
    rho = torch.bmm(torch.transpose(Q, 1, 2), phi) 

    w = torch.tensor([-torch.sin(u), torch.cos(u)]).to(Q.device)[None, ..., None]
    w = torch.repeat_interleave(w, Q.shape[0], 0)

    B = -torch.bmm(torch.transpose(rho * phi, 1, 2), phi) * w


