import torch
from skimage.draw import circle
import numpy as np
import torch.nn.functional as F
from torch.nn import SmoothL1Loss
import matplotlib.pyplot as plt
import shapely.geometry as geom
from scipy import interpolate
from skimage import draw


class LossSplines(torch.nn.Module):
    """
    Computes BCE on a circular region defined by radius_rel
    """

    def __init__(self):
        """
        """
        super(LossSplines, self).__init__()
        self.loss = SmoothL1Loss()


    def forward(self, input, target):

        import pdb; pdb.set_trace()
        shape = target.shape[-2:]
        nodes = [get_nodes(input[b, ...].detach().numpy(), shape)
                 for b in range(input.shape[0])]
        contours = [make_contour(n) for n in nodes]
        masks = torch.tensor([make_mask(c, shape)
                              for c in contours]).type(torch.long)

        target = target.type(torch.long)
        target = target.squeeze(dim=1)

        loss = self.loss(masks, target)

        return loss

def get_nodes(vals, shape):
    n_quadrants = vals.shape[0]

    # Define the arc (presumably ezdxf uses a similar convention)
    centerx, centery = 0, 0
    radius = 1
    numsegments = 1000
    start_angles = np.linspace(0,
                            360 * ((n_quadrants -1) / n_quadrants),
                            n_quadrants)
    inc_angle = 360 / n_quadrants

    pts_cart = []
    for p, start_angle in zip(vals, start_angles):
        rho = p
        theta = np.radians(start_angle + inc_angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        x, y = rho, 0
        p = np.array((x, y))[..., np.newaxis]
        p = np.dot(R, p)
        pts_cart.append(p)
    
    pts_cart = np.array(pts_cart)[... ,0]
    pts_cart[:, 0] = ((pts_cart[:, 0] + 1) / 2) * shape[1]
    pts_cart[:, 1] = ((-pts_cart[:, 1] + 1) / 2) * shape[0]

    return (pts_cart[:, 1], pts_cart[:, 0])

def make_contour(pts):
    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    pts_ = np.array(pts)
    pts_ = np.concatenate((pts_, pts_[:, 0][..., np.newaxis]), axis=1)
    tck, u = interpolate.splprep([pts_[0, :], pts_[1, :]],
                                s=0,
                                per=True)

    # evaluate the spline fits for 1000 evenly spaced distance values
    r, c = interpolate.splev(np.linspace(0, 1, 1000), tck)

    return r, c

def make_mask(pts, shape):

    rr, cc = draw.polygon(pts[0], pts[1], shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rr, cc] = 1

    return mask


    return mask

