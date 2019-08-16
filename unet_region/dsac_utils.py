import time
import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage import segmentation
from scipy.interpolate import RectBivariateSpline
import math
from scipy.ndimage import measurements
from scipy import interpolate


def make_init_snake(radius, shape, L):

    s = np.linspace(0, 2 * np.pi, L)
    init_u = shape / 2 + radius * np.cos(s)
    init_v = shape / 2 + radius * np.sin(s)
    init_u = init_u.reshape([L, 1])
    init_v = init_v.reshape([L, 1])
    init_snake = np.array([init_u[:, 0], init_v[:, 0]]).T

    return init_snake


def acm_inference(map_e, map_a, map_b, map_k, init_snake, gamma, delta_s,
                  max_px_move,
                  n_iter, verbose=False):

    snakes = []
    for i in range(map_e.shape[0]):

        # # get edge map of data map
        # edge_map = gaussian(map_e[i, 0, ...], sigma)
        # edge_map[0, :] = edge_map[1, :]
        # edge_map[-1, :] = edge_map[-2, :]
        # edge_map[:, 0] = edge_map[:, 1]
        # edge_map[:, -1] = edge_map[:, -2]


        # optimize
        snake_ = active_contour_steps(
            map_e[i, 0, ...],
            init_snake,
            map_a[i, 0, ...],
            map_b[i, 0, ...],
            map_k[i, 0, ...],
            delta_s,
            gamma=gamma,
            max_px_move=max_px_move,
            max_iterations=n_iter,
            verbose=verbose)
        snakes.append(snake_)

    return snakes



def active_contour_steps(image, snake, alpha, beta, kappa,
                         delta_s,
                         w_line=0, w_edge=1, gamma=0.01,
                         max_px_move=1.0,
                         max_iterations=2500,
                         convergence=0.1,
                         verbose=False):
    """Active contour model.

    Active contours by fitting snakes to features of images. Supports single
    and multichannel 2D images. Snakes can be periodic (for segmentation) or
    have fixed and/or free ends.
    The output snake has the same length as the input boundary.
    As the number of points is constant, make sure that the initial snake
    has enough points to capture the details of the final contour.

    Parameters
    ----------
    image : (N, M) or (N, M, 3) ndarray
        Input image.
    snake : (N, 2) ndarray
        Initial snake coordinates. For periodic boundary conditions, endpoints
        must not be duplicated.
    alpha : (N, M)  ndarray (float)
         Higher values makes snake contract faster.
    beta : (N, M)  ndarray (float)
        Higher values makes snake smoother.
    kappa : (N, M)  ndarray (float)
        Balloon term.
    w_line : float, optional
        Controls attraction to brightness. Use negative values to attract toward
        dark regions.
    w_edge : float, optional
        Controls attraction to edges. Use negative values to repel snake from
        edges.
    gamma : float, optional
        Explicit time stepping parameter.
    max_px_move : float, optional
        Maximum pixel distance to move per iteration.
    max_iterations : int, optional
        Maximum iterations to optimize snake shape.
    convergence: float, optional
        Convergence criteria.

    Returns
    -------
    snake : (N, 2) ndarray
        Optimised snake, same shape as input parameter.

    """
    max_iterations = int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations should be >0.")
    convergence_order = 10
    img = img_as_float(image)

    # Interpolate for smoothness:
    intp = RectBivariateSpline(np.arange(img.shape[1]),
                               np.arange(img.shape[0]),
                               img.T, kx=2, ky=2, s=0)

    start = time.time()

    x, y = snake[:, 0].astype(np.float), snake[:, 1].astype(np.float)
    x_start = x.copy()
    y_start = y.copy()

    # check if snake is closed
    is_closed = False
    if((x[0] == x[-1]) and (y[0] == y[-1])):
        is_closed = True
        x = x[:-1]
        y = y[:-1]
        
    dx, dy = np.zeros(x.shape), np.zeros(x.shape)
    n = len(x)
    xsave = np.empty((convergence_order, n))
    ysave = np.empty((convergence_order, n))

    # Build snake shape matrix for Euler equation
    alpha_snake = alpha[x.round().astype(int), y.round().astype(int)]
    a_diag = np.diag(alpha_snake)
    A = np.roll(a_diag, -1, axis=0) + \
        np.roll(a_diag, -1, axis=1) - \
        (a_diag + np.diag(np.roll(alpha_snake, -1)))
    A = -A
    
    beta_snake = beta[x.round().astype(int), y.round().astype(int)]
    b_diag = np.diag(beta_snake)
    Bd0 = np.diag(np.roll(beta_snake, 1)) + \
        4*b_diag + \
        np.diag(np.roll(beta_snake, -1))
    Bd1 = -2 * np.roll(np.diag(np.roll(beta_snake, -1) + beta_snake), -1, axis=0)
    Bdm1 = -2 * np.roll(np.diag(np.roll(beta_snake, 1) + beta_snake), -1, axis=1)
    Bd2 = np.roll(np.diag(np.roll(beta_snake, -1)), -2, axis=0) 
    Bdm2 = np.roll(np.diag(np.roll(beta_snake, 1)), 2, axis=0)
    B = Bd0 + Bd1 + Bdm1 + Bd2 + Bdm2
        
    # make matrix that will be inverted
    M_internal = np.eye(n) + 2*gamma*(A/delta_s + B/(delta_s**2))
    inv = np.linalg.inv(M_internal)

    # Explicit time stepping for image energy minimization:
    for i in range(max_iterations):
        # find derivative of "data" external energy function
        fx = intp(x, y, dx=1, grid=False)
        fy = intp(x, y, dy=1, grid=False)

        # ----------------
        # Get kappa values between nodes (balloon term)
        # ----------------
        kappa_collection = kappa[fx.round().astype(int),
                                 fy.round().astype(int)]

        # make snake vectors at x(s-1), x(s+1)
        xp1 = np.roll(x, -1)
        xm1 = np.roll(x, 1)

        # make snake vectors at v_(s-1), v_(s+1)
        yp1 = np.roll(y, -1)
        ym1 = np.roll(y, 1)

        # Get the derivative of the balloon energy
        h = np.arange(1, n + 1)
        int_ends_x_prev = xm1 - x
        int_ends_x_next = xp1 - x
        int_ends_y_prev = ym1 - y
        int_ends_y_next = yp1 - y

        # derivatives point in the normal (inside) direction when all k-values are equal
        s = 10
        dEb_dx = (int_ends_x_next / s**2) * np.sum(
            h * kappa_collection)
        dEb_dx += (int_ends_x_prev / s**2) * np.sum(
            h * kappa_collection)

        dEb_dy = (int_ends_y_next / s**2) * np.sum(
            h * kappa_collection)
        dEb_dy += (int_ends_y_prev / s**2) * np.sum(
            h * kappa_collection)

        # Movements are capped to max_px_move per iteration:
        dx = fx - dEb_dx
        dy = fy - dEb_dy
        norms = np.linalg.norm(np.array((dx, dy)).T, axis=1)
        dx = max_px_move * (dx / norms.max())
        dy = max_px_move * (dy / norms.max())

        x = inv @ (x + gamma*dx)
        y = inv @ (y + gamma*dy)

        # coordinates are capped to image frame
        x = np.clip(x, a_max=img.shape[1] - 1, a_min=0)
        y = np.clip(y, a_max=img.shape[0] - 1, a_min=0)

        # if(i > 20):
        #     plt.plot(x_start[:-1], y_start[:-1], 'bo-', label='(x, y)')
        #     plt.quiver(x, y,
        #                dx, dy,
        #                alpha=0.2)
        #     angles = [clockwiseangle((142, 131), (x_, y_))
        #             for x_, y_ in zip(x, y)]
        #     for j, a in enumerate(angles):
        #         plt.annotate('{}: {:.1f}'.format(j, a*180/np.pi), (x[j], y[j]))
        #     plt.plot((x + gamma*dx), (y + gamma*dy), 'ro-', label='(x_new, y_new)')
        #     plt.suptitle('iter: {}'.format(i))
        #     plt.grid(True)
        #     plt.axes().set_aspect('equal', 'datalim')
        #     plt.show()

        
        # Convergence criteria needs to compare to a number of previous
        # configurations since oscillations can occur.
        # j = i % (convergence_order+1)
        # if j < convergence_order:
        #     xsave[j, :] = x
        #     ysave[j, :] = y
        # else:
        #     dist = np.min(np.max(np.abs(xsave-x[None, :]) +
        #                          np.abs(ysave-y[None, :]), 1))
        #     if dist < convergence:
        #         break

    end = time.time()
    if(verbose):
        print('finished in {} iterations in {} s'.format(i+1, end-start))

    if(is_closed):
        x = np.concatenate((x, x[:1]))
        y = np.concatenate((y, y[:1]))

    return np.array([x, y]).T


def clockwiseangle(origin, point):

    refvec = (1, 0)

    # Vector between point and the origin: v = p - o
    vector = [point[0] - origin[0], point[1] - origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0] / lenvector, vector[1] / lenvector]
    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[
        1]  # x1*x2 + y1*y2
    diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[
        1]  # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2 * math.pi + angle
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle


def make_spline_contour(truth, L, s, k, per, ds_contour_rate=0.1):


    truth_ = np.pad(truth, ((1, ), (1, )), mode='constant',
                    constant_values=False)
    contour = segmentation.find_boundaries(truth_, mode='thick')
    y_c, x_c = measurements.center_of_mass(truth)
    y, x = np.where(contour)
    x -= 1
    y -= 1

    inds = sorted(np.random.choice(x.shape[0],
                                   size=int(ds_contour_rate*x.shape[0]),
                                   replace=False))
    x = x[inds].tolist()
    y = y[inds].tolist()

    pts = [(x, y) for x, y in zip(x, y)]

    sort_fn = lambda pt: clockwiseangle((x_c, y_c), pt)

    pts = sorted(pts, key=sort_fn)

    pts = np.array(pts)
    x, y = pts[:, 0], pts[:, 1]
    nodes = np.zeros([L, 2])
    [tck, u] = interpolate.splprep([x, y], s=2, k=1)
    [nodes[:, 0], nodes[:, 1]] = interpolate.splev(np.linspace(0, 1, L), tck)

    return nodes
