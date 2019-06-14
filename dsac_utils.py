import time
import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from skimage.filters import sobel, gaussian
from skimage.util import img_as_float
from scipy.interpolate import RectBivariateSpline


def make_init_snake(radius, shape, L):

    s = np.linspace(0, 2 * np.pi, L)
    init_u = shape / 2 + radius * np.cos(s)
    init_v = shape / 2 + radius * np.sin(s)
    init_u = init_u.reshape([L, 1])
    init_v = init_v.reshape([L, 1])
    init_snake = np.array([init_u[:, 0], init_v[:, 0]]).T

    return init_snake


def acm_inference(map_e, map_a, map_b, map_k, init_snake, gamma, max_px_move,
                  delta_s, n_iter, sigma):

    snake_hist = []
    for i in range(map_e.shape[0]):

        # get edge map of data map
        edge_map = gaussian(map_e[i, 0, ...], sigma)
        edge_map[0, :] = edge_map[1, :]
        edge_map[-1, :] = edge_map[-2, :]
        edge_map[:, 0] = edge_map[:, 1]
        edge_map[:, -1] = edge_map[:, -2]

        snake_hist.append([])

        # optimize
        snake_hist_ = active_contour_steps(
            edge_map,
            init_snake,
            alpha=map_a[i, 0, ...],
            beta=map_b[i, 0, ...],
            kappa=map_k[i, 0, ...],
            gamma=gamma,
            max_px_move=max_px_move,
            max_iterations=n_iter)
        snake_hist[-1].append(snake_hist_)

    return snake_hist



def active_contour_steps(image, snake, alpha, beta, kappa,
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
    valid_bcs = ['periodic', 'free', 'fixed', 'free-fixed',
                 'fixed-free', 'fixed-fixed', 'free-free']
    img = img_as_float(image)
    RGB = img.ndim == 3

    # Find edges using sobel:
    if w_edge != 0:
        if RGB:
            edge = [sobel(img[:, :, 0]), sobel(img[:, :, 1]),
                    sobel(img[:, :, 2])]
        else:
            edge = [sobel(img)]
        for i in range(3 if RGB else 1):
            edge[i][0, :] = edge[i][1, :]
            edge[i][-1, :] = edge[i][-2, :]
            edge[i][:, 0] = edge[i][:, 1]
            edge[i][:, -1] = edge[i][:, -2]
    else:
        edge = [0]

    # Superimpose intensity and edge images:
    if RGB:
        img = w_line*np.sum(img, axis=2) \
            + w_edge*sum(edge)
    else:
        img = w_line*img + w_edge*edge[0]

    # Interpolate for smoothness:
    intp = RectBivariateSpline(np.arange(img.shape[1]),
                               np.arange(img.shape[0]),
                               img.T, kx=2, ky=2, s=0)

    start = time.time()

    x, y = snake[:, 0].astype(np.float), snake[:, 1].astype(np.float)
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
    M_internal = gamma * np.eye(n) + (A + B )
    inv = np.linalg.inv(M_internal)

    # Explicit time stepping for image energy minimization:
    for i in range(max_iterations):
        # interpolate
        fx = intp(x, y, dx=1, grid=False)
        fy = intp(x, y, dy=1, grid=False)

        # ----------------
        # Get kappa values between nodes (balloon term)
        # ----------------
        # make snake vectors at x(s-1), x(s+1)
        xp1 = np.roll(x, 1)
        xm1 = np.roll(x, -1)

        # make snake vectors at v_(s-1), v_(s+1)
        yp1 = np.roll(y, 1)
        ym1 = np.roll(y, -1)

        kappa_collection = kappa[fx.round().astype(int),
                                 fy.round().astype(int)]

        # Get the derivative of the balloon energy
        h = np.arange(1, n + 1)
        int_ends_x_next = xm1 - x
        int_ends_x_prev = xp1 - x
        int_ends_y_next = ym1 - y
        int_ends_y_prev = yp1 - y

        # contribution from the i+1 triangles
        dEext_dx = (int_ends_y_next / n**2) * np.sum(
            h * kappa_collection)
        dEext_dy = (int_ends_y_prev / n**2) * np.sum(
            h * kappa_collection)

        # contribution from the i+1 triangles
        dEext_dx += (int_ends_x_next / n**2) * np.sum(
            h * kappa_collection)
        dEext_dy += (int_ends_x_prev / n**2) * np.sum(
            h * kappa_collection)

        xn = inv @ (gamma*x + fx)
        yn = inv @ (gamma*y + fy)

        # Movements are capped to max_px_move per iteration:
        dx = max_px_move*np.tanh(xn-x + dEext_dx)
        dy = max_px_move*np.tanh(yn-y + dEext_dy)
        x += dx
        y += dy

        # coordinates are capped to image frame
        x = np.clip(x, a_max=img.shape[1] - 1, a_min=0)
        y = np.clip(y, a_max=img.shape[0] - 1, a_min=0)
        
        # Convergence criteria needs to compare to a number of previous
        # configurations since oscillations can occur.
        j = i % (convergence_order+1)
        if j < convergence_order:
            xsave[j, :] = x
            ysave[j, :] = y
        else:
            dist = np.min(np.max(np.abs(xsave-x[None, :]) +
                                 np.abs(ysave-y[None, :]), 1))
            if dist < convergence:
                break

    end = time.time()
    print('finished in {} iterations in {} s'.format(i+1, end-start))
    return np.array([x, y]).T

def active_contour_steps_old(edge_map,
                         du,
                         dv,
                         snake,
                         alpha,
                         beta,
                         kappa,
                         gamma,
                         max_px_move,
                         delta_s,
                         n_iter=100):


    snake_u, snake_v = snake[:, 0].astype(np.float), snake[:, 1].astype(np.float)

    L = snake_u.size
    M = edge_map.shape[0]
    N = edge_map.shape[1]
    u = snake_u
    v = snake_v

    # ----------------
    # Make matrices A and B as in eq. 2 and 3
    # ----------------
    fu = edge_map[u.round().astype(int), v.round().astype(int)]
    fv = edge_map[u.round().astype(int), v.round().astype(int)]
    a = alpha[0, u.round().astype(int), v.round().astype(int)].squeeze()
    b = beta[0, u.round().astype(int), v.round().astype(int)].squeeze()

    a_diag = np.diag(a)
    A = np.roll(a_diag, -1, axis=0) + \
        np.roll(a_diag, -1, axis=1) - \
        (a_diag + np.diag(np.roll(a, -1)))
    A = -A
    
    b_diag = np.diag(b)
    Bd0 = np.diag(np.roll(b, 1)) + 4*b_diag + np.diag(np.roll(b, -1))
    Bd1 = -2 * np.roll(np.diag(np.roll(b, -1) + b), -1, axis=0)
    Bdm1 = -2 * np.roll(np.diag(np.roll(b, 1) + b), -1, axis=1)
    Bd2 = np.roll(np.diag(np.roll(b, -1)), -2, axis=0) 
    Bdm2 = np.roll(np.diag(np.roll(b, 1)), 2, axis=0)
    B = Bd0 + Bd1 + Bdm1 + Bd2 + Bdm2
        
    # make matrix that will be inverted
    M_internal = gamma * np.eye(L) + (A / delta_s + B / (delta_s * delta_s))
    inv = np.linalg.inv(M_internal)

    # spline interpolation object
    intp = RectBivariateSpline(np.arange(edge_map.shape[1]),
                               np.arange(edge_map.shape[0]),
                               edge_map.T, kx=2, ky=2, s=0)

    for i in range(n_iter):
        # ----------------
        # Get kappa values between nodes (balloon term)
        # ----------------

        # make snake vectors at u_(s-1), u_(s+1)
        snake_up1 = np.concatenate((snake_u[-1:], snake_u[0:-1]), 0)
        snake_um1 = np.concatenate([snake_u[1:], snake_u[0:1]], 0)

        # make snake vectors at v_(s-1), v_(s+1)
        snake_vp1 = np.concatenate((snake_v[-1:], snake_v[0:-1]), 0)
        snake_vm1 = np.concatenate([snake_v[1:], snake_v[0:1]], 0)

        # Linear interpolation on each segment of snake
        u_interps = intp(u.ravel(), v.ravel(), dx=1, grid=False)
        v_interps = intp(u.ravel(), v.ravel(), dy=1, grid=False)

        kappa_collection = kappa[0,
                                 u_interps.round().astype(int),
                                 v_interps.round().astype(int)]

        # Get the derivative of the balloon energy
        s = kappa_collection.shape[0]
        h = np.arange(1, s + 1)
        int_ends_u_next = snake_um1 - snake_u
        int_ends_u_prev = snake_up1 - snake_u
        int_ends_v_next = snake_vm1 - snake_v
        int_ends_v_prev = snake_vp1 - snake_v

        # contribution from the i+1 triangles to dE/du
        dEk_du = (int_ends_v_next / s**2).squeeze() * np.sum(
            h * kappa_collection)
        dEk_du += (int_ends_v_prev / s**2).squeeze() * np.sum(
            h * kappa_collection)

        # contribution from the i+1 triangles to dE/dv
        dEk_dv = (int_ends_u_next / s**2).squeeze() * np.sum(
            h * kappa_collection)
        dEk_dv += (int_ends_u_prev / s**2).squeeze() * np.sum(
            h * kappa_collection)

        snake_u_new = inv @ (gamma * snake_u + u_interps)
        snake_v_new = inv @ (gamma * snake_v + v_interps)

        du = max_px_move * np.tanh(snake_u_new - snake_u)
        dv = max_px_move * np.tanh(snake_v_new - snake_v)

        snake_u += du
        snake_v += dv

        # du = -max_px_move * (
        #     (fu - dEk_du.view(fu.shape)) * gamma).tanh() + du
        # dv = -max_px_move * (
        #     (fv - dEk_dv.view(fv.shape)) * gamma).tanh() + dv

        # snake_u =  @ (gamma * snake_u + du)
        # snake_v = torch.inverse(
        #     gamma * torch.eye(L) +
        #     (A / delta_s + B / (delta_s * delta_s))) @ (gamma * snake_v + dv)

        # Movements are capped to max_px_move per iteration:
        #snake_u += du
        #snake_v += dv
        snake_u = np.clip(snake_u, a_max=M - 1, a_min=0)
        snake_v = np.clip(snake_v, a_max=N - 1, a_min=0)

    return np.array((snake_u, snake_v)).T
