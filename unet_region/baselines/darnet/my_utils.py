import numpy as np
from shapely.geometry import Polygon, LineString
import warnings
from shapely.geometry.polygon import LinearRing
import time
import math
from scipy.interpolate import RectBivariateSpline
import skfmm
from skimage import segmentation
import matplotlib.pyplot as plt
from scipy.ndimage import measurements
from scipy import interpolate
from scipy.ndimage import distance_transform_edt


class RaySnake:
    def __init__(self, rc, L, r0=None, shape=None, arr=None):

        self.L = L
        self.rc = rc

        if ((r0 is not None) and (shape is not None)):
            self.r = np.array(L * [r0])
            self.shape = shape
        else:
            self.r = self.get_contour(arr, rc, L)
            self.shape = arr.shape
            if (arr is None):
                raise Exception('Provide either r0 and shape, or arr')

        self.make_bounds()

        self.r = self.clip_r()

    def make_bounds(self):
        thetas = np.linspace(0, 2 * np.pi, self.L)

        bounds = []

        for theta in thetas:
            r_max_x_0 = np.abs(
                (self.shape[1] - 1 - self.rc[0]) / np.cos(theta))
            r_max_x_1 = np.abs((-self.rc[0]) / np.cos(theta))
            r_max_y_0 = np.abs(
                (self.shape[0] - 1 - self.rc[1]) / np.sin(theta))
            r_max_y_1 = np.abs((-self.rc[1]) / np.sin(theta))
            bounds.append(np.min((r_max_x_0, r_max_x_1, r_max_y_0, r_max_y_1)))
        self.bounds = bounds
        return self.bounds

    def clip_r(self):

        r = [np.clip(r, a_min=0, a_max=b) for b, r in zip(self.bounds, self.r)]
        self.r = np.array(r)
        return self.r

    def update_r(self, new_r):
        self.r = new_r
        self.clip_r()

    @property
    def delta_theta(self):
        return 2 * np.pi / self.L

    @property
    def cart_coords(self):
        thetas = np.linspace(0, 2 * np.pi, self.L)[:-1]

        cart_coords = [(self.rc[0] + r * np.cos(theta),
                        self.rc[1] + r * np.sin(theta))
                       for r, theta in zip(self.r[:-1], thetas)]
        cart_coords += [cart_coords[0]]

        return np.array(cart_coords)


def active_contour_steps(image,
                         snake,
                         beta,
                         kappa,
                         delta_t=1,
                         gamma=0.01,
                         max_px_move=1.0,
                         max_iterations=2500,
                         convergence=0.1,
                         verbose=False):
    """Active contour model.

    Returns
    -------
    snake : (N, 2) ndarray
        Optimised snake, same shape as input parameter.

    """
    max_iterations = int(max_iterations)

    if max_iterations <= 0:
        raise ValueError("max_iterations should be >0.")

    convergence_order = 10

    start = time.time()

    intp = RectBivariateSpline(
        np.arange(image.shape[1]),
        np.arange(image.shape[0]),
        image.T,
        kx=2,
        ky=2,
        s=0)

    # Explicit time stepping for image energy minimization:
    for i in range(max_iterations):
        cart_coords = snake.cart_coords

        x, y = cart_coords[:, 0], cart_coords[:, 1]
        x_int = x.round().astype(int)
        y_int = y.round().astype(int)

        # Build snake shape matrix for Euler equation
        a = 2 * beta[np.roll(y_int, -1
                             ), np.roll(x_int, -1)] * np.cos(
                                 2 * snake.delta_theta)
        b = -4 * (beta[np.roll(y_int, -1),
                       np.roll(x_int, -1)] + \
                beta[np.roll(y_int, 1),
                     np.roll(x_int, 1)]*np.cos(snake.delta_theta))
        c = 2 * (beta[np.roll(y_int, -1), np.roll(x_int, -1)] + \
            4 * beta[y_int, x_int] + \
            beta[np.roll(y_int, 1), np.roll(x_int, 1)])
        d = -4 * (beta[y_int, x_int] + \
            beta[np.roll(y_int, 1),
                 np.roll(x_int, 1)])*np.cos(snake.delta_theta)
        e = 2 * (beta[np.roll(y_int, 1), np.roll(x_int, 1)]) * np.cos(
            2 * snake.delta_theta)

        # find derivative of "data" external energy function
        fx = intp(x, y, dx=1, grid=False)
        fy = intp(x, y, dy=1, grid=False)

        fx *= np.cos(np.arange(snake.L) * snake.delta_theta)
        fy *= np.sin(np.arange(snake.L) * snake.delta_theta)

        f = fx + fy - kappa[y_int, x_int] / np.max(snake.bounds)

        A = np.diag(c)
        A += np.roll(np.diag(b), -1, axis=0)
        A += np.roll(np.diag(a), -2, axis=0)
        A += np.roll(np.diag(d), 1, axis=0)
        A += np.roll(np.diag(e), 2, axis=0)

        rho = snake.r
        drho = delta_t * (rho @ A + f)
        drho = np.clip(drho, a_min=-max_px_move, a_max=max_px_move)
        new_rho = rho - drho
        snake.update_r(new_rho)

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

    end = time.time()
    if (verbose):
        print('finished in {} iterations in {} s'.format(i + 1, end - start))

    return snake


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


def make_spline_contour(arr, rc, ds_contour_rate=0.1, n=1000, s=0, k=3):

    arr_ = np.pad(arr, ((1, ), (1, )), mode='constant', constant_values=False)
    contour = segmentation.find_boundaries(arr_, mode='thick')
    y, x = np.where(contour)

    x = x.tolist()
    y = y.tolist()

    pts = [(x, y) for x, y in zip(x, y)]

    sort_fn = lambda pt: clockwiseangle(rc, pt)

    pts = sorted(pts, key=sort_fn)

    pts = np.array(pts)
    x, y = pts[:, 0], pts[:, 1]
    nodes = np.zeros([n, 2])
    [tck, u] = interpolate.splprep([x, y], s=s, k=k)
    [nodes[:, 0], nodes[:, 1]] = interpolate.splev(np.linspace(0, 1, n), tck)

    return nodes


def get_contour(a, rc, L):

    contour_nodes = make_spline_contour(a, 100, rc)
    contour_polyg = Polygon(contour_nodes)

    line_length = np.max(a.shape)
    p0 = rc
    p1s = [(rc[0] + line_length * np.cos(theta),
            rc[1] + line_length * np.sin(theta))
           for theta in np.linspace(0, 2 * np.pi, L)]

    # store lines from rc to all remote points
    rays = [LineString([p0, p1]) for p1 in p1s]

    intersects = [contour_polyg.intersection(ray) for ray in rays]

    intersects_coords = []
    for l in intersects:
        try:
            intersects_coords.append(l.coords[-1])
        except:
            try:
                intersects_coords.append(l[0].coords[-1])
            except:
                intersects_coords.append(rc)

    intersects_coords = np.array(intersects_coords)

    intersects_coords_centered = intersects_coords - rc
    radii = np.linalg.norm(intersects_coords_centered, axis=1)

    return radii, intersects_coords


def process_truth_dar(sample, L, init_rho_rel):
    """
    This is called on segmentation mask to extract the ground truth snake, etc...
    """
    truth = sample['label/segmentation'][..., 0]
    shape = truth.shape
    rc = np.array([shape[1] // 2, shape[0] // 2])

    contour_nodes = make_spline_contour(truth, rc, s=0, k=3, n=1000)

    interp_coords, interp_radii, angles, delta_angles = interpolate_ground_truth_polygon(
        L,
        rc,
        contour_nodes)
    # interp_radii, r = get_contour(truth, rc, L)

    sample['interp_radii'] = interp_radii[..., np.newaxis, np.newaxis]
    sample['init_contour_origin'] = rc[..., np.newaxis, np.newaxis]
    sample['interp_xy'] = interp_coords[..., np.newaxis]
    sample['interp_angles'] = angles[..., np.newaxis, np.newaxis]
    sample['delta_angles'] = delta_angles[..., np.newaxis, np.newaxis, np.newaxis]

    truth_contour = (segmentation.find_boundaries(truth > 0))
    data = distance_transform_edt(np.logical_not(truth_contour))
    beta = data.copy()
    beta[truth > 0] = 0
    kappa = data.copy()
    kappa[truth == 0] = 0

    sample['label/edt_D'] = data[..., np.newaxis]
    sample['label/edt_beta'] = beta[..., np.newaxis]
    sample['label/edt_kappa'] = kappa[..., np.newaxis]

    init_rho = init_rho_rel * np.max(truth.shape)
    sample['init_contour_radii'] = np.array(
        [init_rho] * int(L))[..., np.newaxis, np.newaxis]

    return sample


def interpolate_ground_truth_polygon(N, rc, poly):
    """ Given ground truth polygon, interpolate points for ground truth active rays.
    """
    angles, delta_angles = np.linspace(
        0, 2 * np.pi, N, endpoint=False, retstep=True)
    interp_coords = []
    interp_radii = []

    for theta in angles:
        dy = np.sin(theta)
        dx = np.cos(theta)
        d = np.array([dx, dy])
        v3 = np.array([-dy, dx])

        a = poly[:-1, :]
        b = poly[1:, :]
        v1 = a - rc
        v2 = a - b

        t2 = np.inner(v1, v3) / np.inner(v2, v3)
        t1 = np.cross(v1, -v2) / np.inner(-v2, v3)

        intersections = np.logical_and(t1 > 0, np.logical_and(
            0 <= t2, t2 <= 1))
        radii = t1 * intersections
        radii[np.isnan(radii)] = 0

        if np.count_nonzero(radii) > 0:
            radius = np.min(radii[radii.nonzero()])
        else:
            radius = 0

        interp_radii.append(radius)
        interp_coords.append(rc + radius * d)
            

    assert len(interp_radii) == N, "Radii is {}".format(
        interp_radii)

    interp_coords = np.array(interp_coords)
    interp_radii = np.array(interp_radii)

    return interp_coords, interp_radii, angles, delta_angles
