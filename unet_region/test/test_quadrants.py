import numpy as np
import shapely.geometry as geom
import matplotlib.pyplot as plt
from scipy import interpolate
from skimage import draw

def make_plot(fname, seed=None):
    n_quadrants = 8
    out_shape = 256

    # Define the arc (presumably ezdxf uses a similar convention)
    centerx, centery = 0, 0
    radius = 1
    numsegments = 1000
    start_angles = np.linspace(0,
                            360 * ((n_quadrants -1) / n_quadrants),
                            n_quadrants)
    inc_angle = 360 / n_quadrants

    polys = []
    for start_angle in start_angles:

        end_angle = start_angle + inc_angle
        # The coordinates of the arc
        theta = np.radians(np.linspace(start_angle, end_angle, numsegments))
        x = centerx + radius * np.cos(theta)
        y = centery + radius * np.sin(theta)

        xy = np.array((x, y))
        center = np.array((centerx, centery))[..., np.newaxis]
        pts = np.concatenate((center, xy), axis=1)
        poly = geom.Polygon([(pts[0, i], pts[1, i])
                            for i in range(pts.shape[1])])
        polys.append(poly)


    fig, ax = plt.subplots(1, 2)

    colors = []
    for poly in polys:
        line = ax[0].plot(*poly.exterior.xy, '--')
        colors.append(line[0].get_color())

    # make random point on each quadrants
    if(seed is not None):
        np.random.seed(seed)

    rho_range = np.linspace(0.2, 1, 100)
    theta_range = np.linspace(0, 1, 100)

    # pts = [(np.random.choice(rho_range), np.random.choice(theta_range))
    #        for _ in range(n_quadrants)]
    pts = [(np.random.choice(rho_range), 0.5)
        for _ in range(n_quadrants)]

    pts_cart = []
    for p, start_angle, color in zip(pts, start_angles, colors):
        rho = p[0]
        theta = np.radians(start_angle + inc_angle*p[1])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        x, y = rho, 0
        p = np.array((x, y))[..., np.newaxis]
        p = np.dot(R, p)
        pts_cart.append(p)
        ax[0].plot(p[0], p[1], 'o', color=color)

    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    pts_cart = np.array(pts_cart)[... ,0]
    pts_cart = np.concatenate((pts_cart, pts_cart[0, ...][np.newaxis, ...]), axis=0)
    import pdb; pdb.set_trace()
    tck, u = interpolate.splprep([pts_cart[:, 0], pts_cart[:, 1]],
                                s=0,
                                per=True)

    # evaluate the spline fits for 1000 evenly spaced distance values
    xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)
    ax[0].plot(xi, yi, 'k')
    ax[0].set_aspect('equal')
    ax[0].grid()

    # draw full shape
    xi = ((xi + 1) / 2) * out_shape
    yi = -yi
    yi = ((yi + 1) / 2) * out_shape
    rr, cc = draw.polygon(yi, xi, (out_shape, out_shape))
    img = np.zeros((out_shape, out_shape), dtype=bool)
    img[rr, cc] = 1

    ax[1].imshow(img)
    fig.savefig(fname)

    # plt.show()

n_figs = 50
fnames = ['fig_{:04d}.png'.format(i)
          for i in range(n_figs)]
for i in range(n_figs):
    print('{}/{}'.format(i+1, n_figs))
    make_plot(fnames[i], i)
