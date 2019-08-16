import numpy as np
from scipy.misc import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t)
                                  for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    n_quadrants = 8
    np.random.seed(0)
    angles = np.linspace(0,360, 2*n_quadrants+1)
    radii = np.random.choice(np.linspace(0, 1, 100),
                             size=angles.size)
    nPoints = 3
    points = np.random.rand(nPoints,2)
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]

    xvals, yvals = bezier_curve(points, nTimes=50)
    plt.plot(xvals, yvals)
    plt.plot(xpoints, ypoints, "ro")
    plt.plot([xpoints[0], xpoints[1]],
             [ypoints[0], ypoints[1]],
             'b--')
    plt.plot([xpoints[1], xpoints[2]],
             [ypoints[1], ypoints[2]],
             'b--')
    for nr in range(len(points)):
        plt.text(points[nr][0], points[nr][1], nr)

    plt.grid()
    plt.show()
