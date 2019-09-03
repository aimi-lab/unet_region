# -*- coding: utf-8 -*-
"""
Fit an ellipse for a point cloud

@author: Nicolas Guarin-Zapata
"""
from __future__ import division, print_function
import numpy as np
from numpy import sin, cos, arctan2, mean
from numpy.linalg import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def ellip_fun(coef, xi, yi):
    A, B, C, D, E, F = coef
    return norm(A*xi**2 + B*xi*yi + C*yi**2 + D*xi + E*yi + F)**2


def ellip_const(coef):
    return 4*coef[0]*coef[2] - coef[1]**2


def ellip_initial(xi, yi):
    
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]
    
    cov = np.cov(xi, yi)
    vals, vecs = eigsorted(cov)
    theta = arctan2(*vecs[:,0][::-1])
    x_new = cos(theta)*xi + sin(theta)*yi
    y_new = -sin(theta)*xi + cos(theta)*yi
    a = 0.5*(np.max(x_new) - np.min(x_new))
    b = 0.5*(np.max(y_new) - np.min(y_new))
    x0 = mean(xi)
    y0 = mean(yi)
    
    A = (a*sin(theta))**2 + (b*cos(theta))**2
    B = 2*(b**2 - a**2)*sin(theta)*cos(theta)
    C = (a*cos(theta))**2 + (b*sin(theta))**2
    D = -2*A*x0 - B*y0
    E = -B*x0 - 2*C*y0
    F = A*x0**2 + B*x0*y0 + C*y0**2 - a**2*b**2
    return A, B, C, D, E, F
    

a = 2
b = 1
theta_0 = np.pi/4
x0 = 10
y0 = 0
npts = 200
disp = 1/100
np.random.seed(seed=1)
theta = np.linspace(0, np.pi, npts)
xi = x0 + a*(cos(theta)*cos(theta_0) + disp*np.random.normal(size=npts)) -\
        b*(sin(theta)*sin(theta_0) + disp*np.random.normal(size=npts))
yi = y0 + a*(cos(theta)*sin(theta_0) + disp*np.random.normal(size=npts))+\
        b*(sin(theta)*cos(theta_0) + disp*np.random.normal(size=npts))
coef_0 = ellip_initial(xi, yi)
cons = {"type": "ineq", "fun": ellip_const}
opts = {'disp': True, "ftol": 1e-8, "maxiter": 300}
res = minimize(ellip_fun, coef_0, args=(xi, yi), method="SLSQP", tol=1e-8,
               options=opts, constraints=cons)

A, B, C, D, E, F = res.x
x_grid, y_grid = np.mgrid[np.min(xi):np.max(xi):101j,
                          np.min(yi):np.max(yi):101j]
z_grid = A*x_grid**2 + B*x_grid*y_grid + C*y_grid**2 + D*x_grid + E*y_grid  + F
plt.plot(xi, yi, ".", alpha=0.2)
plt.contour(x_grid, y_grid, z_grid, [0], linewidths=2, colors="black")
plt.show()
