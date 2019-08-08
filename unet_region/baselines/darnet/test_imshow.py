"""
=================
An animated image
=================

This example demonstrates how to animate an image.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)


ax = plt.imshow(f(x, y))
x_c = 50 + 10*np.cos(np.linspace(0, 2*np.pi, 10))
y_c = 80 + 10*np.sin(np.linspace(0, 2*np.pi, 10))
plt.plot(x_c, y_c, 'ro-')

arr = ax._A

plt.show()
plt.imshow(arr);plt.show()
