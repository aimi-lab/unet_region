from unet_region.rw_net import rw_utils as rwu
import numpy as np
import matplotlib.pyplot as plt

size = 100
n_diags = 5
A = np.zeros((size, size))

# construct sparse symmetric matrix
for k in range(n_diags):
    if(k == 0):
        A += np.diag(np.random.rand(size - k), k)
    else:
        diag = np.random.rand(size - k)
        A += np.diag(diag, k)
        A += np.diag(diag, -k)

A = A / np.sum(A, axis=0)

A_inv = np.linalg.inv(A)

lambda_, Q = rwu.eigendecomposition_downsample(A, 1, n_vec=88)
A_inv_ds = Q.dot(np.diag(lambda_.ravel())).dot(Q.T)

fig, ax = plt.subplots(2, 3)
ax = ax.flatten()
ax[0].imshow(A)
ax[0].set_title('A')
ax[1].imshow(A_inv)
ax[1].set_title('A_inv no ds')
ax[2].imshow(A_inv_ds)
ax[2].set_title('A_inv ds')
ax[3].imshow(A_inv_ds.dot(A))
ax[3].set_title('A dot A_inv ds')
ax[4].imshow(A.dot(A_inv))
ax[4].set_title('A dot A_inv')
fig.show()
# plt.imshow(A);plt.show()
