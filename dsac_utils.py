import numpy as np
import torch.nn.functional as F
import torch


def make_init_snake(radius, shape, L):

    s = np.linspace(0, 2 * np.pi, L)
    init_u = shape / 2 + 5 * np.cos(s)
    init_v = shape / 2 + 5 * np.sin(s)
    init_u = init_u.reshape([L, 1])
    init_v = init_v.reshape([L, 1])
    init_snake = np.array([init_u[:, 0], init_v[:, 0]]).T

    return init_snake


def acm_inference(map_e, map_a, map_b, map_k, init_snake, gamma, max_px_move,
                  delta_s):
    # generate kernels for spatial gradients
    grad_u_weights = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                  dtype=torch.float32)
    grad_v_weights = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                  dtype=torch.float32)
    grad_u_weights = grad_u_weights.expand(1, 1, 3, 3)
    grad_v_weights = grad_v_weights.expand(1, 1, 3, 3)

    for i in range(map_e.shape[1]):

        # compute spatial gradient
        Du = F.conv2d(
            map_e[i, ...].unsqueeze(0),
            grad_u_weights,
            groups=map_e.shape[1],
            padding=1).squeeze()

        Dv = F.conv2d(
            map_e[i, ...].unsqueeze(0),
            grad_v_weights,
            groups=map_e.shape[1],
            padding=1).squeeze()

        u = init_snake[:, 0:1]
        v = init_snake[:, 1:2]
        du = np.zeros(u.shape)
        dv = np.zeros(v.shape)
        snake_hist = []

        # optimize
        u, v, du, dv = active_contour_step(
            Du, Dv, du, dv, u, v, map_a[i, ...], map_b[i, ...], map_k[i, ...],
            torch.tensor(gamma), max_px_move, delta_s)
        snake_hist.append(np.array([u[:, 0], v[:, 0]]).T)

    return snake_hist


def active_contour_step(Du, Dv, du, dv, snake_u, snake_v, alpha, beta, kappa,
                        gamma, max_px_move, delta_s):

    import pdb; pdb.set_trace()
    L = snake_u.shape[0]
    u = snake_u.round().type(torch.long)
    v = snake_v.round().type(torch.long)

    # Explicit time stepping for image energy minimization:
    fu = Du[u, v]
    fv = Du[u, v]
    a = alpha[0, u, v]
    b = beta[0, u, v]
    am1 = torch.cat([a[L - 1:L], a[0:L - 1]], 0)
    a0d0 = torch.diag(a)
    am1d0 = torch.diag(am1)
    a0d1 = torch.cat([a0d0[0:L, L - 1:L], a0d0[0:L, 0:L - 1]], 1)
    am1dm1 = torch.torch([am1d0[0:L, 1:L], am1d0[0:L, 0:1]], 1)

    bm1 = torch.cat([b[L - 1:L], b[0:L - 1]], 0)
    b1 = torch.cat([b[1:L], b[0:1]], 0)
    b0d0 = torch.diag(b)
    bm1d0 = torch.diag(bm1)
    b1d0 = torch.diag(b1)
    b0dm1 = torch.cat([b0d0[0:L, 1:L], b0d0[0:L, 0:1]], 1)
    b0d1 = torch.cat([b0d0[0:L, L - 1:L], b0d0[0:L, 0:L - 1]], 1)
    bm1dm1 = torch.cat([bm1d0[0:L, 1:L], bm1d0[0:L, 0:1]], 1)
    b1d1 = torch.cat([b1d0[0:L, L - 1:L], b1d0[0:L, 0:L - 1]], 1)
    bm1dm2 = torch.cat([bm1d0[0:L, 2:L], bm1d0[0:L, 0:2]], 1)
    b1d2 = torch.cat([b1d0[0:L, L - 2:L], b1d0[0:L, 0:L - 2]], 1)

    A = -am1dm1 + (a0d0 + am1d0) - a0d1
    B = bm1dm2 - 2 * (bm1dm1 + b0dm1) + (
        bm1d0 + 4 * b0d0 + b1d0) - 2 * (b0d1 + b1d1) + b1d2

    # Get kappa values between nodes
    s = 10
    range_float = tf.cast(tf.range(s), tf.float32)
    snake_u1 = tf.concat([snake_u[L - 1:L], snake_u[0:L - 1]], 0)
    snake_v1 = tf.concat([snake_v[L - 1:L], snake_v[0:L - 1]], 0)
    snake_um1 = tf.concat([snake_u[1:L], snake_u[0:1]], 0)
    snake_vm1 = tf.concat([snake_v[1:L], snake_v[0:1]], 0)
    u_interps = tf.cast(
        snake_u + tf.round(
            tf.multiply(range_float, (snake_um1 - snake_u)) / s), tf.int32)
    v_interps = tf.cast(
        snake_v + tf.round(
            tf.multiply(range_float, (snake_vm1 - snake_v)) / s), tf.int32)
    kappa_collection = tf.gather(
        tf.reshape(kappa, tf.TensorShape([M * N])), u_interps * M + v_interps)
    #kappa_collection = tf.reshape(kappa_collection,tf.TensorShape([L,s]))
    #kappa_collection = tf.Print(kappa_collection,[kappa_collection],summarize=1000)
    # Get the derivative of the balloon energy
    js = tf.cast(tf.range(1, s + 1), tf.float32)
    s2 = 1 / (s * s)
    int_ends_u_next = s2 * (snake_um1 - snake_u
                            )  # snake_u[next_i] - snake_u[i]
    int_ends_u_prev = s2 * (snake_u1 - snake_u)  # snake_u[prev_i] - snake_u[i]
    int_ends_v_next = s2 * (snake_vm1 - snake_v
                            )  # snake_v[next_i] - snake_v[i]
    int_ends_v_prev = s2 * (snake_v1 - snake_v)  # snake_v[prev_i] - snake_v[i]
    # contribution from the i+1 triangles to dE/du

    dEb_du = tf.multiply(
        tf.reduce_sum(
            tf.multiply(
                js,
                tf.gather(
                    kappa_collection, tf.range(s - 1, -1, delta=-1), axis=1)),
            axis=1), tf.squeeze(int_ends_v_next))
    dEb_du -= tf.multiply(
        tf.reduce_sum(tf.multiply(js, kappa_collection), axis=1),
        tf.squeeze(int_ends_v_prev))

    dEb_dv = -tf.multiply(
        tf.reduce_sum(
            tf.multiply(
                js,
                tf.gather(
                    tf.gather(
                        kappa_collection,
                        tf.concat([tf.range(L - 1, L),
                                   tf.range(L - 1)], 0),
                        axis=0),
                    tf.range(s - 1, -1, delta=-1),
                    axis=1)),
            axis=1), tf.squeeze(int_ends_u_next))
    dEb_dv += tf.multiply(
        tf.reduce_sum(
            tf.multiply(
                js,
                tf.gather(
                    kappa_collection,
                    tf.concat([tf.range(L - 1, L),
                               tf.range(L - 1)], 0),
                    axis=0),
            ),
            axis=1), tf.squeeze(int_ends_u_prev))

    du = -max_px_move * tf.tanh(
        (fu - tf.reshape(dEb_du, fu.shape)) * gamma) * 0.5 + du * 0.5
    dv = -max_px_move * tf.tanh(
        (fv - tf.reshape(dEb_dv, fv.shape)) * gamma) * 0.5 + dv * 0.5
    snake_u = tf.matmul(
        tf.matrix_inverse(
            tf.eye(L._value) +
            2 * gamma * (A / delta_s + B / (delta_s * delta_s))),
        snake_u + gamma * du)
    snake_v = tf.matmul(
        tf.matrix_inverse(
            tf.eye(L._value) +
            2 * gamma * (A / delta_s + B / (delta_s * delta_s))),
        snake_v + gamma * dv)

    #snake_u = np.matmul(np.linalg.inv(np.eye(L, L) + 2 * gamma * (A / delta_s + B / np.square(delta_s))),
    #                    snake_u + gamma * du)
    #snake_v = np.matmul(np.linalg.inv(np.eye(L, L) + 2 * gamma * (A / delta_s + B / np.square(delta_s))),
    #                    snake_v + gamma * dv)

    #snake_u += du
    #snake_v += dv
    snake_u = tf.minimum(snake_u, tf.cast(M, tf.float32) - 1)
    snake_v = tf.minimum(snake_v, tf.cast(N, tf.float32) - 1)
    snake_u = tf.maximum(snake_u, 1)
    snake_v = tf.maximum(snake_v, 1)

    return snake_u, snake_v, du, dv
