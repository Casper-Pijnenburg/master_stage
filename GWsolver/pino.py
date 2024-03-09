import numpy as np
from triqs.gf import *
from itertools import product

def exact_polarization(mesh, t):
    g = Gf(mesh = mesh, target_shape = [2, 2])

    indices = (0, 1)
    for i, j in product(indices, indices):
        for w in mesh:
            g[w][i, j] = (-1) ** (i - j) * (1 / (w - 2 * t) - 1 / (w + 2 * t)) / 4
 
    return BlockGf(block_list = [g] * 2, name_list = ['up', 'dn'], make_copies = False)

def exact_screened_potential(mesh, t, U):
    g = Gf(mesh = mesh, target_shape = [2, 2])

    h_sqrd = 4 * t ** 2 + 4 * U * t

    indices = (0, 1)
    for i, j in product(indices, indices):
        for w in mesh:
            g[w][i, j] = U * int(i == j) + (-1) ** (i - j) * 2 * U ** 2 * t / (complex(w) ** 2 - h_sqrd)
 
    return BlockGf(block_list = [g] * 2, name_list = ['up', 'dn'], make_copies = False)


def exact_self_energy(mesh, t, U):
    g = Gf(mesh = mesh, target_shape = [2, 2])

    h_sqrd = 4 * t ** 2 + 4 * U * t
    h = np.sqrt(h_sqrd)

    indices = (0, 1)
    for i, j in product(indices, indices):
        for w in mesh:
            g[w][i, j] = U * int(i == j) + U ** 2 * t * (1 / (w - t - h) + (-1) ** (i - j) / (w + t + h)) / (2 * h)
 
    return BlockGf(block_list = [g] * 2, name_list = ['up', 'dn'], make_copies = False)


def exact_g(mesh, t, U, interacting = True):
    u = U * int(interacting)
    g = Gf(mesh = mesh, target_shape = [2, 2])

    h_sqrd = 4 * t ** 2 + 4 * u * t
    h = np.sqrt(h_sqrd)

    A = np.sqrt((2 * t + h + u / 2) ** 2 + 4 * u ** 2 * t / h)
    B = np.sqrt((2 * t + h - u / 2) ** 2 + 4 * u ** 2 * t / h)

    w1p = 0.5 * (u / 2 - h + A)
    w1m = 0.5 * (u / 2 - h - A)
    w2p = 0.5 * (u / 2 + h + B)
    w2m = 0.5 * (u / 2 + h - B)

    indices = (0, 1)
    for i, j in product(indices, indices):
        for w in mesh:
            g[w][i, j] = (-1) ** (i - j) * ((0.25 + (h + 2 * t + u / 2) / (4 * A)) / (w - w1p) + (0.25 - (h + 2 * t + u / 2) / (4 * A)) / (w - w1m)) \
                         + (0.25 + (-h - 2 * t + u/2) / (4 * B)) / (w - w2p) + (0.25 - (-h - 2 * t + u/2) / (4 * B)) / (w - w2m)
 
    return BlockGf(block_list = [g] * 2, name_list = ['up', 'dn'], make_copies = False)