import numpy as np
from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq, MeshDLRImTime
import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import time
import multiprocessing

from gwsolver import GWSolver

def generate_G(tij, mesh, spin_names = ['up', 'dn']):
    g_inv = Gf(mesh = mesh, target_shape = np.shape(tij))
    g_inv << iOmega_n - tij.transpose()
    g = g_inv.inverse()
    return BlockGf(block_list = [g] * 2, name_list = spin_names, make_copies = False)


def coulomb_matrix(orbitals, U, non_local = True):
    Vij = np.zeros([orbitals] * 2)
    for i in range(orbitals):
        for j in range(orbitals):
            Vij[i, j] = round(0.75 * U / (abs(i - j) + 1), 2)

        Vij[i, i] = U
    if non_local:
        return Vij

    return np.diag(Vij.diagonal())


orbitals = 500
N_fix = orbitals
t = 1.0
U = 0.5
beta = 100


tij = np.zeros([orbitals] * 2)
for i in range(orbitals - 1):
    tij[i, i + 1] = -t
    tij[i + 1, i] = -t

V = coulomb_matrix(orbitals, U, non_local = False)

iw_mesh = MeshDLRImFreq(beta = beta, statistic = 'Fermion', w_max = 5.0, eps = 1e-12, symmetrize = True)
G_iw = generate_G(tij, iw_mesh)


start = time.perf_counter()
gw = GWSolver(G_iw, V, self_interactions = False, hartree_flag = True, fock_flag = True, mu = 0, N_fix = N_fix, max_iter = 1, number_of_cores = 8)
print(time.perf_counter() - start)
print(gw.W_w_time)