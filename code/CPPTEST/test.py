import numpy as np

from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq, MeshDLRImTime

from triqs.atom_diag import *
import numpy as np
from itertools import product
import matplotlib.pylab as plt
import time
from triqs.plot.mpl_interface import oplot,plt

import triqs_tprf
from triqs_tprf.lattice import polarization
from triqs_tprf.lattice import screened_potential
from triqs_tprf.lattice import dyn_self_energy
from triqs_tprf.lattice import hartree_self_energy
from triqs_tprf.lattice import fock_self_energy
from triqs_tprf.lattice import dyson_mu
from triqs_tprf.lattice import dyson_mu_sigma

def generate_g0_w(tij, mesh, spin_names = ['up', 'dn']):
    g_inv = Gf(mesh = mesh, target_shape = np.shape(tij))
    g_inv << iOmega_n - tij.transpose()
    g = g_inv.inverse()
    return BlockGf(block_list = [g] * 2, name_list = spin_names, make_copies = False)

def coulomb_matrix(orbitals, U, non_local = True):
    Vij = np.zeros([orbitals] * 2)
    for i in range(orbitals):
        for j in range(orbitals):
            Vij[i, j] = round(U / (abs(i - j) + 1), 2)
    

    if non_local:
        return Vij

    return np.diag(Vij.diagonal())


def W_py(P_w, V, self_interactions):
        W = P_w.copy()

        V_t = V.copy()

        if not self_interactions:
            np.fill_diagonal(V_t, 0)

        I = np.eye(len(V))

        A = I - V_t * P_w['up']
        B =  - V * P_w['dn']
        C =  - V * P_w['up']
        D = I - V_t * P_w['dn']

        A_inv = A.inverse()

        S = (D - C * A_inv * B).inverse()

        W['up'] = (A_inv + A_inv * B * S * C * A_inv) * V_t - A_inv * B * S * V;
        W['dn'] = -S * C * A_inv * V + S * V_t;
    
        return W

orbitals = 300

t = 1.0

tij = np.zeros([orbitals] * 2)
for i in range(orbitals - 1):
    tij[i, i + 1] = -t
    tij[i + 1, i] = -t

v = coulomb_matrix(orbitals, 1.0)
v_t = v.copy()
np.fill_diagonal(v_t, 0)

beta = 100

self_interactions = False
hartree_flag = False
fock_flag = False

mesh = MeshDLRImFreq(beta = beta, statistic = 'Fermion',  w_max = 5.0, eps = 1e-12)
bmesh = MeshDLRImFreq(beta = beta, statistic = 'Boson',  w_max = 5.0, eps = 1e-12, symmetrize = True)
mesh_sym = MeshDLRImFreq(beta = beta, statistic = 'Fermion',  w_max = 5.0, eps = 1e-12, symmetrize = True)

G_w = generate_g0_w(tij, mesh_sym)
P_w = polarization(G_w, bmesh, 8)

print("STARTING W")
start = time.perf_counter()
W_w_py = W_py(P_w, v, self_interactions)
print(time.perf_counter() - start)