import numpy as np
from triqs.gf import *
from triqs.operators import *
from triqs.operators.util.hamiltonians import h_int_slater
from triqs.operators.util import *
from triqs.operators.util.hamiltonians import *
from triqs.atom_diag import *
from triqs_tprf.lattice import dyson_mu
from triqs_tprf.lattice import dyson_mu_sigma


def ed_occupation(tij, V, beta, mu):
    spin_names = ['up', 'dn']
    orbital_names = range(len(tij))


    fundamental_operators = [(spin_name, orbital_name) for spin_name, orbital_name in product(spin_names, orbital_names)]

    N = 0
    for i in orbital_names:
        N += n('up', i) + n('dn', i)

    H = h_int_slater(spin_names, len(orbital_names), V, off_diag = True, complex = True)
        
    for spin, i, j in product(spin_names, orbital_names, orbital_names):
        H += tij[i, j] * c_dag(spin, i) * c(spin, j)
        
    occ, ad = number_of_particles_from_ad(H, fundamental_operators, N, beta, mu)

    return occ

def ed_mu(tij, V, beta, N_fix, N_tol):
    spin_names = ['up', 'dn']
    orbital_names = range(len(tij))


    fundamental_operators = [(spin_name, orbital_name) for spin_name, orbital_name in product(spin_names, orbital_names)]

    N = 0
    for i in orbital_names:
        N += n('up', i) + n('dn', i)

    H = h_int_slater(spin_names, len(orbital_names), V, off_diag = True, complex = True)
        
    for spin, i, j in product(spin_names, orbital_names, orbital_names):
        H += tij[i, j] * c_dag(spin, i) * c(spin, j)

    ad, mu = fix_atomic_diag(H, fundamental_operators, N_fix, N, beta, mu = 0, N_tol = N_tol)
    density_matrix = atomic_density_matrix(ad, beta)
    filling = trace_rho_op(density_matrix, N, ad)

    return mu, filling.real

def number_of_particles_from_ad(H, fundamental_operators, number_operator, beta, mu):
    atomic_diagonal = AtomDiagComplex(H + mu * number_operator, fundamental_operators)
    density_matrix = atomic_density_matrix(atomic_diagonal, beta)
    filling = trace_rho_op(density_matrix, number_operator, atomic_diagonal)
    return filling.real, atomic_diagonal

def fix_atomic_diag(H, fundamental_operators, N_fix, number_operator, beta, mu = 0, N_tol = 1e-5):
    if not N_fix:
        occupation, atomic_diagonal = number_of_particles_from_ad(H, fundamental_operators, number_operator, beta, mu)
        return atomic_diagonal, mu
    else:
        
        occupation, atomic_diagonal = number_of_particles_from_ad(H, fundamental_operators, number_operator, beta, mu)
        step = abs(occupation - N_fix)
        previous_direction = None
        while abs(occupation - N_fix) > N_tol:
            
            if occupation - N_fix > 0:
                if previous_direction == 'decrement':
                    step /= 2
                previous_direction = 'increment'
                mu += step
            if occupation - N_fix < 0:
                if previous_direction == 'increment':
                    step /= 2
                previous_direction = 'decrement'
                mu -= step
            occupation, atomic_diagonal = number_of_particles_from_ad(H, fundamental_operators, number_operator, beta, mu)

        return atomic_diagonal, mu

def N(g_w):
    return g_w.total_density().real
    
def _dyson_dispatch(g_w, mu, sigma_w = None):
    G = g_w.copy()
    mu_gf = g_w['up'].copy()
    mu_gf.data[:] = np.eye(g_w['up'][Idx(0)].shape[0]) * mu
    if sigma_w is not None:
        for block, g in g_w:
            G[block] = (g_w[block].inverse() - mu_gf - sigma_w[block]).inverse()
        return G
    
    for block, g in g_w:
        G[block] = (g_w[block].inverse() - mu_gf).inverse()
    return G

def dyson_equation(g_w, mu, sigma_w = None, N_fix = False, N_tol = 1e-5):
    if not N_fix:
        if mu == 0 and sigma_w is None:
            return g_w, mu
        return _dyson_dispatch(g_w, mu, sigma_w), mu
    
    else:
        
        previous_direction = None

        occupation = N(_dyson_dispatch(g_w, mu, sigma_w))
        step = abs(occupation - N_fix)
        while abs(occupation - N_fix) > N_tol:
            # print(f'occupation: {occupation}, mu: {mu}')

            if occupation - N_fix > 0.0:
                if previous_direction == 'decrement':
                    step /= 2.0
                previous_direction = 'increment'
                mu += step
            if occupation - N_fix < 0.0:
                if previous_direction == 'increment':
                    step /= 2.0
                previous_direction = 'decrement'
                mu -= step
            
            occupation = N(_dyson_dispatch(g_w, mu, sigma_w))
            

        return _dyson_dispatch(g_w, mu, sigma_w), mu


def exact_diag(tij, V, N_fix, beta = 100, nw = 1 * 1024, N_tol = 1e-5, mu = 0.0):
    spin_names = ['up', 'dn']
    orbital_names = range(len(tij))


    fundamental_operators = [(spin_name, orbital_name) for spin_name, orbital_name in product(spin_names, orbital_names)]

    N = 0
    for i in orbital_names:
        N += n('up', i) + n('dn', i)

    H = h_int_slater(spin_names, len(orbital_names), V, off_diag = True, complex = True)
        
    for spin, i, j in product(spin_names, orbital_names, orbital_names):
        H += tij[i, j] * c_dag(spin, i) * c(spin, j)
        
    atomic_diagonal, mu = fix_atomic_diag(H, fundamental_operators, N_fix, N, beta, mu = mu, N_tol = N_tol)

    gf_struct = [('dn', len(orbital_names)),
                ('up', len(orbital_names))]

    G_w = atomic_g_iw(atomic_diagonal, beta, gf_struct, nw)
    return G_w, mu



def exact_diag_v(tij, V, N_0_fix, N_gw_fix, beta = 100, nw = 1 * 1024, N_tol = 1e-5, mu_0 = 0.0, mu_gw = 0.0):
    spin_names = ['up', 'dn']
    orbital_names = range(len(tij))


    fundamental_operators = [(spin_name, orbital_name) for spin_name, orbital_name in product(spin_names, orbital_names)]

    N = 0
    for i in orbital_names:
        N += n('up', i) + n('dn', i)

    H = h_int_slater(spin_names, len(orbital_names), V, off_diag = True, complex = True)
        
    for spin, i, j in product(spin_names, orbital_names, orbital_names):
        H += tij[i, j] * c_dag(spin, i) * c(spin, j)
        
    atomic_diagonal, mu1 = fix_atomic_diag(H, fundamental_operators, N_0_fix, N, beta, mu = mu_0, N_tol = N_tol)

    gf_struct = [('dn', len(orbital_names)),
                ('up', len(orbital_names))]

    G_w = atomic_g_iw(atomic_diagonal, beta, gf_struct, nw)
    G_w, mu2 = dyson_equation(G_w, mu_gw, sigma_w = None, N_fix = N_gw_fix, N_tol = N_tol)

    return G_w, mu1, mu2