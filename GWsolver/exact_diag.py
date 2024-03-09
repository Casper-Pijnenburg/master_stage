
from triqs.gf import *
from triqs.operators import *
from triqs.operators.util.hamiltonians import h_int_slater
from triqs.operators.util import *
from triqs.operators.util.hamiltonians import *
from triqs.atom_diag import *

def number_of_particles_from_ad(H, fundamental_operators, number_operator, beta, mu):
    Hmu = H + mu * number_operator
    atomic_diagonal = AtomDiagComplex(Hmu, fundamental_operators)
    density_matrix = atomic_density_matrix(atomic_diagonal, beta)
    filling = trace_rho_op(density_matrix, number_operator, atomic_diagonal)
    return filling.real, atomic_diagonal

def fix_atomic_diag(H, fundamental_operators, target_occupation, number_operator, beta, mu = None, eps = 1e-6):
    if mu is None:
        mu = 0

    step = 1.0

    occupation, atomic_diagonal = number_of_particles_from_ad(H, fundamental_operators, number_operator, beta, mu)

    previous_direction = None
    iterator = 0
    while abs(occupation - target_occupation) > eps:
        if occupation - target_occupation > 0:
            if previous_direction == 'decrement':
                step /= 2
            previous_direction = 'increment'
            mu += step
        if occupation - target_occupation < 0:
            if previous_direction == 'increment':
                step /= 2
            previous_direction = 'decrement'
            mu -= step
        occupation, atomic_diagonal = number_of_particles_from_ad(H, fundamental_operators, number_operator, beta, mu)
        iterator += 1
        # print(f"ED Iteration {iterator}: mu = {mu}, N = {occupation}", end='\r')
    return atomic_diagonal, mu

def exact_diag(tij, V, occupation = None, beta = 100, nw = 1 * 1024):
    spin_names = ['up', 'dn']
    orbital_names = range(len(tij))

    if occupation is None:
        occupation = len(tij)

    fundamental_operators = [(spin_name, orbital_name) for spin_name, orbital_name in product(spin_names, orbital_names)]

    N = 0
    for i in orbital_names:
        N += n('up', i) + n('dn', i)

    H = h_int_slater(spin_names, len(orbital_names), V, off_diag = True, complex = True)
        
    for spin, i, j in product(spin_names, orbital_names, orbital_names):
        H += tij[i, j] * c_dag(spin, i) * c(spin, j)
        
    atomic_diagonal, mu = fix_atomic_diag(H, fundamental_operators, occupation, N, beta)

    gf_struct = [('dn', len(orbital_names)),
                ('up', len(orbital_names))]

    G_w = atomic_g_iw(atomic_diagonal, beta, gf_struct, nw)
    return G_w