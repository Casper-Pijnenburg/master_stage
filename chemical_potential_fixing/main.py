import numpy as np
import time

from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq
from triqs_tprf.lattice import inv

from itertools import product
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from gwsolver import GWSolver
from exact_diag import exact_diag, ed_occupation, ed_mu
from pino import exact_g, analytical_g

from hubbard import GWHubbardDimer
matplotlib.rcParams.update({'font.size': 13})


def plot(a, b, spin = 'up'):
    orbitals = len(a[spin].data[0, 0, :])
    fig, axs = plt.subplots(orbitals, orbitals, figsize = (10 * orbitals, 10 * orbitals))
    spin = 'up'


    for i in range(orbitals):
        for j in range(orbitals):
            # axs[i, j].set_facecolor('black')
            # axs[i, j].xaxis.label.set_color('white')
            # axs[i, j].tick_params(axis = 'x', colors = 'white')
            # axs[i, j].yaxis.label.set_color('white')
            # axs[i, j].tick_params(axis = 'y', colors = 'white')
            axs[i, j].set_xlim(-5, 5)
            # axs[i, j].xaxis.label.set_fontsize(20)

            axs[i, j].scatter([w.imag for w in a.mesh.values()], a[spin].data[:, i, j].real, color = 'blue', zorder = 1, s = 20)
            axs[i, j].scatter([w.imag for w in a.mesh.values()], a[spin].data[:, i, j].imag, color = 'red', zorder = 1, s = 20)

            axs[i, j].plot([w.imag for w in b.mesh.values()], b[spin].data[:, i, j].real, color = 'black', zorder = 0)
            axs[i, j].plot([w.imag for w in b.mesh.values()], b[spin].data[:, i, j].imag, color = 'black', zorder = 0)

def generate_g0_w(tij, mesh, spin_names = ['up', 'dn']):
    g_inv = Gf(mesh = mesh, target_shape = np.shape(tij))
    g_inv << iOmega_n - tij.transpose()
    g = inv(g_inv, 8)
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

def coulomb_tensor(orbitals, U, non_local = True):
    Vij = coulomb_matrix(orbitals, U, non_local)
    Vijkl = np.zeros([orbitals] * 4)
    for i in range(orbitals):
        for j in range(orbitals):
            Vijkl[i, j, i, j] = Vij[i, j]
    return Vijkl

def N(g_w):
    return g_w.total_density().real

def GWtoED(orbitals, N_fix, t, U, beta = 20, max_iter = 1, non_local = False, pade = False):

    tij = np.zeros([orbitals] * 2)
    for i in range(orbitals - 1):
        tij[i, i + 1] = -t
        tij[i + 1, i] = -t

    Vij = coulomb_matrix(orbitals, U, non_local = non_local)
    Vijkl = coulomb_tensor(orbitals , U, non_local = non_local)

    iw_mesh_f = MeshDLRImFreq(beta = beta, statistic = 'Fermion',  w_max = 20.0, eps = 1e-12, symmetrize = True)
    g0_w = generate_g0_w(tij, iw_mesh_f)

    ED, mu_ed = exact_diag(tij, Vijkl, N_fix, beta = beta, nw = 1024)

    gw = GWSolver(g0_w, Vij, self_interactions = False, hartree_flag = True, fock_flag = True, mu = 0, N_fix = N_fix, max_iter = max_iter, full_mesh = False)
    
    # print(f'g0_w occupation: {round(N(gw.g0_w), 5)}, Chemical potential: {gw.mu0}')
    # print(f'g_w occupation: {round(N(gw.g_w), 5)}, Chemical potential: {gw.mu}')
    # print(f'ED occupation: {round(N(ED), 5)}, Chemical potential: {mu_ed}')
    # print(N(ED))
    # print(gw.mu)
    # plot(gw.g_w, ED)

    fig, ax = plt.subplots(figsize = (6, 6), tight_layout = True)
    a = gw.g_w
    b = ED
    spin = 'up'
    ax.set_xlim(-5, 5)
    ax.scatter([w.imag for w in a.mesh.values()], a[spin].data[:, 0, 0].real, color = 'blue', zorder = 1, s = 20, label = 'Real')
    ax.scatter([w.imag for w in a.mesh.values()], a[spin].data[:, 0, 0].imag, color = 'red', zorder = 1, s = 20, label = 'Imag')

    ax.plot([w.imag for w in b.mesh.values()], b[spin].data[:, 0, 0].real, color = 'black', zorder = 0, label = 'E.D.')
    ax.plot([w.imag for w in b.mesh.values()], b[spin].data[:, 0, 0].imag, color = 'black', zorder = 0)
    # ax.text(0.6, 0.9, f"# E.D = {round(N(ED), 4)}", transform = ax.transAxes, fontsize = 15, ha = 'left')
    # ax.text(0.6, 0.85, f"# GW = {round(N(gw.g_w), 4)}", transform = ax.transAxes, fontsize = 15, ha = 'left')
    # ax.text(0.6, 0.8, f"μ = {round(gw.mu, 4)}", transform = ax.transAxes, fontsize = 15, ha = 'left')


    text_ED = f"# E.D = {round(N(ED), 4)}"
    text_GW = f"# GW = {round(N(gw.g_w), 4)}"
    text_mu = f"μ = {round(gw.mu, 4)}"

    # Determine the position of the equal sign
    equal_pos = max(len(text_ED.split('=')[0]), len(text_GW.split('=')[0]), len(text_mu.split('=')[0]))

    # Plot the text
    ax.text(0.6, 0.85, text_ED, transform=ax.transAxes, fontsize=15, ha='left')
    ax.text(0.6, 0.8, text_GW, transform=ax.transAxes, fontsize=15, ha='left')

    shift = 0.014

    ax.text(0.6 + equal_pos * shift, 0.75, text_mu, transform=ax.transAxes, fontsize=15, ha='left')
    ax.text(0.6 + equal_pos * shift, 0.9, f"U = {U}", transform=ax.transAxes, fontsize=15, ha='left')


    ax.set_xlabel('iω')
    ax.legend(loc = 2)
    # plt.savefig('sc2.pdf')
    plt.show()


def _error(ED, g0, Vij, N_fix, max_iter):
    def rel_error(ed, g):
        orbitals = g['up'][Idx(0)].shape[0]
        size = len(list(g.mesh.values()))
        errors = np.zeros((size, orbitals, orbitals)) * 1j
        for i, w in enumerate(g.mesh):
            ed = ED['up'](w)
            g_up = g['up'].data[i, :, :]
            errors[i] = (ed - g_up) / ed

        return 100 * np.max(np.abs(errors))
    

    gw = GWSolver(g0, Vij, self_interactions = False, hartree_flag = True, fock_flag = True, mu = 0, N_fix = N_fix, max_iter = max_iter, full_mesh = False)
    
    return rel_error(ED, gw.g_w)


def errors(orbitals, N_fix, beta = 100, max_iter = 10, non_local = False):
    def rel_error(ed, g):
        size = len(list(g.mesh.values()))
        errors = np.zeros((size, orbitals, orbitals)) * 1j
        for i, w in enumerate(g.mesh):
            ed = ED['up'](w)
            g_up = g['up'].data[i, :, :]
            errors[i] = (ed - g_up) / ed

        return 100 * np.max(np.abs(errors))
 
    t = 1.0
    tij = np.zeros([orbitals] * 2)
    for i in range(orbitals - 1):
        tij[i, i + 1] = -t
        tij[i + 1, i] = -t

    iw_mesh_f = MeshDLRImFreq(beta = beta, statistic = 'Fermion',  w_max = 20.0, eps = 1e-12, symmetrize = True)
    g0_w = generate_g0_w(tij, iw_mesh_f)

    U_list = np.linspace(0, 2.0, 200, endpoint = True)
    errors_list = []
    for j, U in enumerate(U_list):
        Vij = coulomb_matrix(orbitals, U, non_local = non_local)
        Vijkl = coulomb_tensor(orbitals , U, non_local = non_local)

        errors = []

        ED, mu_ed = exact_diag(tij, Vijkl, N_fix, beta = beta, nw = 1024)
  
        for i in range(1, max_iter + 2):
            errors.append(_error(ED, g0_w, Vij, N_fix, i))

        errors.append(_error(ED, g0_w, Vij, N_fix, 200))

        errors_list.append(errors)

    errors_list = np.array(errors_list).transpose()
    fig, ax = plt.subplots(figsize = (8, 6), tight_layout = True)
    lw = 2

    print(errors_list)

    for i in range(len(errors_list)):
        if i == 0:
            ax.plot(U_list, errors_list[i, :], label = f'One-shot', lw = lw)
        elif i == 1:
            ax.plot(U_list, errors_list[i, :], label = f'1 iteration', lw = lw)
        elif i == len(errors_list) - 1:
            ax.plot(U_list, errors_list[i, :], label = f'Fully self-consistent', lw = lw)
        else:
            ax.plot(U_list, errors_list[i, :], label = f'{i} iterations', lw = lw)

    ax.set_xlim(0, 2.001)
    ax.set_ylim(0, 15)
    ax.set_xlabel('U/t', fontsize = 20)
    ax.set_ylabel('Max relative error (%)', fontsize = 20)
    ax.legend(loc = 2)
    ax.grid()
    plt.savefig('errors4.pdf')
    plt.show()


    return


def occupation(orbitals, N_fix, t, U, beta, max_iter, N_tol, non_local = False):

    def dyson(g, mu):
        return (g.inverse() - mu).inverse()
    
    def occ(g, mu):
        return dyson(g, mu).total_density().real
    

    tij = np.zeros([orbitals] * 2)
    for i in range(orbitals - 1):
        tij[i, i + 1] = -t
        tij[i + 1, i] = -t

    Vij = coulomb_matrix(orbitals, U, non_local = non_local)
    Vijkl = coulomb_tensor(orbitals, U, non_local = non_local)

    
    mu_list = np.linspace(-orbitals * t, orbitals * t, 300)
    occ_list_gw = np.zeros_like(mu_list)
    occ_list_g0 = np.zeros_like(mu_list)
    occ_list_ed = np.zeros_like(mu_list)
    for i, mu in enumerate(mu_list):
        print(f'Index {i + 1}/{len(mu_list)}', end = '\r')
        gw = GWSolver(generate_g0_w(tij, MeshDLRImFreq(beta = beta, statistic = 'Fermion',  w_max = 20.0, eps = 1e-12, symmetrize = True)), Vij,\
            self_interactions = False, hartree_flag = True, fock_flag = True, mu = mu, N_fix = False, N_tol = N_tol, max_iter = max_iter, full_mesh = False, verbose = False)
        
        occ_list_gw[i] = gw.g_w.total_density().real
        occ_list_g0[i] = gw.g0_w.total_density().real

        occ_list_ed[i] = ed_occupation(tij, Vijkl, beta, mu)
    print('\n')
    fig, ax = plt.subplots(1, figsize = (6, 6))
    # ax.xaxis.label.set_fontsize(20)

    ax.plot(mu_list, occ_list_gw, color = 'red', lw = 2, label = 'gw', zorder = 1)
    ax.plot(mu_list, occ_list_g0, color = 'blue', lw = 2, label = 'g0', zorder = 1)
    ax.plot(mu_list, occ_list_ed, color = 'green', lw = 2, label = 'ED', zorder = 1)
    ax.legend(fontsize = 13)
    ax.set_xlabel('Chemical Potential')
    ax.set_ylabel('Occupation')
    ax.set_xlim(-orbitals * t, orbitals * t)
    ax.set_ylim(0 - 0.05 * orbitals, 2 * orbitals + 0.05 * orbitals)
    if N_fix is not False:
        gw = GWSolver(generate_g0_w(tij, MeshDLRImFreq(beta = beta, statistic = 'Fermion',  w_max = 20.0, eps = 1e-12, symmetrize = True)), Vij,\
            self_interactions = False, hartree_flag = True, fock_flag = True, mu = 0, N_fix = N_fix, N_tol = N_tol, max_iter = max_iter, full_mesh = False)
        # print(gw.mu, gw.N(gw.g_w))

        ax.scatter(gw.mu0, gw.g_w.total_density().real, color = 'black', zorder = 2, s = 50)
        ax.scatter(gw.mu0, gw.g0_w.total_density().real, color = 'black', zorder = 2, s = 50)
        found_mu_ed, occ_ed = ed_mu(tij, Vijkl, beta, N_fix, N_tol)
        ax.scatter(found_mu_ed, occ_ed, color = 'black', zorder = 2, s = 50)
        
    ax.grid(zorder = 0)
    ax.set_axisbelow(True)



    plt.savefig('o4.pdf')

    plt.show()

    return


def occupation2(orbitals, N_fix, t, U, beta, max_iter, N_tol, non_local = False):


    tij = np.zeros([orbitals] * 2)
    for i in range(orbitals - 1):
        tij[i, i + 1] = -t
        tij[i + 1, i] = -t

    Vij = coulomb_matrix(orbitals, U, non_local = non_local)
    Vijkl = coulomb_tensor(orbitals, U, non_local = non_local)

    
    mu_list = np.linspace(-orbitals * t - 1, orbitals * t + 1, 500)
    occ_list_ed = np.zeros_like(mu_list)
    for i, mu in enumerate(mu_list):
        print(f'Index {i + 1}/{len(mu_list)}', end = '\r')
        occ_list_ed[i] = ed_occupation(tij, Vijkl, beta, mu)
    print('\n')
    fig, ax = plt.subplots(1, figsize = (6, 6))
    # ax.xaxis.label.set_fontsize(20)

    ax.plot(mu_list, occ_list_ed, color = 'blue', lw = 3, label = 'ED', zorder = 1)
    # ax.legend(fontsize = 13)
    ax.set_xlabel('Chemical Potential')
    ax.set_ylabel('Occupation')
    ax.set_xlim(np.min(mu_list), np.max(mu_list))
    ax.set_ylim(0 - 0.05 * orbitals, 2 * orbitals + 0.05 * orbitals)
        
    ax.grid(zorder = 0)
    ax.set_axisbelow(True)



    plt.savefig('o201.pdf')

    plt.show()

    return

def spectral(g_w):
    A = 0
    for name, g in g_w:
        A += -np.imag(np.trace(g.data[:, :, :], axis1 = 1, axis2 = 2))
    return A

def symmetric_difference(omega, value, A):
    w_index = np.argmin(np.abs(omega - value))

    left_sum = np.sum(A[:w_index])
    right_sum = np.sum(A[w_index:])
    return left_sum - right_sum

def find_center_freq(g):
    omega = np.array([w for w in g.mesh.values()])
    A = spectral(g)

    w_step = 5.0

    w = 0
    difference = symmetric_difference(omega, w, A)
    last_difference = 2 * difference

    previous_direction = None
    iter = 0
    while abs(difference) > 0.0001:
        if abs(difference - last_difference) < 1e-6:
            break

        iter += 1
        if difference < 0:
            if previous_direction == 'decrement':
                w_step /= 2
            previous_direction = 'increment'
            w += w_step
        else:
            if previous_direction == 'increment':
                w_step /= 2
            previous_direction = 'decrement'
            w -= w_step

        last_difference = difference
        difference = symmetric_difference(omega, w, A)
    
    return w

def offset(g_w, w, w_center_arg):
    return np.argmin(np.abs(w - find_center_freq(g_w))) - w_center_arg


def shift(orbitals, t, U, beta, mu1, mu2, non_local = False):

    N_tol = 1e-12

    tij = np.zeros([orbitals] * 2)
    for i in range(orbitals - 1):
        tij[i, i + 1] = -t
        tij[i + 1, i] = -t

    Vij = coulomb_matrix(orbitals, U, non_local = non_local)
    Vijkl = coulomb_tensor(orbitals , U, non_local = non_local)

    iw_mesh_f = MeshDLRImFreq(beta = beta, statistic = 'Fermion',  w_max = 20.0, eps = 1e-12, symmetrize = True)
    g0_w = generate_g0_w(tij, iw_mesh_f)
    ED, mu = exact_diag(tij, Vijkl, N_fix = False, beta = beta, nw = 1024, N_tol = N_tol, mu = mu1)
    ED1, mu = exact_diag(tij, Vijkl, N_fix = False, beta = beta, nw = 1024, N_tol = N_tol, mu = mu2)
    print(N(ED))
    print(N(ED1))


    w_mesh = MeshReFreq(window = (-10, 10), n_w = 10 * 2048)
    w = np.array([W for W in w_mesh.values()])

    w_min = -5
    w_max = -w_min

    w_min_arg = np.argmin(np.abs(w - w_min))
    w_max_arg = np.argmin(np.abs(w - w_max))
    w_center_arg = np.argmin(np.abs(w))


    G_w_temp = Gf(mesh = w_mesh, target_shape = [orbitals, orbitals])
    G_w = BlockGf(block_list = [G_w_temp] * 2, name_list = ['up', 'dn'])
    for block, g in G_w:
        G_w[block].set_from_pade(ED[block], n_points = 100, freq_offset = 0.15)

    g_w_temp = Gf(mesh = w_mesh, target_shape = [orbitals, orbitals])
    g_w = BlockGf(block_list = [g_w_temp] * 2, name_list = ['up', 'dn'])
    for block, g in G_w:
        g_w[block].set_from_pade(ED1[block], n_points = 100, freq_offset = 0.15)


    G_w_offset = 1 * offset(G_w, w, w_center_arg)
    g_w_offset = 1 * offset(g_w, w, w_center_arg) - 25

    fig, axs = plt.subplots(1, 2, figsize = (12, 6), tight_layout = True)
    axs[0].xaxis.label.set_fontsize(15)
    axs[0].yaxis.label.set_fontsize(15)
    axs[0].set_xlim(w_min, w_max)

    axs[1].xaxis.label.set_fontsize(15)
    axs[1].set_xlim(w_min, w_max)

    # ax.plot([w for w in G_w.mesh.values()][w_min_arg:w_max_arg], spectral(G_w)[w_min_arg + G_w_offset:w_max_arg + G_w_offset], color = 'black', zorder = 0)
    # ax.plot([w for w in g_w.mesh.values()][w_min_arg:w_max_arg], spectral(g_w)[w_min_arg + g_w_offset:w_max_arg + g_w_offset], color = 'red', zorder = 1, linestyle = '-')

    axs[0].plot([w for w in G_w.mesh.values()][w_min_arg:w_max_arg], spectral(G_w)[w_min_arg:w_max_arg], color='blue', alpha=1, linestyle='-', lw = 2, zorder = 0, label = f'μ = {mu1:.1f}')
    axs[0].plot([w for w in g_w.mesh.values()][w_min_arg:w_max_arg], spectral(g_w)[w_min_arg:w_max_arg], color='red', alpha=1, linestyle='-', lw = 2, zorder = 1, label = f'μ = {mu2:.1f}')

    axs[1].plot([w for w in G_w.mesh.values()][w_min_arg:w_max_arg], spectral(G_w)[w_min_arg + G_w_offset:w_max_arg + G_w_offset], color='blue', alpha=1, linestyle='-', lw = 2, zorder = 0, label = f'μ = {mu1:.1f}')
    axs[1].plot([w for w in g_w.mesh.values()][w_min_arg:w_max_arg], spectral(g_w)[w_min_arg + g_w_offset:w_max_arg + g_w_offset], color='red', alpha=1, linestyle='--', lw = 2, zorder = 1, label = f'μ = {mu2:.1f}')

    axs[0].legend()
    axs[1].legend()


    axs[0].set_ylabel('A(ω)')
    axs[0].set_xlabel('ω')
    axs[1].set_xlabel('ω')

    # plt.savefig('spec.pdf')
    plt.show()


def shift2(orbitals, N_fix, t, U, beta, max_iter, non_local = False):

    N_tol = 1e-5

    tij = np.zeros([orbitals] * 2)
    for i in range(orbitals - 1):
        tij[i, i + 1] = -t
        tij[i + 1, i] = -t

    Vij = coulomb_matrix(orbitals, U, non_local = non_local)
    Vijkl = coulomb_tensor(orbitals , U, non_local = non_local)

    iw_mesh_f = MeshDLRImFreq(beta = beta, statistic = 'Fermion',  w_max = 20.0, eps = 1e-12, symmetrize = True)
    g0_w = generate_g0_w(tij, iw_mesh_f)
    ED, mu = exact_diag(tij, Vijkl, N_fix = N_fix, beta = beta, nw = 1024, N_tol = N_tol, mu = 0)
    gw = GWSolver(g0_w, Vij, self_interactions = False, hartree_flag = True, fock_flag = True, N_fix = N_fix, max_iter = max_iter, full_mesh = True, N_tol = N_tol, mu_0 = 0.0, mu_gw = 0.0)
    print(N(ED))
    print(N(gw.g_w))


    w_mesh = MeshReFreq(window = (-10, 10), n_w = 10 * 2048)
    w = np.array([W for W in w_mesh.values()])

    w_min = -6
    w_max = -w_min

    w_min_arg = np.argmin(np.abs(w - w_min))
    w_max_arg = np.argmin(np.abs(w - w_max))
    w_center_arg = np.argmin(np.abs(w))


    G_w_temp = Gf(mesh = w_mesh, target_shape = [orbitals, orbitals])
    G_w = BlockGf(block_list = [G_w_temp] * 2, name_list = ['up', 'dn'])
    for block, g in G_w:
        G_w[block].set_from_pade(ED[block], n_points = 100, freq_offset = 0.15)

    g_w_temp = Gf(mesh = w_mesh, target_shape = [orbitals, orbitals])
    g_w = BlockGf(block_list = [g_w_temp] * 2, name_list = ['up', 'dn'])
    for block, g in G_w:
        g_w[block].set_from_pade(gw.g_w[block], n_points = 100, freq_offset = 0.15)


    G_w_offset = 1 * offset(G_w, w, w_center_arg)
    g_w_offset = 1 * offset(g_w, w, w_center_arg)

    fig, axs = plt.subplots(1, figsize = (6, 6), tight_layout = True)
    axs.xaxis.label.set_fontsize(15)
    axs.yaxis.label.set_fontsize(15)
    axs.set_xlim(w_min, w_max)
    # axs.set_ylim(0, 1)

    # axs[1].xaxis.label.set_fontsize(15)
    # axs[1].set_xlim(w_min, w_max)

    # ax.plot([w for w in G_w.mesh.values()][w_min_arg:w_max_arg], spectral(G_w)[w_min_arg + G_w_offset:w_max_arg + G_w_offset], color = 'black', zorder = 0)
    # ax.plot([w for w in g_w.mesh.values()][w_min_arg:w_max_arg], spectral(g_w)[w_min_arg + g_w_offset:w_max_arg + g_w_offset], color = 'red', zorder = 1, linestyle = '-')

    # axs[0].plot([w for w in G_w.mesh.values()][w_min_arg:w_max_arg], spectral(G_w)[w_min_arg:w_max_arg], color='blue', alpha=1, linestyle='-', lw = 2, zorder = 0, label = f'ED')
    # axs[0].plot([w for w in g_w.mesh.values()][w_min_arg:w_max_arg], spectral(g_w)[w_min_arg:w_max_arg], color='red', alpha=1, linestyle='-', lw = 2, zorder = 1, label = f'GW')

    axs.plot([w for w in G_w.mesh.values()][w_min_arg:w_max_arg], spectral(G_w)[w_min_arg + G_w_offset:w_max_arg + G_w_offset], color='blue', alpha=1, linestyle='-', lw = 2, zorder = 0, label = f'ED')
    axs.plot([w for w in g_w.mesh.values()][w_min_arg:w_max_arg], spectral(g_w)[w_min_arg + g_w_offset:w_max_arg + g_w_offset], color='red', alpha=0.9, linestyle='--', lw = 2, zorder = 1, label = f'GW')

    axs.legend()


    axs.set_ylabel('A(ω)')
    axs.set_xlabel('ω')

    axs.text(0.01, 0.95, f'Orbitals = {orbitals}', transform=axs.transAxes, fontsize=15, ha='left')
    axs.text(0.01, 0.9, f'N_fix = {N_fix}', transform=axs.transAxes, fontsize=15, ha='left')
    axs.text(0.01, 0.85, f'U/t = {U}', transform=axs.transAxes, fontsize=15, ha='left')
    axs.text(0.01, 0.8, f'beta = {beta}', transform=axs.transAxes, fontsize=15, ha='left')
    axs.text(0.01, 0.75, f'# iterations = {gw.iter_reached}', transform=axs.transAxes, fontsize=15, ha='left')
    


    left, bottom, width, height = [0.72, 0.3, 0.2, 0.2]  # Adjust these values as needed
    inset_ax = fig.add_axes([left, bottom, width, height])
    inset_ax.plot([w for w in G_w.mesh.values()][w_min_arg:w_max_arg], spectral(G_w)[w_min_arg + G_w_offset:w_max_arg + G_w_offset], color='blue', alpha=1, linestyle='-', lw = 2, zorder = 0)
    inset_ax.plot([w for w in g_w.mesh.values()][w_min_arg:w_max_arg], spectral(g_w)[w_min_arg + g_w_offset:w_max_arg + g_w_offset], color='red', alpha=0.9, linestyle='--', lw = 2, zorder = 1)

    # Set limits for the inset axes
    inset_ax.set_xlim(2.2, 3.7)  # Limit x-axis to zoom in
    inset_ax.set_ylim(0, 0.3)
    inset_ax.set_xlabel('ω')  # Limit y-axis to zoom in

    plt.savefig('oneshot.pdf')
    plt.show()


def main():
    shift2(orbitals = 2, N_fix = 2, t = 1.0, U = 0.75, beta = 100, max_iter = 1, non_local = False)
    return

if __name__ == '__main__':
    main()