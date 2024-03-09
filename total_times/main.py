import numpy as np
from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq, MeshDLRImTime
import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import time
import multiprocessing
from triqs_tprf.lattice import inv

from gwsolver import GWSolver

def get_cpu_info():
    cpu_info = {}
    with open('/proc/cpuinfo', 'r') as cpuinfo:
        for line in cpuinfo:
            if line.strip():
                key, value = line.strip().split(':')
                cpu_info[key.strip()] = value.strip()
    return cpu_info

def get_threads_per_core():
    cpu_info = get_cpu_info()
    siblings = int(cpu_info.get('siblings', 1))
    cores = int(cpu_info.get('cpu cores', 1))
    threads_per_core = siblings // cores
    return threads_per_core


def generate_G(tij, mesh, spin_names = ['up', 'dn']):
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


def speed_up(array):
    base = array[0]
    return base / array

def seconds_to_hms(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return hours, minutes, seconds

def val_to_perc(val):
    perc = 100 * val
    if round(perc, 0) == 100:
        return int(perc)
    return round(perc, 1)

def main():
    orbitals = 700
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
    gw = GWSolver(G_iw, V, self_interactions = False, hartree_flag = True, fock_flag = True, mu = 0, N_fix = N_fix, max_iter = 25, number_of_cores = 8)
    tot = time.perf_counter() - start


    fig, ax = plt.subplots(figsize = (4, 4), tight_layout = True)

    # ax.set_xlabel('Number of cores')
    # ax.set_ylabel('Speed-up time')
    # ax.grid()
    # ax.set_axisbelow(True)
    # ax.legend()
    # ax.set_xticks(range(1, available_cores + 1))

    categories = np.array(['Full Solver', 'G0', 'P', 'W', 'Σ', 'Hartree', 'Fock', 'GW', 'Other'])[::-1]
    colors = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'teal'])[::-1]

    other = tot - gw.g0_w_time - gw.P_w_time - gw.W_w_time - gw.sigma_w_time - gw.hartree_time - gw.fock_time - gw.g_w_time
    values = (np.array([tot, gw.g0_w_time, gw.P_w_time, gw.W_w_time, gw.sigma_w_time, gw.hartree_time, gw.fock_time, gw.g_w_time, other]) / tot)[::-1]

    sorted_indices = sorted(range(len(values)), key=lambda k: values[k])
    values_sorted = [values[i] for i in sorted_indices]
    categories_sorted = [categories[i] for i in sorted_indices]
    colors_sorted = [colors[i] for i in sorted_indices]

    bars = ax.barh(categories_sorted, values_sorted, color = colors_sorted)
    ax.set_xlim(0, 1.25)
    for bar, value in zip(bars, values_sorted):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, ' ' + f'{val_to_perc(value)}%', ha='left', va='center')
    ax.set_xticks([])

    ax.text(0.95, 0.35, f"Orbitals = {orbitals}", transform = ax.transAxes, fontsize = 10, ha = 'right')
    ax.text(0.95, 0.3, f"U / t = {U}", transform = ax.transAxes, fontsize = 10, ha = 'right')
    ax.text(0.95, 0.25, f"beta = {beta}", transform = ax.transAxes, fontsize = 10, ha = 'right')
    ax.text(0.95, 0.2, f"# DLR points = {len(iw_mesh)}", transform = ax.transAxes, fontsize = 10, ha = 'right')
    ax.text(0.95, 0.15, f"Iteration(s) = {gw.iter_reached}", transform = ax.transAxes, fontsize = 10, ha = 'right')
    ax.text(0.95, 0.1, f"N_fix = {N_fix}", transform = ax.transAxes, fontsize = 10, ha = 'right')
    h, m, s = seconds_to_hms(tot)
    ax.text(0.95, 0.05, f"Execution time = {int(h)}h {int(m)}m {round(s, 2)}s", transform = ax.transAxes, fontsize = 10, ha = 'right')

    plt.savefig("test.pdf")

    return

if __name__ == '__main__':
    main()