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
            Vij[i, j] = round(1.0 * U / (abs(i - j) + 1), 2)

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

    available_cores = multiprocessing.cpu_count() // get_threads_per_core()
    tot_times = np.zeros(available_cores)
    g0_w_times = np.zeros(available_cores)
    P_w_times = np.zeros(available_cores)
    W_w_times = np.zeros(available_cores)
    sigma_w_times = np.zeros(available_cores)
    hartree_times = np.zeros(available_cores)
    fock_times = np.zeros(available_cores)
    g_w_times = np.zeros(available_cores)

    printed = False

    for num_cores in range(available_cores, 0, -1):

        start = time.perf_counter()
        gw = GWSolver(G_iw, V, self_interactions = False, hartree_flag = True, fock_flag = True, mu = 0, N_fix = N_fix, max_iter = 25, number_of_cores = num_cores)
        tot_times[num_cores - 1] = time.perf_counter() - start
        g0_w_times[num_cores - 1] = gw.g0_w_time
        P_w_times[num_cores - 1] = gw.P_w_time
        W_w_times[num_cores - 1] = gw.W_w_time
        sigma_w_times[num_cores - 1] = gw.sigma_w_time
        hartree_times[num_cores - 1] = gw.hartree_time
        fock_times[num_cores - 1] = gw.fock_time
        g_w_times[num_cores - 1] = gw.g_w_time

        if not printed:
            h, m, s = seconds_to_hms(16 * tot_times[-1])
            print(f"Estimated time = {int(h)}h {int(m)}m {round(s, 2)}s")
            printed = True


    fig, ax = plt.subplots(figsize = (6, 6), tight_layout = True)
    ax.plot(range(1, available_cores + 1), range(1, available_cores + 1), marker = 'o', color = 'black', label = 'Perfect Speed-up', zorder = 1)
    ax.plot(range(1, available_cores + 1), speed_up(tot_times), marker = 'o', color = 'blue', label = 'Full Solver', zorder = 2)
    ax.plot(range(1, available_cores + 1), speed_up(g0_w_times), marker = 'o', color = 'orange', label = 'G0 (Dyson Chem. pot. fix)', zorder = 1)
    ax.plot(range(1, available_cores + 1), speed_up(P_w_times), marker = 'o', color = 'green', label = 'Polarization', zorder = 1)
    ax.plot(range(1, available_cores + 1), speed_up(W_w_times), marker = 'o', color = 'red', label = 'Screened Potential', zorder = 1)
    ax.plot(range(1, available_cores + 1), speed_up(sigma_w_times), marker = 'o', color = 'purple', label = 'Dynamical self-energy', zorder = 1)
    ax.plot(range(1, available_cores + 1), speed_up(hartree_times), marker = 'o', color = 'brown', label = 'Hartree self-energy', zorder = 1)
    ax.plot(range(1, available_cores + 1), speed_up(fock_times), marker = 'o', color = 'pink', label = 'Fock self-energy', zorder = 1)
    ax.plot(range(1, available_cores + 1), speed_up(g_w_times), marker = 'o', color = 'gray', label = 'GW (Dyson)', zorder = 1)

    ax.set_xlim(0.5, available_cores + 0.5)

    # ax.set_title(f'Speed-up test with {orbitals} orbitals and {len(iw_mesh)} DLR points,\nconvergence reached in {gw.iter_reached} iteration(s) and N_fix = {N_fix}')

    ax.set_xlabel('Number of cores')
    ax.set_ylabel('Speed-up time')
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_xticks(range(1, available_cores + 1))

    gs = GridSpec(3, 2, figure=fig)
    ax_sub = fig.add_subplot(gs[2, 1])
    categories = np.array(['Full Solver', 'G0', 'P', 'W', 'Σ', 'Hartree', 'Fock', 'GW', 'Other'])[::-1]
    colors = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'teal'])[::-1]

    tot = tot_times[-1]
    other = tot - g0_w_times[-1] - P_w_times[-1] - W_w_times[-1] - sigma_w_times[-1] - hartree_times[-1] - fock_times[-1] - g_w_times[-1]
    values = (np.array([tot_times[-1], g0_w_times[-1], P_w_times[-1], W_w_times[-1], sigma_w_times[-1], hartree_times[-1], fock_times[-1], g_w_times[-1], other]) / tot)[::-1]

    sorted_indices = sorted(range(len(values)), key=lambda k: values[k])
    values_sorted = [values[i] for i in sorted_indices]
    categories_sorted = [categories[i] for i in sorted_indices]
    colors_sorted = [colors[i] for i in sorted_indices]

    bars = ax_sub.barh(categories_sorted, values_sorted, color = colors_sorted)
    ax_sub.set_xlim(0, 1.275)
    for bar, value in zip(bars, values_sorted):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, ' ' + f'{val_to_perc(value)}%', ha='left', va='center')
    ax_sub.set_xticks([])

    ax_sub.text(0.97, 0.55, f"Orbitals = {orbitals}", transform = ax_sub.transAxes, fontsize = 10, ha = 'right')
    ax_sub.text(0.97, 0.45, f"U / t = {U}", transform = ax_sub.transAxes, fontsize = 10, ha = 'right')
    ax_sub.text(0.97, 0.35, f"beta = {beta}", transform = ax_sub.transAxes, fontsize = 10, ha = 'right')
    ax_sub.text(0.97, 0.25, f"# DLR points = {len(iw_mesh)}", transform = ax_sub.transAxes, fontsize = 10, ha = 'right')
    ax_sub.text(0.97, 0.15, f"Iteration(s) = {gw.iter_reached}", transform = ax_sub.transAxes, fontsize = 10, ha = 'right')
    ax_sub.text(0.97, 0.05, f"N_fix = {N_fix}", transform = ax_sub.transAxes, fontsize = 10, ha = 'right')
    # h, m, s = seconds_to_hms(tot_times[0])
    # ax_sub.text(0.95, 0.05, f"Time = {int(h)}h {int(m)}m {round(s, 2)}s", transform = ax_sub.transAxes, fontsize = 10, ha = 'right')

    plt.savefig("test.pdf")

    return

if __name__ == '__main__':
    main()