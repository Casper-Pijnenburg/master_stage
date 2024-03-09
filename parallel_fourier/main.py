import numpy as np
from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq, MeshDLRImTime
import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import time
import multiprocessing

from triqs_tprf.lattice import iw_to_tau
from triqs_tprf.lattice import iw_to_tau_p
from triqs_tprf.lattice import iw_to_tau_p2
from triqs_tprf.lattice import tau_to_iw
from triqs_tprf.lattice import tau_to_iw_p
from triqs_tprf.lattice import inv

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


def speed_up(array):
    base = array[0]
    return base / array

def main():
    orbitals = 2000
    t = 1.0
    beta = 100


    tij = np.zeros([orbitals] * 2)
    for i in range(orbitals - 1):
        tij[i, i + 1] = -t
        tij[i + 1, i] = -t


    mesh = MeshDLRImFreq(beta = beta, statistic = 'Fermion',  w_max = 5.0, eps = 1e-12, symmetrize = True)
    bmesh = MeshDLRImFreq(beta = beta, statistic = 'Boson',  w_max = 5.0, eps = 1e-12, symmetrize = True)
    bmesh_t = MeshDLRImTime(beta = beta, statistic = 'Boson',  w_max = 5.0, eps = 1e-12, symmetrize = True)


    G_w = generate_G(tij, mesh)

    available_cores = multiprocessing.cpu_count() // get_threads_per_core()
    row = np.zeros(available_cores)
    scalar = np.zeros(available_cores)
    for num_cores in range(available_cores, 0, -1):
        start = time.perf_counter()
        G_t = iw_to_tau_p(G_w, num_cores)
        row[num_cores - 1] = time.perf_counter() - start
        start = time.perf_counter()
        G_t = iw_to_tau_p2(G_w, num_cores)
        scalar[num_cores - 1] = time.perf_counter() - start




    fig, ax = plt.subplots(figsize = (6, 6), tight_layout = True)
    ax.plot(range(1, available_cores + 1), range(1, available_cores + 1), marker = 'o', color = 'black', label = 'Perfect Speed-up', zorder = 1)
    ax.plot(range(1, available_cores + 1), speed_up(row), marker = 'o', color = 'blue', label = 'Row Fourier', zorder = 2)
    ax.plot(range(1, available_cores + 1), speed_up(scalar), marker = 'o', color = 'red', label = 'Scalar Fourier', zorder = 2)

    ax.set_xlim(0.5, available_cores + 0.5)

    # ax.set_title(f'Speed-up test with {orbitals} orbitals and {len(iw_mesh)} DLR points,\nconvergence reached in {gw.iter_reached} iteration(s) and N_fix = {N_fix}')

    ax.set_xlabel('Number of cores')
    ax.set_ylabel('Speed-up time')
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_xticks(range(1, available_cores + 1))

    gs = GridSpec(4, 2, figure=fig)
    ax_sub = fig.add_subplot(gs[3, 1])
    categories = np.array(['Row', 'Scalar'])[::-1]
    colors = np.array(['blue', 'red'])[::-1]
    abs_values = (np.array([row[-1], scalar[-1]]))[::-1]
    values = (np.array([row[-1], scalar[-1]]) / scalar[-1])[::-1]

    sorted_indices = sorted(range(len(values)), key=lambda k: values[k])
    values_sorted = [values[i] for i in sorted_indices]
    abs_sorted = [abs_values[i] for i in sorted_indices]
    categories_sorted = [categories[i] for i in sorted_indices]
    colors_sorted = [colors[i] for i in sorted_indices]

    bars = ax_sub.barh(categories_sorted, values_sorted, color = colors_sorted)
    ax_sub.set_xlim(0, 1.275)
    for bar, value in zip(bars, abs_sorted):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, ' ' + f'{round(value, 2)} s', ha='left', va='center')
    ax_sub.set_xticks([])
    ax_sub.set_title('Execution time 8 cores')


    plt.savefig("test.pdf")

    return

if __name__ == '__main__':
    main()