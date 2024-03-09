import numpy as np
import matplotlib.pyplot as plt
import time
import os
import multiprocessing

from gwsolvercpp import GWSolverCPP
from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq, MeshDLRImTime

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

def main():
    orbitals = 200
    t = 1.0

    tij = np.zeros([orbitals] * 2)
    for i in range(orbitals - 1):
        tij[i, i + 1] = -t
        tij[i + 1, i] = -t

    U = 0.1
    V = coulomb_matrix(orbitals, U)

    beta = 100

    mesh = MeshDLRImFreq(beta = beta, statistic = 'Fermion', w_max = 1.0, eps = 1e-12, symmetrize = True)
    print(f"Number of orbitals indices = {orbitals}")
    print(f"Number of DLR mesh points = {len(list(mesh.values()))}")
    G_w = generate_g0_w(tij, mesh)

    available_cores = multiprocessing.cpu_count() // get_threads_per_core()
    execution_times = []
    for num_cores in range(1, available_cores + 1):
        start = time.perf_counter()
        gw = GWSolverCPP(G_w, V, True, True, True, orbitals, num_cores)
        execution_time = time.perf_counter() - start
        print(f"Execution time with {num_cores} cores: {execution_time:.2f}")
        execution_times.append(execution_time)

    fig, ax = plt.subplots(figsize = (6, 6), tight_layout = True)
    ax.plot(range(1, available_cores + 1), execution_times, marker = 'o', color = 'red')
    ax.set_xlim(0.5, available_cores + 0.5)
    # ax.set_yscale('log')
    ax.set_ylim(0, 1.1 * np.max(execution_times))
    ax.set_xlabel('Number of cores')
    ax.set_ylabel('Execution time (s)')
    ax.grid()
    ax.set_xticks(range(1, available_cores + 1))


    ax.text(0.2, 0.85, f"Number of orbitals indices = {orbitals}", horizontalalignment = 'left', verticalalignment = 'center', transform = ax.transAxes, fontsize = 15)
    ax.text(0.2, 0.8, f"Number of DLR mesh points = {len(list(mesh.values()))}", horizontalalignment = 'left', verticalalignment = 'center', transform = ax.transAxes, fontsize = 15)
    plt.savefig("test.pdf")

if __name__ == "__main__":
    main()
    


