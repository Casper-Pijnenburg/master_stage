import numpy as np
from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq, MeshDLRImTime
import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import time
import multiprocessing
from threadpoolctl import threadpool_limits


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
    orbitals = 7000
    t = 1.0



    tij = np.zeros([orbitals] * 2)
    for i in range(orbitals - 1):
        tij[i, i + 1] = -t
        tij[i + 1, i] = -t



    available_cores = multiprocessing.cpu_count() // get_threads_per_core()
    tot_times = np.zeros(available_cores)

    for num_cores in range(available_cores, 0, -1):
        with threadpool_limits(limits=num_cores, user_api='blas'):
            start = time.perf_counter()
            T = np.dot(tij, tij)

            
            tot_times[num_cores - 1] = time.perf_counter() - start




            
    fig, ax = plt.subplots(figsize = (6, 6), tight_layout = True)
    ax.plot(range(1, available_cores + 1), range(1, available_cores + 1), marker = 'o', color = 'black', label = 'Perfect Speed-up', zorder = 1)
    ax.plot(range(1, available_cores + 1), speed_up(tot_times), marker = 'o', color = 'blue', label = 'Full Solver', zorder = 2)

    ax.set_xlim(0.5, available_cores + 0.5)

    # ax.set_title(f'Speed-up test with {orbitals} orbitals and {len(iw_mesh)} DLR points,\nconvergence reached in {gw.iter_reached} iteration(s) and N_fix = {N_fix}')

    ax.set_xlabel('Number of cores')
    ax.set_ylabel('Speed-up time')
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_xticks(range(1, available_cores + 1))

    plt.savefig("test.pdf")

    return

if __name__ == '__main__':
    main()