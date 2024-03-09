import numpy as np

from triqs.gf import Gf, MeshImFreq, Idx
from triqs.gf import inverse, iOmega_n
from triqs.lattice.tight_binding import TBLattice


from triqs_tprf.lattice import lattice_dyson_g0_wk
from triqs_tprf.lattice import lattice_dyson_g_wk

from triqs_tprf.lattice import fourier_wk_to_wr
from triqs_tprf.lattice import chi_wr_from_chi_wk

from triqs_tprf.gw_solver import GWSolver

from hugoDimer import GWHubbardDimer
from casperDimer import GWA

from itertools import product


def polarizationCheck(myPolarization, hugoPolarization):
    np.testing.assert_array_almost_equal(hugoPolarization[:, Idx(0, 0, 0)][0,0,0,0].data, myPolarization['up'].data[:, 0, 0, 0, 0])
    np.testing.assert_array_almost_equal(hugoPolarization[:, Idx(1, 0, 0)][0,0,0,0].data, myPolarization['up'].data[:, 1, 1, 0, 0])
    
def screenedPotentialCheck(myPotential, hugoPotential):
    np.testing.assert_array_almost_equal(hugoPotential[:, Idx(0, 0, 0)][0,0,0,0].data, myPotential['up'].data[:, 0, 0, 0, 0])
    np.testing.assert_array_almost_equal(hugoPotential[:, Idx(1, 0, 0)][0,0,0,0].data, myPotential['up'].data[:, 1, 1, 0, 0])

def selfEnergyCheck(myEnergy, hugoEnergy):
    np.testing.assert_array_almost_equal(hugoEnergy[:, Idx(0, 0, 0)][0,0].data, myEnergy['up'].data[:, 0, 0])
    np.testing.assert_array_almost_equal(hugoEnergy[:, Idx(1, 0, 0)][0,0].data, myEnergy['up'].data[:, 0, 1])


def compareFlags(hartree_flag, fock_flag, spin_flag):
    beta = 20.0
    U = 1.5
    t = 1.0
    nw = 1024
    
    hugo = GWHubbardDimer(
        beta = beta, U = U, 
        t = t, nw = nw, maxiter = 1,
        self_interaction = True, spinless = spin_flag,
        hartree_flag = hartree_flag, fock_flag = fock_flag)
    
    casper = GWA(
        beta = beta, U = U, 
        t = t, nw = nw, spinless = spin_flag, static_flag = False, 
        hartree_flag = hartree_flag, fock_flag = fock_flag)

    print("Hartree flag: {}".format(hartree_flag))
    print("Fock flag: {}".format(fock_flag))
    print("Spin flag: {}".format(spin_flag)) 

    polarizationCheck(casper.P_w, hugo.P_wr)
    print("Polarization check: Passed")
    
    screenedPotentialCheck(casper.W_w, hugo.W_wr)
    print("Screened Potential check: Passed")

    selfEnergyCheck(casper.sigma_w, hugo.sigma_wr)
    print("Self-Energy check: Passed")


flags = [True, False]
for hartree_flag, fock_flag, spin_flag in product(flags, flags, flags):
    compareFlags(hartree_flag = hartree_flag, fock_flag = fock_flag, spin_flag = spin_flag)










##########################################
