import numpy as np


from triqs.gf import Gf, MeshImFreq, Idx
from triqs.gf import inverse, iOmega_n
from triqs.lattice.tight_binding import TBLattice


from triqs_tprf.lattice import lattice_dyson_g0_wk
from triqs_tprf.lattice import lattice_dyson_g_wk

from triqs_tprf.lattice import fourier_wk_to_wr
from triqs_tprf.lattice import chi_wr_from_chi_wk

from triqs_tprf.gw_solver import GWSolver

class GWHubbardDimer:

    def __init__(
            self,
            beta=20.0, U=1.5, t=1.0, mu=0.0, nw = 5 * 1024, maxiter=100,
            self_interaction=False, spinless=False,
            gw_flag=True, hartree_flag=False, fock_flag=False,
            N_fix = False):
        
        wmesh = MeshImFreq(beta, 'Fermion', nw)

        if spinless:
            tb_opts = dict(
                units = [(1, 0, 0)],
                orbital_positions = [(0,0,0)],
                orbital_names = ['0'],
                )
            I = np.eye(1)
            
        else:
            tb_opts = dict(
                units = [(1, 0, 0)],
                orbital_positions = [(0,0,0)] * 2,
                orbital_names = ['up_0', 'do_0'],
                )
            I = np.eye(2)

        # Have to use t/2 hopping in the TBLattice since it considers
        # hoppings in both directions, which doubles the total hopping
        # for the Hubbard dimer.

        H_r = TBLattice(hopping = {
            (+1,): -0.5 * t * I,
            (-1,): -0.5 * t * I,
            }, **tb_opts)

        kmesh = H_r.get_kmesh(n_k=(2, 1, 1))
        self.e_k = H_r.fourier(kmesh)

        if self_interaction:
            
            V_aaaa = np.zeros((2, 2, 2, 2))

            V_aaaa[0, 0, 0, 0] = U
            V_aaaa[1, 1, 1, 1] = U
            
            V_aaaa[1, 1, 0, 0] = U
            V_aaaa[0, 0, 1, 1] = U

            self.V_aaaa = V_aaaa
            
        if spinless:

            V_aaaa = np.zeros((1, 1, 1, 1))
            V_aaaa[0, 0, 0, 0] = U
            self.V_aaaa = V_aaaa
            
        if not spinless and not self_interaction:
            
            from triqs.operators import n, c, c_dag, Operator, dagger
            from triqs_tprf.gw import get_gw_tensor
            
            self.H_int = U * n('up',0) * n('do',0)
            self.fundamental_operators = [c('up', 0), c('do', 0)]
            self.V_aaaa = get_gw_tensor(self.H_int, self.fundamental_operators)
        
        self.V_k = Gf(mesh=kmesh, target_shape=self.V_aaaa.shape)
        self.V_k.data[:] = self.V_aaaa    

        gw = GWSolver(self.e_k, self.V_k, wmesh, mu=mu, N_fix = N_fix)
        gw.solve_iter(
            maxiter=maxiter,
            gw=gw_flag, hartree=hartree_flag, fock=fock_flag,
            spinless=spinless)
        gw.calc_real_space()
        
        self.gw = gw

        for key, val in gw.__dict__.items():
            setattr(self, key, val)