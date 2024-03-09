import numpy as np
from threadpoolctl import threadpool_limits

from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq
from triqs.gf import make_gf_dlr, make_gf_dlr_imtime, make_gf_dlr_imfreq

from triqs_tprf.lattice import polarization
from triqs_tprf.lattice import screened_potential
from triqs_tprf.lattice import dyn_self_energy
from triqs_tprf.lattice import hartree_self_energy
from triqs_tprf.lattice import fock_self_energy
from triqs_tprf.lattice import dyson_mu
from triqs_tprf.lattice import dyson_mu_sigma
from triqs_tprf.lattice import total_density
from triqs_tprf.lattice import inv

import time

from itertools import product
        
class GWSolver():
    def __init__(self,
                 g0_w, V, self_interactions = False,
                 hartree_flag = True, fock_flag = True,
                 mu = 0, N_fix = False, N_tol = 1e-6, max_iter = 1, full_mesh = False, number_of_cores = 8, verbose = False):
        
        self.self_interactions, self.hartree_flag, self.fock_flag = self_interactions, hartree_flag, fock_flag
        self.N_fix, self.N_tol = N_fix, N_tol
        self.mu = mu
        self.full_mesh = full_mesh
        self.verbose = verbose
        self.number_of_cores = number_of_cores

        self.blocks = [name for name, g in g0_w]
        self.target_shape = g0_w[self.blocks[0]][Idx(0)].shape
        start = time.perf_counter()
        self.g0_w, self.mu0 = self.dyson_equation(g0_w, self.mu, sigma_w = None, N_fix = self.N_fix)
        self.g0_w_time = time.perf_counter() - start
        # print(f'g0_w done is {time.perf_counter() - start} s')

        if self.verbose:
            print(f'g0_w occupation: {self.N(self.g0_w)} at chemical potential {self.mu0}')


        self.fmesh = self.g0_w.mesh
        self.bmesh = MeshDLRImFreq(self.fmesh.beta, 'Boson', w_max = self.fmesh.w_max, eps = self.fmesh.eps, symmetrize = True)
 
        self.V = V


        self.g_w = self.g0_w.copy()
        self.sigma_w = self.g0_w.copy()
        self.g_w_old = self.g_w.copy()

        self.hartree_time = 0.0
        self.fock_time = 0.0
        self.P_w_time = 0.0
        self.W_w_time = 0.0
        self.sigma_w_time = 0.0
        self.g_w_time = 0.0



        for iter in range(max_iter):
            print(f'Iteration {iter + 1} using {self.number_of_cores} cores  ', end = '\r')
            self.sigma_w.zero()

            start = time.perf_counter()
            self.P_w = polarization(self.g_w, self.bmesh, self.number_of_cores)
            self.P_w_time += time.perf_counter() - start
            # print(f'P_w done in {time.perf_counter() - start} s in iteration {iter}')

            start = time.perf_counter()
            self.W_w = self.screened_potential(self.P_w, self.V, self.self_interactions, self.number_of_cores)
            self.W_w_time += time.perf_counter() - start
            # print(f'W_w done is {time.perf_counter() - start} s in iteration {iter}')

            start = time.perf_counter()
            self.sigma_w += dyn_self_energy(self.g_w, self.W_w, V, self.self_interactions, self.number_of_cores)
            self.sigma_w_time += time.perf_counter() - start
            # print(f'sigma_w done is {time.perf_counter() - start} s in iteration {iter}')

            # if self.hartree_flag:
            start = time.perf_counter()
            self.sigma_w += hartree_self_energy(self.g_w, self.V, self.self_interactions, self.number_of_cores)
            self.hartree_time += time.perf_counter() - start
            # print(f'hartree done is {time.perf_counter() - start} s in iteration {iter}')
            # if self.fock_flag:
            start = time.perf_counter()
            self.sigma_w += fock_self_energy(self.g_w, self.V, self.self_interactions, self.number_of_cores)
            self.fock_time += time.perf_counter() - start
            # print(f'fock done is {time.perf_counter() - start} s in iteration {iter}')

            start = time.perf_counter()
            self.g_w, self.mu = self.dyson_equation(self.g0_w, 0.0, sigma_w = self.sigma_w, N_fix = False)
            self.g_w_time += time.perf_counter() - start
            # print(f'g_w done is {time.perf_counter() - start} s in iteration {iter}')
            # print(f'g_w occupation: {self.N(self.g_w)} at chemical potential {self.mu}')

            diff = np.max(np.abs((self.g_w - self.g_w_old)['up'].data))
            self.g_w_old = self.g_w.copy()
            if diff < 1e-7:
                self.iter_reached = iter + 1
                if self.verbose:
                    print('\n')
                    print(f'Convergence reached at iteration {iter}')
                break

            if iter == max_iter - 1:
                self.iter_reached = iter + 1
                # print('\n')
                # print("WARNING: Maximum iterations reached")

        # start = time.perf_counter()
        # self.g_w, self.mu = self.dyson_equation(self.g_w, 0.0, sigma_w = None, N_fix = N_fix)
        # self.g_w_time += time.perf_counter() - start

        if self.verbose:
            print(f'g_w occupation: {self.N(self.g_w)} at chemical potential {self.mu}')
        if self.full_mesh:
            n_iw = 1024

            g_dlr = make_gf_dlr(self.g_w)
            self.g_w = make_gf_imfreq(g_dlr, n_iw)

            g0_dlr = make_gf_dlr(self.g0_w)
            self.g0_w = make_gf_imfreq(g0_dlr, n_iw)

    def screened_potential(self, P_w, V, self_interactions, cores):
        with threadpool_limits(limits=cores, user_api='blas'):    
            V_t = V.copy()

            if not self_interactions:
                np.fill_diagonal(V_t, 0)

            I = np.eye(len(V))

            A = I - V_t * P_w['up']
            B =  - V * P_w['dn']
            C =  - V * P_w['up']
            D = I - V_t * P_w['dn']

            A_inv = inv(A, cores)

            S = inv(D - C * A_inv * B, cores)

            P_w['up'] = (A_inv + A_inv * B * S * C * A_inv) * V_t - A_inv * B * S * V;
            P_w['dn'] = -S * C * A_inv * V + S * V_t;
        
        return P_w

    def N(self, g_w):
        return total_density(g_w, self.number_of_cores)
        
    def _dyson_dispatch(self, g_w, mu, sigma_w = None):
        if sigma_w is not None:
            return dyson_mu_sigma(g_w, mu, sigma_w, self.number_of_cores)
        return dyson_mu(g_w, mu, self.number_of_cores)
    

    def dyson_equation(self, g_w, mu, sigma_w = None, N_fix = False):
        if not N_fix:
            if mu == 0 and sigma_w is None:
                return g_w, mu
            return self._dyson_dispatch(g_w, mu, sigma_w), mu
        else:
            step = 1.0
            previous_direction = None

            # occupation = self.N(self._dyson_dispatch(g_w, mu, sigma_w))
            occupation = self.N(g_w)
            while abs(occupation - N_fix) > self.N_tol:
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
                
                occupation = self.N(self._dyson_dispatch(g_w, mu, sigma_w))
                

            return self._dyson_dispatch(g_w, mu, sigma_w), mu

    def dyson_equation2(self, g_w, mu, sigma_w = None, N_fix = False):
        if not N_fix:
            return self._dyson_dispatch(g_w, mu, sigma_w), mu
        else:
            def target_function(mu):
                g = self._dyson_dispatch(g_w, mu, sigma_w)
                # print(f'occupation: {self.N(g)}, mu: {mu}')
                return self.N(g) - N_fix

            from scipy.optimize import root_scalar

            sol = root_scalar(target_function, method='brentq', bracket= np.array([-self.target_shape[0], self.target_shape[0]]), rtol=self.N_tol)
            mu = sol.root
            return self._dyson_dispatch(g_w, mu, sigma_w), mu
    
