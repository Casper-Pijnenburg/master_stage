import numpy as np

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

from itertools import product
        
class GWSolver():
    def __init__(self,
                 g0_w, V, self_interactions = False,
                 hartree_flag = True, fock_flag = True,
                 N_fix = False, N_tol = 1e-5, max_iter = 1, mu_0 = 0.0, mu_gw = 0.0, full_mesh = False, verbose = False):
        

        self.self_interactions, self.hartree_flag, self.fock_flag = self_interactions, hartree_flag, fock_flag
        self.N_fix, self.N_tol = N_fix, N_tol
        self.mu_0 = mu_0
        self.mu_gw = mu_gw
        self.full_mesh = full_mesh
        self.verbose = verbose

        

        self.blocks = [name for name, g in g0_w]
        self.target_shape = g0_w[self.blocks[0]][Idx(0)].shape

        self.g0_w, self.mu_0 = self.dyson_equation(g0_w, self.mu_0, sigma_w = None, N_fix = self.N_fix)
        if self.verbose:
            print(f'g0_w occupation: {self.N(self.g0_w)} at chemical potential {self.mu0}')


        self.fmesh = self.g0_w.mesh
        self.bmesh = MeshDLRImFreq(self.fmesh.beta, 'Boson', w_max = self.fmesh.w_max, eps = self.fmesh.eps, symmetrize = True)
 
        self.V = V


        self.g_w = self.g0_w.copy()
        self.sigma_w = self.g0_w.copy()
        self.g_w_old = self.g_w.copy()


        for iter in range(max_iter):
            print(f'Iteration {iter + 1}/{max_iter}', end = '\r')
            self.sigma_w.zero()

            if self.hartree_flag:
                self.sigma_w += hartree_self_energy(self.g_w, self.V, self.self_interactions, 8)
            
            if self.fock_flag:
                self.sigma_w += fock_self_energy(self.g_w, self.V, self.self_interactions, 8)

            self.P_w = polarization(self.g_w, self.bmesh, 8)
            self.W_w = screened_potential(self.P_w, self.V, self.self_interactions, 8)
            self.sigma_w += dyn_self_energy(self.g_w, self.W_w, V, self.self_interactions, 8)

            self.g_w, self.mu_gw = self.dyson_equation(self.g0_w, 0.0, sigma_w = self.sigma_w, N_fix = False)
            # print(f'g_w occupation: {self.N(self.g_w)} at chemical potential {self.mu}')

            diff = np.max(np.abs((self.g_w - self.g_w_old)['up'].data))
            self.g_w_old = self.g_w.copy()
            if diff < 1e-7:
                self.iter_reached = iter
                if self.verbose:
                    print(f'Convergence reached at iteration {iter}')
                break

            if iter == max_iter - 1:
                self.iter_reached = iter
                # if self.verbose:   
                print("WARNING: Maximum iterations reached")

        self.g_w, self.mu_gw = self.dyson_equation(self.g_w, 0.0, sigma_w = None, N_fix = self.N_fix)
        if self.verbose:
            print(f'g_w occupation: {self.N(self.g_w)} at chemical potential {self.mu}')
        if self.full_mesh:
            n_iw = 1024

            g_dlr = make_gf_dlr(self.g_w)
            self.g_w = make_gf_imfreq(g_dlr, n_iw)

            g0_dlr = make_gf_dlr(self.g0_w)
            self.g0_w = make_gf_imfreq(g0_dlr, n_iw)
  

    def screened_potential(self, P_w, V, self_interactions): 
        V_t = V.copy()

        if not self_interactions:
            np.fill_diagonal(V_t, 0)

        I = np.eye(len(V))

        A = I - V_t * P_w['up']
        B =   -   V * P_w['dn']
        C =   -   V * P_w['up']
        D = I - V_t * P_w['dn']

        A_inv = inv(A, 8)

        S = inv(D - C * A_inv * B, 8)

        P_w['up'] = (A_inv + A_inv * B * S * C * A_inv) * V_t - A_inv * B * S * V;
        P_w['dn'] = -S * C * A_inv * V + S * V_t;
        
        return P_w
    

    def N(self, g_w):
        return total_density(g_w, 8)
        
    def _dyson_dispatch(self, g_w, mu, sigma_w = None):
        if sigma_w is not None:
            return dyson_mu_sigma(g_w, mu, sigma_w, 8)
        return dyson_mu(g_w, mu, 8)
    
    
    def dyson_equation(self, g_w, mu, sigma_w = None, N_fix = False):
        if not N_fix:
            if mu == 0 and sigma_w is None:
                return g_w, mu
            return self._dyson_dispatch(g_w, mu, sigma_w), mu
        
        else:
            
            previous_direction = None

            occupation = self.N(self._dyson_dispatch(g_w, mu, sigma_w))
            step = abs(occupation - N_fix)
            # step = 1.0
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
    
