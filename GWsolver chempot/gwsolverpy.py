import numpy as np

from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq
from triqs.gf import make_gf_dlr, make_gf_dlr_imtime, make_gf_dlr_imfreq

from itertools import product
        
class GWSolverPy():
    def __init__(self,
                 g0_w, V, self_interactions = False,
                 hartree_flag = True, fock_flag = True,
                 mu = 0, N_fix = False, N_tol = 1e-6, max_iter = 1, full_mesh = False, verbose = False):
        

        self.self_interactions, self.hartree_flag, self.fock_flag = self_interactions, hartree_flag, fock_flag


        self.blocks = [name for name, g in g0_w]
        self.target_shape = g0_w[self.blocks[0]][Idx(0)].shape

        self.N_fix = N_fix
        if self.N_fix is None:
            self.N_fix = self.target_shape[0]


        # self.g0_w = self.fix_particle_number(g0_w, self.N_fix)
        self.g0_w = g0_w

        self.g0_dlr = make_gf_dlr(self.g0_w)
        self.g0_t = make_gf_dlr_imtime(self.g0_dlr)

        self.fmesh = self.g0_w.mesh
        self.beta = self.fmesh.beta
        self.w_max = self.fmesh.w_max
        self.eps = self.fmesh.eps
        self.bmesh = MeshDLRImFreq(self.beta, 'Boson', w_max = self.w_max, eps = 1e-12, symmetrize = True)
        self.indices = range(self.target_shape[0])
    
        
        V_shape = Gf(mesh = self.bmesh, target_shape = self.target_shape)
        V_shape.data[:] = V
        self.V = BlockGf(block_list = [V_shape] * 2, name_list = self.blocks, make_copies = False)


        self.sigma_w = self.g0_w.copy()
        self.g_w = self.g0_w.copy()
        self.g_w_old = self.g_w.copy()


        for iter in range(max_iter):
            self.sigma_w.zero()

            self.P_w = self.polarization(self.g_w)

            self.W_w = self.screenedPotential(self.V, self.P_w)

            self.rho_w = self.density(self.g_w)
            
            self.sigma_w = self.selfEnergy(self.g_w, self.W_w)

            self.sigma_w += self.hartree(self.V, self.rho_w)


            self.sigma_w += self.fock(self.V, self.rho_w)

            self.g_w = self.oneShotG(self.g0_w, self.sigma_w)


            # diff = np.max(np.abs((self.g_w - self.g_w_old)['up'].data))
            # self.g_w_old = self.g_w.copy()
            # if diff < 1e-7:
            #     print(f'Convergence reached at iteration {iter} with diff {diff}')
            #     break

        # self.g_w = self.fix_particle_number(self.g_w, self.N_fix)

    def polarization(self, g_w):
        g_dlr = make_gf_dlr(g_w)
        g_t = make_gf_dlr_imtime(g_dlr)
        P_dlr = make_gf_dlr(Gf(mesh = self.bmesh, target_shape = self.target_shape))
        P_t = make_gf_dlr_imtime(P_dlr)
        P_t = BlockGf(block_list = [P_t] * 2, name_list = self.blocks)
        
        for block in self.blocks:
            for a, b in product(self.indices, self.indices):
                P_t[block].data[:, a, b] = -g_t[block].data[:, a, b] * g_t[block].data[::-1, b, a]

        P_dlr = make_gf_dlr(P_t)
        return make_gf_dlr_imfreq(P_dlr)

    def screenedPotential(self, V, P_w):
        W = P_w.copy()

        full_shape = tuple(2 * shape for shape in self.target_shape)
        P_full = Gf(mesh = self.bmesh, target_shape = full_shape)
        idm_full = Gf(mesh = self.bmesh, target_shape = full_shape)
        V_full = Gf(mesh = self.bmesh, target_shape = full_shape)

        up = slice(0, self.target_shape[0])
        dn = slice(self.target_shape[0], 2 * self.target_shape[0])


        P_full.data[:, up, up] = P_w['up'].data[:]
        P_full.data[:, dn, dn] = P_w['dn'].data[:]

        idm_full.data[:] = np.eye(2 * self.target_shape[0])

        V_full.data[:, up, dn] = V['up'].data[:]
        V_full.data[:, dn, up] = V['dn'].data[:]

        V_full.data[:, up, up] = V['up'].data[:]
        V_full.data[:, dn, dn] = V['dn'].data[:]
        
        if not self.self_interactions:
            for i, w in enumerate(V_full.mesh.values()):
                for j in range(2 * self.target_shape[0]):
                    V_full.data[i, j, j] = 0

        W_full = (idm_full - V_full * P_full).inverse() * V_full

        W['up'].data[:] = W_full.data[:, up, up]
        W['dn'].data[:] = W_full.data[:, dn, dn]

        return W
        
    
    def selfEnergy(self, g_w, W_w):

        g_dlr = make_gf_dlr(g_w)
        g_t = make_gf_dlr_imtime(g_dlr)
        V = self.V.copy()
        if not self.self_interactions:
            for i, w in enumerate(V.mesh.values()):
                for block in self.blocks:
                    for j in range(self.target_shape[0]):
                        V[block].data[i, j, j] = 0

        W_dynamic_t_dlr = make_gf_dlr(W_w - V)
        W_dynamic_t = make_gf_dlr_imtime(W_dynamic_t_dlr)

        

        sigma_t = g_t.copy()
        for block in self.blocks:
            sigma_t[block].data[:] = -W_dynamic_t[block].data[:] * g_t[block].data[:]

        sigma_dlr = make_gf_dlr(sigma_t)
        return make_gf_dlr_imfreq(sigma_dlr)

    def density(self, g_w):
        rho = g_w.copy()
        g_dlr = make_gf_dlr(g_w)
        for block in self.blocks:
            rho[block].data[:] = g_dlr[block].density()
        return rho 

    def logic(self, block, spin):
        if block != spin:
            return 1
        if block == spin and self.self_interactions:
            return 1
        return 0

    def hartree(self, V, rho):
        v = V['up'].data[0, :, :]
        v_t = v.copy()
        np.fill_diagonal(v_t, 0)

        hartree = rho.copy()
        hartree.zero()

        H_up = np.zeros((self.target_shape[0], self.target_shape[0])) * 1j
        H_dn = np.zeros((self.target_shape[0], self.target_shape[0])) * 1j

        for i in self.indices:
            up_temp = 0 * 1j
            dn_temp = 0 * 1j
            for j in self.indices:
                up_temp += v_t[i, j] * rho['up'].data[0, j, j].real + v[i, j] * rho['dn'].data[0, j, j].real
                dn_temp += v[i, j] * rho['up'].data[0, j, j].real + v_t[i, j] * rho['dn'].data[0, j, j].real
            H_up[i, i] = up_temp 
            H_dn[i, i] = dn_temp

        hartree['up'].data[:] = H_up
        hartree['dn'].data[:] = H_dn
        return hartree
    
    def fock(self, V, rho):
        v = V['up'].data[0, :, :]
        v_t = v.copy()
        np.fill_diagonal(v_t, 0)

        V_t = V.copy()
        V_t.zero()
        V_t['up'].data[:] = v_t
        V_t['dn'].data[:] = v_t


        fock = rho.copy()
        for block in self.blocks:
            fock[block].data[:] = - v_t * rho[block].data[:]
        return fock
        
    def oneShotG(self, g0_w, sigma_w):
        return (g0_w.inverse() - sigma_w).inverse()
   
    def N(self, g_w):
        g_dlr = make_gf_dlr(g_w)
        return g_dlr.total_density().real
        
    def dyson(self, g_w, mu):
        G = g_w.copy()
        mu_gf = g_w[self.blocks[0]].copy()
        mu_gf.data[:] = np.eye(self.target_shape[0]) * mu
        for block, g in g_w:
            G[block] = (g_w[block].inverse() - mu_gf).inverse()
        return G


    def fix_particle_number(self, g, target_occupation, mu = None, eps = 1e-3):
        if mu is None:
            mu = 0
        step = 1.0

        previous_direction = None

        occupation = self.N(self.dyson(g, mu))
        iterator = 0
        while abs(occupation - target_occupation) > eps:
            if occupation - target_occupation > 0:
                if previous_direction == 'decrement':
                    step /= 2
                previous_direction = 'increment'
                mu += step
            if occupation - target_occupation < 0:
                if previous_direction == 'increment':
                    step /= 2
                previous_direction = 'decrement'
                mu -= step
            occupation = self.N(self.dyson(g, mu))
            iterator += 1
        self.mu = mu
        return self.dyson(g, mu)
    
