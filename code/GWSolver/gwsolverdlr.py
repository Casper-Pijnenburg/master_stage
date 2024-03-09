import numpy as np

from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq
from triqs.gf import make_gf_dlr, make_gf_dlr_imtime, make_gf_dlr_imfreq

from itertools import product

class GWSolverDLR():
    def __init__(self,
                 g0_w, V, self_interactions = False,
                 hartree_flag = False, fock_flag = False,
                 N_fix = None):
        
        self.self_interactions, self.hartree_flag, self.fock_flag = self_interactions, hartree_flag, fock_flag


        self.blocks = [name for name, g in g0_w]
        self.target_shape = g0_w[self.blocks[0]][Idx(0)].shape

        self.N_fix = N_fix

        self.g0_w = self.fix_particle_number(g0_w, self.N_fix)

        self.g0_dlr = make_gf_dlr(self.g0_w)
        self.g0_t = make_gf_dlr_imtime(self.g0_dlr)

        self.fmesh = self.g0_w.mesh
        self.beta = self.fmesh.beta
        self.w_max = self.fmesh.w_max
        self.eps = self.fmesh.eps
        self.bmesh = MeshDLRImFreq(self.beta, 'Boson', w_max = self.w_max, eps = 1e-15)
        self.indices = range(self.target_shape[0])
    
        
        V_shape = Gf(mesh = self.bmesh, target_shape = self.target_shape)
        V_shape.data[:] = V
        self.V = BlockGf(block_list = [V_shape] * 2, name_list = self.blocks, make_copies = False)

        self.P_w = self.polarization(self.g0_dlr)

        self.W_w = self.screenedPotential(self.V, self.P_w)

        self.rho_w = self.density(self.g0_w)
        
        self.sigma_w = self.selfEnergy(self.g0_t, self.W_w)

        if self.hartree_flag:
            self.hartree = self.hartree(self.V, self.rho_w)
            self.sigma_w += self.hartree
        else:
            self.hartree = 0

        if self.fock_flag and self.self_interactions:
            self.fock = self.fock(self.V, self.rho_w)
            self.sigma_w += self.fock
        else:
            self.fock = 0

        self.g_w = self.oneShotG(self.g0_w, self.sigma_w)
        self.g_w = self.fix_particle_number(self.g_w, self.N_fix)

    def polarization(self, g0_dlr):
        P_dlr = make_gf_dlr(Gf(mesh = self.bmesh, target_shape = self.target_shape))
        P_t = make_gf_dlr_imtime(P_dlr)
        P_t = BlockGf(block_list = [P_t] * 2, name_list = self.blocks)
        
        for block in self.blocks:
            for a, b in product(self.indices, self.indices):
                for i, tau in enumerate(P_t.mesh.values()):
                        P_t[block].data[i, a, b] = -g0_dlr[block](tau)[a, b] * g0_dlr[block](self.beta - tau)[b, a]

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

        if self.self_interactions:
            V_full.data[:, up, up] = V['up'].data[:]
            V_full.data[:, dn, dn] = V['dn'].data[:]
        

        for i, w in enumerate(V_full.mesh.values()):
            for j in range(2 * self.target_shape[0]):
                V_full.data[i, j, j] = 0

        W_full = (idm_full - V_full * P_full).inverse() * V_full

        W['up'].data[:] = W_full.data[:, up, up]
        W['dn'].data[:] = W_full.data[:, dn, dn]

        return W
        
    
    def selfEnergy(self, g0_t, W_w):
        V = self.V.copy()
        for i, w in enumerate(V.mesh.values()):
            for block in self.blocks:
                for j in range(self.target_shape[0]):
                    V[block].data[i, j, j] = 0

        W_dynamic_t_dlr = make_gf_dlr(W_w - V * int(self.self_interactions))
        W_dynamic_t = make_gf_dlr_imtime(W_dynamic_t_dlr)

        sigma_t = g0_t.copy()
        for block in self.blocks:
            sigma_t[block].data[:] = -W_dynamic_t[block].data[:] * g0_t[block].data[:]

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
        hartree = rho.copy()
        hartree.zero()

        for block in self.blocks:
            for i in self.indices:
                for j in self.indices:
                    hartree[block].data[:, i, i] += self.logic(block, 'up') * V['up'].data[:, i, j] * rho['up'].data[:, j, j] \
                                                  + self.logic(block, 'dn') * V['dn'].data[:, i, j] * rho['dn'].data[:, j, j]

        return hartree
    
    def fock(self, V, rho):
        fock = rho.copy()
        for block in self.blocks:
            fock[block].data[:] = -V[block].data[:] * rho[block].data[:]
        return fock
        
    def oneShotG(self, g0_w, sigma_w):
        return (g0_w.inverse() - sigma_w).inverse()
   
    def N(self, g_w):
        g_dlr = make_gf_dlr(g_w)
        rho = g_dlr.density()
        N = 0
        for block in self.blocks:
            N += np.sum(np.diag(rho[block].data)).real
        return N
        
    def dyson(self, g_w, mu):
        G = g_w.copy()
        mu_gf = G.copy()
        for block in self.blocks:
            mu_gf[block].data[:] = mu * np.eye(self.target_shape[0])
        for block in self.blocks:
            G[block] = (g_w[block].inverse() - mu_gf[block]).inverse()
        return G

    def fix_particle_number(self, g, N, eps = 1e-3):
        g_w = g.copy()
        step = 1.0

        particle_number = self.N(g_w)

        while abs(particle_number - N) > eps:
            # print(particle_number)
            if particle_number - N > 0:
                g_w = self.dyson(g_w, step)
            elif particle_number - N < 0:
                g_w = self.dyson(g_w, -step)
            step = 0.99 * step
            particle_number = self.N(g_w)

        return g_w