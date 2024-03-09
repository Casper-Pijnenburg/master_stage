import numpy as np

from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq
from triqs.gf import make_gf_dlr, make_gf_dlr_imtime, make_gf_dlr_imfreq

from itertools import product

class GWSolver():
    def __init__(self,
                 g0_w, V, 
                 hartree_flag = False, fock_flag = False,
                 mu = 0, N_fix = None):
        
        self.hartree_flag, self.fock_flag = hartree_flag, fock_flag
        self.mu = mu

        self.g0_w = g0_w
        self.g0_t = make_gf_from_fourier(self.g0_w)

        if isinstance(self.g0_w, BlockGf):
            self.block_struct = True
            self.blocks = [name for name, g in g0_w]
            self.target_shape = self.g0_w[self.blocks[0]][Idx(0)].shape

            self.N_fix = N_fix
        else: 
            self.block_struct = False
            self.target_shape = self.g0_w[Idx(0)].shape

            if N_fix is not None:
                self.N_fix = N_fix // 2
            else:
                self.N_fix = N_fix

        self.fmesh = self.g0_w.mesh
        self.beta = self.fmesh.beta
        self.nw = len(list(self.fmesh))
        self.bmesh = MeshImFreq(self.beta, 'Boson', self.nw // 2)
        self.indices = range(self.target_shape[0])
        
        self.tensor_shape = [self.target_shape[0]] * 4
        self.reshape = [index ** 2 for index in self.target_shape]
        

        if self.block_struct:
            V_shape = Gf(mesh = self.bmesh, target_shape = self.tensor_shape)
            self.V = BlockGf(block_list = [V_shape] * 2, name_list = self.blocks)
            for block in self.blocks:
                self.V[block].data[:] = V
        else: 
            self.V = Gf(mesh = self.bmesh, target_shape = self.tensor_shape)
            self.V.data[:] = V



        self.P_w = self.polarization(self.g0_t)
        
        if not self.block_struct:
            self.W_w = self.screenedPotential(self.V, 2 * self.P_w)
        else:
            self.W_w = self.screenedPotential(self.V, self.P_w)

        self.rho_w = self.density(self.g0_w)
        self.sigma_w = self.selfEnergy(self.g0_t, self.W_w)

        
        if self.hartree_flag:
            self.hartree = self.hartree(self.V, self.rho_w)
            self.sigma_w += self.hartree
        else:
            self.hartree = 0

        if self.fock_flag:
            self.fock = self.fock(self.V, self.rho_w)
            self.sigma_w += self.fock
        else:
            self.fock = 0

        self.mu, self.g_w = self.greenFunction(self.g0_w, self.sigma_w, mu = self.mu, N_fix = self.N_fix)
        

    def polarization(self, g0_t):
        P_shape = Gf(mesh = self.bmesh, target_shape = self.tensor_shape)
        if self.block_struct:
            P_t = make_gf_from_fourier(BlockGf(block_list = [P_shape] * 2, name_list = self.blocks))
            for block in self.blocks:
                for a, b, c, d in product(self.indices, self.indices, self.indices, self.indices):
                    P_t[block].data[:, a, b, c, d] = -g0_t[block].data[:, a, c] * g0_t[block].data[::-1, d, b]
            return make_gf_from_fourier(P_t)
        else:
            P_t = make_gf_from_fourier(P_shape)
            for a, b, c, d in product(self.indices, self.indices, self.indices, self.indices):
                P_t.data[:, a, b, c, d] = -g0_t.data[:, a, c] * g0_t.data[::-1, d, b]
            return make_gf_from_fourier(P_t)
        
            
    
    def screenedPotential(self, V, P_w):
        if self.block_struct:
            W = P_w.copy()

            reshaped_block = Gf(mesh = self.bmesh, target_shape = self.reshape)
            reshaped = BlockGf(block_list = [reshaped_block] * 2, name_list = self.blocks)

            v_reshaped = reshaped.copy()
            P_reshaped = reshaped.copy()
            idm = reshaped.copy()

            size = len(list(self.bmesh))

            for block in self.blocks:
                v_reshaped[block].data[:] = V[block].data.reshape(size, *self.reshape)
                P_reshaped[block].data[:] = P_w[block].data.reshape(size, *self.reshape)
                idm[block].data[:] = np.eye(self.reshape[0])

            VP = reshaped.copy()
            for block in self.blocks:
                for spin in self.blocks:
                    VP[block] += v_reshaped[spin] * P_reshaped[spin]


            W_reshaped = (idm - VP).inverse() * v_reshaped

            for block in self.blocks:
                W[block].data[:] = W_reshaped[block].data.reshape(size, *self.tensor_shape)

            return W
        else:
            W = P_w.copy()

            reshaped = Gf(mesh = self.bmesh, target_shape = self.reshape)

            v_reshaped = reshaped.copy()
            P_reshaped = reshaped.copy()
            idm = reshaped.copy()

            size = len(list(self.bmesh))

            v_reshaped.data[:] = V.data.reshape(size, *self.reshape)
            P_reshaped.data[:] = P_w.data.reshape(size, *self.reshape)
            idm.data[:] = np.eye(self.reshape[0])


            W_reshaped = (idm - v_reshaped * P_reshaped).inverse() * v_reshaped

            
            W.data[:] = W_reshaped.data.reshape(size, *self.tensor_shape)

            return W
    
    def selfEnergy(self, g0_t, W_w):
        
        W_dynamic_t = make_gf_from_fourier(W_w - self.V)

        if self.block_struct:
            sigma_t = g0_t.copy()
            for block in self.blocks:
                sigma_t[block].data[:] = 0
            cpy = sigma_t.copy()

            for block in self.blocks:
                for a, b in product(self.indices, self.indices):
                    acc = cpy.copy()          
                    for c, d in product(self.indices, self.indices):
                        acc[block].data[:, a, b] += -W_dynamic_t[block].data[:, b, c, d, a] * g0_t[block].data[:, c, d]

                    sigma_t[block].data[:, a, b] = acc[block].data[:, a, b]

            return make_gf_from_fourier(sigma_t)

        else:
            sigma_t = g0_t.copy()
            sigma_t.data[:] = 0
            cpy = sigma_t.copy()


            for a, b in product(self.indices, self.indices):
                acc = cpy.copy()          
                for c, d in product(self.indices, self.indices):
                    acc.data[:, a, b] += -W_dynamic_t.data[:, b, c, d, a] * g0_t.data[:, c, d]

                sigma_t.data[:, a, b] = acc.data[:, a, b]

            return make_gf_from_fourier(sigma_t)

    def density(self, g_w):
        rho = g_w.copy()
        if self.block_struct:
            for block in self.blocks:
                rho[block].data[:] = g_w[block].density()
            return rho 
        else:
            rho.data[:] = g_w.density()
            return rho 

    def hartree(self, v, rho):

        if self.block_struct:
            hartree_block = Gf(mesh = self.fmesh, target_shape = self.target_shape)
            hartree = BlockGf(block_list = [hartree_block] * 2, name_list = self.blocks)
            full_hartree = hartree.copy()
            cpy = hartree.copy()

            for block in self.blocks:
                for a, b in product(self.indices, self.indices):
                    acc = cpy.copy()          
                    for c, d in product(self.indices, self.indices):
                        acc[block].data[:-1, a, b] += v[block].data[:, d, b, c, a] * rho[block].data[:-1, c, d]
                        acc[block].data[-1, a, b] += v[block].data[-1, d, b, c, a] * rho[block].data[-1, c, d]


                    hartree[block].data[:, a, b] = acc[block].data[:, a, b]

            
            for block in self.blocks:
                for spin in self.blocks:
                    full_hartree[block] += hartree[spin]

            return full_hartree
        else:
            hartree = Gf(mesh = self.fmesh, target_shape = self.target_shape)
            cpy = hartree.copy()
                
            for a, b in product(self.indices, self.indices):
                acc = cpy.copy()          
                for c, d in product(self.indices, self.indices):
                    acc.data[:-1, a, b] += v.data[:, d, b, c, a] * rho.data[:-1, c, d]
                    acc.data[-1, a, b] += v.data[-1, d, b, c, a] * rho.data[-1, c, d]


                hartree.data[:, a, b] = acc.data[:, a, b]
                    
            return hartree
    
    def fock(self, v, rho):
        if self.block_struct:
            fock_block = Gf(mesh = self.fmesh, target_shape = self.target_shape)
            fock = BlockGf(block_list = [fock_block] * 2, name_list = self.blocks)
            cpy = fock.copy()

            for block in self.blocks:
                for a, b in product(self.indices, self.indices):
                    acc = cpy.copy()          
                    for c, d in product(self.indices, self.indices):
                        acc[block].data[:-1, a, b] += v[block].data[:, b, c, d, a] * rho[block].data[:-1, d, c]
                        acc[block].data[-1, a, b] += v[block].data[-1, b, c, d, a] * rho[block].data[-1, d, c]
                    fock[block].data[:, a, b] = -acc[block].data[:, a, b]

            return fock
        
        else:
            fock = Gf(mesh = self.fmesh, target_shape = self.target_shape)
            cpy = fock.copy()
                
            for a, b in product(self.indices, self.indices):
                acc = cpy.copy()          
                for c, d in product(self.indices, self.indices):
                    acc.data[:-1, a, b] += v.data[:, b, c, d, a] * rho.data[:-1, d, c]
                    acc.data[-1, a, b] += v.data[-1, b, c, d, a] * rho.data[-1, d, c]
                fock.data[:, a, b] = -acc.data[:, a, b]
                    
            return fock
    def N(self, g_w):
        rho = g_w.density()
        if self.block_struct:
            N = 0
            for block in self.blocks:
                N += np.sum(np.diag(rho[block].data)).real
            return N
        else:
            return np.sum(np.diag(rho.data)).real
        
    def greenFunction(self, g0_w, sigma_w = 0, mu = 0, N_fix = None):
        if not N_fix:
            return mu, self.dysonEquation(g0_w, sigma_w, mu)
        else:
            def target_function(mu):
                g_w = self.dysonEquation(g0_w, sigma_w, mu)
                return self.N(g_w) - N_fix
            
            max_iter = 1000
            mu = 5
            for i in range(max_iter):
                if abs(target_function(mu)) < 0.01:
                    break
                else:
                    mu += 0.1 * np.sign(target_function(mu))
            if i == max_iter - 1: 
                print("Max iter reached in finding chemical potential.") 
            g_w = self.dysonEquation(g0_w, sigma_w, mu)
            return mu, g_w

    def dysonEquation(self, g0_w, sigma_w = None, mu = 0):
        if sigma_w is not None:
            if self.block_struct:
                g_w = g0_w.copy()
                for block in self.blocks:
                    g_w[block] = (g0_w[block].inverse() - sigma_w[block] - mu).inverse()
                return g_w
            else:
                return (g0_w.inverse() - sigma_w - mu).inverse()
        else:
            if self.block_struct:
                g_w = g0_w.copy()
                for block in self.blocks:
                    g_w[block] = (g0_w[block].inverse() - mu).inverse()
                return g_w
            else:
                return (g0_w.inverse() - mu).inverse()
            
class GWSolverDLR():
    def __init__(self,
                 g0_w, V, 
                 hartree_flag = False, fock_flag = False,
                 mu = 0, N_fix = None):
        
        self.hartree_flag, self.fock_flag = hartree_flag, fock_flag
        self.mu = mu

        self.g0_w = g0_w
        self.g0_dlr = make_gf_dlr(self.g0_w)
        self.g0_t = make_gf_dlr_imtime(self.g0_dlr)

        if isinstance(self.g0_w, BlockGf):
            self.block_struct = True
            self.blocks = [name for name, g in g0_w]
            self.target_shape = self.g0_w[self.blocks[0]][Idx(0)].shape

            self.N_fix = N_fix
        else: 
            self.block_struct = False
            self.target_shape = self.g0_w[Idx(0)].shape

            if N_fix is not None:
                self.N_fix = N_fix // 2
            else:
                self.N_fix = N_fix

        self.fmesh = self.g0_w.mesh
        self.beta = self.fmesh.beta
        self.w_max = self.fmesh.w_max
        self.eps = self.fmesh.eps
        self.bmesh = MeshDLRImFreq(self.beta, 'Boson', w_max = self.w_max, eps = 1e-15)
        self.indices = range(self.target_shape[0])
        
        self.tensor_shape = [self.target_shape[0]] * 4
        self.reshape = [index ** 2 for index in self.target_shape]
        

        if self.block_struct:
            V_shape = Gf(mesh = self.bmesh, target_shape = self.tensor_shape)
            self.V = BlockGf(block_list = [V_shape] * 2, name_list = self.blocks)
            for block in self.blocks:
                self.V[block].data[:] = V
        else: 
            self.V = Gf(mesh = self.bmesh, target_shape = self.tensor_shape)
            self.V.data[:] = V



        self.P_w = self.polarization(self.g0_dlr)
        if not self.block_struct:
            self.W_w = self.screenedPotential(self.V, 2 * self.P_w)
        else:
            self.W_w = self.screenedPotential(self.V, self.P_w)


        self.W_dynamic_w = self.W_w - self.V
        self.rho_w = self.density(self.g0_w)
        self.sigma_w = self.selfEnergy(self.g0_t, self.W_w)

        if self.hartree_flag:
            self.hartree = self.hartree(self.V, self.rho_w)
            self.sigma_w += self.hartree
        else:
            self.hartree = self.sigma_w.copy()
            if self.block_struct:
                for block in self.blocks:
                    self.hartree[block].data[:] = 0
            else:
                self.hartree.data[:] = 0
            
        if self.fock_flag:
            self.fock = self.fock(self.V, self.rho_w)
            self.sigma_w += self.fock
        else:
            self.fock = self.sigma_w.copy()
            if self.block_struct:
                for block in self.blocks:
                    self.fock[block].data[:] = 0
            else:
                self.fock.data[:] = 0
            
            
        self.mu, self.g_w = self.greenFunction(self.g0_w, self.sigma_w, mu = self.mu, N_fix = self.N_fix)
        

    def polarization(self, g0_dlr):
        P_dlr = make_gf_dlr(Gf(mesh = self.bmesh, target_shape = self.tensor_shape))
        P_t = make_gf_dlr_imtime(P_dlr)
        if self.block_struct:
            P_t = BlockGf(block_list = [P_t] * 2, name_list = self.blocks)
            for block in self.blocks:
                for a, b, c, d in product(self.indices, self.indices, self.indices, self.indices):
                    for i, tau in enumerate(P_t.mesh.values()):
                            P_t[block].data[i, a, b, c, d] = -g0_dlr[block](tau)[a, c] * g0_dlr[block](self.beta - tau)[d, b]
                
            P_dlr = make_gf_dlr(P_t)
            return make_gf_dlr_imfreq(P_dlr)
        else:
            for a, b, c, d in product(self.indices, self.indices, self.indices, self.indices):
                for i, tau in enumerate(P_t.mesh.values()):
                    P_t.data[i, a, b, c, d] = -g0_dlr(tau)[a, c] * g0_dlr(self.beta - tau)[d, b]
            P_dlr = make_gf_dlr(P_t)
            return make_gf_dlr_imfreq(P_dlr)
        
            
    
    def screenedPotential(self, V, P_w):
        if self.block_struct:
            W = P_w.copy()

            reshaped_block = Gf(mesh = self.bmesh, target_shape = self.reshape)
            reshaped = BlockGf(block_list = [reshaped_block] * 2, name_list = self.blocks)

            v_reshaped = reshaped.copy()
            P_reshaped = reshaped.copy()
            idm = reshaped.copy()

            size = len(list(self.bmesh))

            for block in self.blocks:
                v_reshaped[block].data[:] = V[block].data.reshape(size, *self.reshape)
                P_reshaped[block].data[:] = P_w[block].data.reshape(size, *self.reshape)
                idm[block].data[:] = np.eye(self.reshape[0])

            VP = reshaped.copy()
            for block in self.blocks:
                for spin in self.blocks:
                    VP[block] += v_reshaped[spin] * P_reshaped[spin]


            W_reshaped = (idm - VP).inverse() * v_reshaped

            for block in self.blocks:
                W[block].data[:] = W_reshaped[block].data.reshape(size, *self.tensor_shape)

            return W
        else:
            W = P_w.copy()

            reshaped = Gf(mesh = self.bmesh, target_shape = self.reshape)
            v_reshaped = reshaped.copy()
            P_reshaped = reshaped.copy()
            idm = reshaped.copy()
            size = len(list(self.bmesh))

            v_reshaped.data[:] = V.data.reshape(size, *self.reshape)
            P_reshaped.data[:] = P_w.data.reshape(size, *self.reshape)
            idm.data[:] = np.eye(self.reshape[0])

            W_reshaped = (idm - v_reshaped * P_reshaped).inverse() * v_reshaped
                        
            W.data[:] = W_reshaped.data.reshape(size, *self.tensor_shape)
            return W
    
    def selfEnergy(self, g0_t, W_w):
        W_dynamic_t_dlr = make_gf_dlr(W_w - self.V)
        W_dynamic_t = make_gf_dlr_imtime(W_dynamic_t_dlr)

        if self.block_struct:
            sigma_t = g0_t.copy()
            for block in self.blocks:
                sigma_t[block].data[:] = 0
            cpy = sigma_t.copy()

            for block in self.blocks:
                for a, b in product(self.indices, self.indices):
                    acc = cpy.copy()          
                    for c, d in product(self.indices, self.indices):
                        acc[block].data[:, a, b] += -W_dynamic_t[block].data[:, b, c, d, a] * g0_t[block].data[:, d, c]

                    sigma_t[block].data[:, a, b] = acc[block].data[:, a, b]
            sigma_dlr = make_gf_dlr(sigma_t)
            return make_gf_dlr_imfreq(sigma_dlr)

        else:
            sigma_t = g0_t.copy()
            sigma_t.data[:] = 0
            cpy = sigma_t.copy()

            for a, b in product(self.indices, self.indices):
                acc = cpy.copy()          
                for c, d in product(self.indices, self.indices):
                    acc.data[:, a, b] += -W_dynamic_t.data[:, b, c, d, a] * g0_t.data[:, d, c]

                sigma_t.data[:, a, b] = acc.data[:, a, b]
            sigma_dlr = make_gf_dlr(sigma_t)
            return make_gf_dlr_imfreq(sigma_dlr)

    def density(self, g_w):
        rho = g_w.copy()
        g_dlr = make_gf_dlr(g_w)
        if self.block_struct:
            for block in self.blocks:
                rho[block].data[:] = g_dlr[block].density()
            return rho 
        else:
            rho.data[:] = g_dlr.density()
            return rho 

    def hartree(self, v, rho):

        if self.block_struct:
            hartree_block = Gf(mesh = self.fmesh, target_shape = self.target_shape)
            hartree = BlockGf(block_list = [hartree_block] * 2, name_list = self.blocks)
            full_hartree = hartree.copy()
            cpy = hartree.copy()

            for block in self.blocks:
                for a, b in product(self.indices, self.indices):
                    acc = cpy.copy()          
                    for c, d in product(self.indices, self.indices):
                        acc[block].data[:, a, b] += v[block].data[:, c, b, d, a] * rho[block].data[:, c, d]

                    hartree[block].data[:, a, b] = acc[block].data[:, a, b]

            for block in self.blocks:
                for spin in self.blocks:
                    full_hartree[block] += hartree[spin]

            return full_hartree
        
        else:
            hartree = Gf(mesh = self.fmesh, target_shape = self.target_shape)
            cpy = hartree.copy()
                
            for a, b in product(self.indices, self.indices):
                acc = cpy.copy()          
                for c, d in product(self.indices, self.indices):
                    acc.data[:, a, b] += v.data[:, c, b, d, a] * rho.data[:, c, d]

                hartree.data[:, a, b] = acc.data[:, a, b]
                    
            return hartree
    
    def fock(self, v, rho):
        if self.block_struct:
            fock_block = Gf(mesh = self.fmesh, target_shape = self.target_shape)
            fock = BlockGf(block_list = [fock_block] * 2, name_list = self.blocks)
            cpy = fock.copy()

            for block in self.blocks:
                for a, b in product(self.indices, self.indices):
                    acc = cpy.copy()          
                    for c, d in product(self.indices, self.indices):
                        acc[block].data[:, a, b] += v[block].data[:, b, c, d, a] * rho[block].data[:, d, c]

                    fock[block].data[:, a, b] = -acc[block].data[:, a, b]

            return fock
        
        else:
            fock = Gf(mesh = self.fmesh, target_shape = self.target_shape)
            cpy = fock.copy()
                
            for a, b in product(self.indices, self.indices):
                acc = cpy.copy()          
                for c, d in product(self.indices, self.indices):
                    acc.data[:, a, b] += v.data[:, b, c, d, a] * rho.data[:, d, c]

                fock.data[:, a, b] = -acc.data[:, a, b]
                    
            return fock
        
    def N(self, g_w):
        g_dlr = make_gf_dlr(g_w)
        rho = g_dlr.density()
        if self.block_struct:
            N = 0
            for block in self.blocks:
                N += np.sum(np.diag(rho[block].data)).real
            return N
        else:
            return np.sum(np.diag(rho.data)).real
        
    def greenFunction(self, g0_w, sigma_w = 0, mu = 0, N_fix = None):
        if not N_fix:
            return mu, self.dysonEquation(g0_w, sigma_w, mu)
        else:
            def target_function(mu):
                g_w = self.dysonEquation(g0_w, sigma_w, mu)
                return self.N(g_w) - N_fix
            
            max_iter = 1000
            mu = 5
            for i in range(max_iter):
                if abs(target_function(mu)) < 0.01:
                    break
                else:
                    mu += 0.1 * np.sign(target_function(mu))
            if i == max_iter - 1: 
                print("Max iter reached in finding chemical potential.") 
            g_w = self.dysonEquation(g0_w, sigma_w, mu)

            return mu, g_w

    def dysonEquation(self, g0_w, sigma_w = None, mu = 0):
        mu_ = mu
        mu = g0_w.copy()
        if self.block_struct:
            for block in self.blocks:
                mu[block].data[:] = np.eye(self.target_shape[0]) * mu_
        else:
            mu.data[:] = np.eye(self.target_shape[0]) * mu_

        if sigma_w is not None:
            if self.block_struct:
                g_w = g0_w.copy()
                for block in self.blocks:
                    g_w[block] = (g0_w[block].inverse() - sigma_w[block] - mu[block]).inverse()
                return g_w
            else:
                return (g0_w.inverse() - sigma_w - mu).inverse()
        else:
            if self.block_struct:
                g_w = g0_w.copy()
                for block in self.blocks:
                    g_w[block] = (g0_w[block].inverse() - mu).inverse()
                return g_w
            else:
                return (g0_w.inverse() - mu).inverse()