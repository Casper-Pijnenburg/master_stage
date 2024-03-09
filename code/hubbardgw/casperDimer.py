import numpy as np

from triqs.gf import *

from itertools import product

class GWA:
    def __init__(self, 
                 beta = 20.0,  U = 1.5, t = 1.0, nw = 1 * 1024, spinless = False,
                static_flag = False, hartree_flag = False, fock_flag = False):
        
        self.t = t
        self.U = U
        self.eps = 0
        self.beta = beta
        self.nw = int(nw)
        self.spinless = spinless
        
        self.gf_struct = [('up', 2), ('dn', 2)]

        
        self.fmesh = MeshImFreq(beta, 'Fermion', self.nw)
        self.fermionicGF = BlockGf(mesh = self.fmesh, gf_struct = self.gf_struct, target_rank = 2)
        
        self.bmesh = MeshImFreq(beta, 'Boson', self.nw)    
        self.bosonicGF = BlockGf(mesh = self.bmesh, gf_struct = self.gf_struct, target_rank = 2)
        
        self.omegaf = np.imag(self.ifmesh('Fermion'))
        self.omegab = np.imag(self.ifmesh('Boson'))

            
        self.g0_w = self.ExactG(interactions = False)
        self.v = self.coulombPotential()
        
        
        P = self.polarization(self.g0_w)
        self.P_w = make_gf_from_fourier(P)
        
        self.W_w = self.screenedPotential(P, self.v)
        
        sigma = self.selfEnergy(self.g0_w, self.W_w)
        
        self.sigma_w = make_gf_from_fourier(sigma)

        if static_flag:
            self.Σs = self.static(self.g0_w, self.v)
            self.sigma_w += self.Σs
            
        if hartree_flag:
            if self.spinless:
                self.hartree = self.hartree(self.v, self.g0_w)
                self.sigma_w += self.hartree
            else:
                self.hartree = self.U
                self.sigma_w += self.hartree
        
        if fock_flag:
            self.fock = self.fock(self.v, self.g0_w)
            self.sigma_w += self.fock
            


        self.gw_w = self.greenFunction(self.g0_w, self.sigma_w)
    
    def ExactGW(self):
        h = np.sqrt(4 * self.t ** 2 + 4 * self.U * self.t)

        A = np.sqrt((2 * self.t + h + self.U / 2) ** 2 + 4 * self.U ** 2 * self.t / h)
        B = np.sqrt((2 * self.t + h - self.U / 2) ** 2 + 4 * self.U ** 2 * self.t / h)

        w1_p = (2 * self.eps - h + self.U / 2 + A) / 2
        w1_m = (2 * self.eps - h + self.U / 2 - A) / 2

        w2_p = (2 * self.eps + h + self.U / 2 + B) / 2
        w2_m = (2 * self.eps + h + self.U / 2 - B) / 2

        R1 = ( h + 2 * self.t + self.U / 2) / (4 * A)
        R2 = (-h - 2 * self.t + self.U / 2) / (4 * B)
        
        def greenDiag():
            return (0.25 + R1) * inverse(iOmega_n - w1_p) + (0.25 - R1)*inverse(iOmega_n - w1_m) \
                 + (0.25 + R2)* inverse(iOmega_n - w2_p) + (0.25 - R2) * inverse(iOmega_n - w2_m)

        def greenOffDiag():
            return - (0.25 + R1) * inverse(iOmega_n - w1_p) - (0.25 - R1) * inverse(iOmega_n - w1_m) \
                   + (0.25 + R2) * inverse(iOmega_n - w2_p) + (0.25 - R2) * inverse(iOmega_n - w2_m)

        G = self.fermionicGF.copy()
        
        indices = (0, 1)
        for i, j in product(indices, indices):
            G['up'][i, j] << greenDiag() * int(i == j) + greenOffDiag() * int(i != j)
            G['dn'][i, j] << greenDiag() * int(i == j) + greenOffDiag() * int(i != j)

        return G

    def ExactG(self, interactions = True):
        U = self.U * int(interactions)
    
        c = np.sqrt(16 * self.t ** 2 + U * 2)
        a = np.sqrt(2 * (16 * self.t ** 2 / ((c - U) ** 2) + 1))
        b = np.sqrt(2 * (16 * self.t ** 2 / ((c + U) ** 2) + 1))
        
        w1 = self.eps - self.t + (c + U) / 2
        w2 = self.eps + self.t + (c + U) / 2
        w3 = self.eps + self.t - (c - U) / 2
        w4 = self.eps - self.t - (c - U) / 2
        
        R1 = (1 + 4 * self.t / (c - U)) ** 2
        R2 = (1 - 4 * self.t / (c - U)) ** 2
        

        def greenDiag():
            return (R1 * inverse(iOmega_n - w1) + R2 * inverse(iOmega_n - w2)) / (2 * a ** 2) \
                 + (R1 * inverse(iOmega_n - w3) + R2 * inverse(iOmega_n - w4)) / (2 * a ** 2)

        def greenOffDiag():
            return -1 * (R1 * inverse(iOmega_n - w1) - R2 * inverse(iOmega_n - w2)) / (2 * a ** 2) \
                 + (R1 * inverse(iOmega_n - w3) -
                    R2 * inverse(iOmega_n - w4)) / (2 * a ** 2)
        
        G = self.fermionicGF.copy()
        
        indices = (0, 1)
        for i, j in product(indices, indices):
            G['up'][i, j] << greenDiag() * int(i == j) + greenOffDiag() * int(i != j)
            G['dn'][i, j] << greenDiag() * int(i == j) + greenOffDiag() * int(i != j)

        return G

    def ExactP(self):
        def greenDiag():
            return (inverse(iOmega_n - 2 * self.t) - inverse(iOmega_n + 2 * self.t)) / 4

        def greenOffDiag():
            return -1 * (inverse(iOmega_n - 2 * self.t) - inverse(iOmega_n + 2 * self.t)) / 4
        
        G = self.bosonicGF.copy()
        
        indices = (0, 1)
        for i, j in product(indices, indices):
            G['up'][i, j] << greenDiag() * int(i == j) + greenOffDiag() * int(i != j)
            G['dn'][i, j] << greenDiag() * int(i == j) + greenOffDiag() * int(i != j)

        return G

    def ExactW(self):
        Wdiag = Gf(mesh = self.bmesh, target_shape = [])
        Woffdiag = Gf(mesh = self.bmesh, target_shape = [])
        
        h2 = 4 * self.t ** 2 + 4 * self.U * self.t
        
        for w in self.bmesh:
            Wdiag[w] = self.U + 2 * self.U ** 2 * self.t / (complex(w) ** 2 - h2)
            Woffdiag[w] = 0   - 2 * self.U ** 2 * self.t / (complex(w) ** 2 - h2)

        G = self.bosonicGF.copy()
        
        indices = (0, 1)
        for i, j in product(indices, indices):
            G['up'][i, j] << Wdiag * int(i == j) + Woffdiag * int(i != j)
            G['dn'][i, j] << Wdiag * int(i == j) + Woffdiag * int(i != j)

        return G
    
    def ExactΣ(self):
        h = np.sqrt(4 * self.t ** 2 + 4 * self.t * self.U)
        w1 = self.eps + self.t + h
        w2 = self.eps - self.t - h
        
        def greenDiag():
            return self.U / 2 + self.U ** 2 * self.t * (inverse(iOmega_n - w1) + inverse(iOmega_n - w2)) / (2 * h)

        def greenOffDiag():
            return      0     + self.U ** 2 * self.t * (inverse(iOmega_n - w1) - inverse(iOmega_n - w2)) / (2 * h)
        
        G = self.fermionicGF.copy()
        
        indices = (0, 1)
        for i, j in product(indices, indices):
            G['up'][i, j] << greenDiag() * int(i == j) + greenOffDiag() * int(i != j)
            G['dn'][i, j] << greenDiag() * int(i == j) + greenOffDiag() * int(i != j)

        return G

    def coulombPotential(self):
        v = BlockGf(mesh = self.bmesh, gf_struct = self.gf_struct, target_rank = 4)
        for name, g in v:
            v[name].data[:, 0, 0, 0, 0] = self.U
            v[name].data[:, 1, 1, 1, 1] = self.U


        if not self.spinless:
            v[name].data[:, 1, 1, 0, 0] = self.U
            v[name].data[:, 0, 0, 1, 1] = self.U
            
        return v

    def density(self, G):
        rho = BlockGf(mesh = self.bmesh, gf_struct = self.gf_struct, target_rank = 2)
        for name, g in G:
            rho[name].data[:] = G[name].density()
        return rho

    
    def onMesh(*args):
        arglist = []
        for arg in args:
            arglist.append(arg)
            
        arglist.pop(0)
        if len(arglist) <= 1:
            return arglist[0]
        
        
        for i, arg in enumerate(arglist[1:]):
            new_arg = arglist[0].copy()
            for name, g in arg:
                new_arg[name].data[:] = arg[name].data[:]
            arglist[i+1] = new_arg
        return arglist
    

    def ifmesh(self, statistic):
        if statistic == 'Fermion':
            wmax = (2 * (self.nw - 1) + 1) * np.pi / self.beta
            return np.linspace(-wmax, wmax, 2 * self.nw) * 1j
        elif statistic == 'Boson':
            wmax = (2 * (self.nw - 1)) * np.pi / self.beta
            mesh_p = np.linspace(0, wmax, self.nw) * 1j
            mesh_m = (-np.linspace(0, wmax, self.nw) * 1j)[::-1]
            return np.concatenate((mesh_m[:-1], mesh_p))

    def polarization(self, G):
        
        G = make_gf_from_fourier(G)   
        P = make_gf_from_fourier(BlockGf(mesh = self.bmesh, gf_struct = self.gf_struct, target_rank = 4))
        
        for name, g in P: 
            indices = [0, 1]
            for a, b, c, d in product(indices, indices, indices, indices):
                P[name].data[:, a, b, c, d] = -G[name].data[:, d, a] * G[name].transpose().data[::-1, b, c]

        return P
    
    def screenedPotential(self, P, v):
        P = make_gf_from_fourier(P)
        
        W = BlockGf(mesh = self.bmesh, gf_struct = self.gf_struct, target_rank = 4)

        reshaped = BlockGf(mesh = self.bmesh, gf_struct = [('up', 4), ('dn', 4)], target_rank = 2)

        v_reshaped = reshaped.copy()
        P_reshaped = reshaped.copy()
        idm = reshaped.copy()

        size = np.shape(W['up'].data)[0]

        for name, g, in W:
            v_reshaped[name].data[:] = v[name].data.reshape(size, 4, 4)
            P_reshaped[name].data[:] = P[name].data.reshape(size, 4, 4)
            idm[name].data[:] = np.eye(4)

        W_reshaped = (idm - 2 * v_reshaped * P_reshaped).inverse() * v_reshaped

        for name, g in W:
            W[name].data[:] = W_reshaped[name].data.reshape(size, 2, 2, 2, 2)

        return W


    def screenedPotential2(self, P, v):
        P = make_gf_from_fourier(P)

        W = v.copy()

        for i in range(500):

            Wnew = BlockGf(mesh = self.bmesh, gf_struct = self.gf_struct, target_rank = 4)
    
            for name, g in W:
                indices = [0, 1]
                for a, b, c, d in product(indices, indices, indices, indices):
                    acc = 0
                    for e, f, g, h in product(indices, indices, indices, indices):
                        acc += v[name].data[:, a, b, e, f] * P[name].data[:, f, e, g, h] * W[name].data[:, h, g, c, d]
                    Wnew[name].data[:, a, b, c, d] = v[name].data[:, a, b, c, d] + acc
            
            W = Wnew.copy()

        return W

    
    def selfEnergy(self, G, W):
        
        G = make_gf_from_fourier(G)
        
        W_dynamic = make_gf_from_fourier(W - self.v)

        Σ = make_gf_from_fourier(BlockGf(mesh = self.fmesh, gf_struct = self.gf_struct, target_rank = 2))
        cpy = Σ.copy()

        for name, g in Σ:
            indices = [0, 1]
            for a, b in product(indices, indices):
                acc = cpy.copy()          
                indices = [0, 1]
                for c, d in product(indices, indices):
                    acc[name].data[:, a, b] += -W_dynamic[name].data[:, a, c, b, d] * G[name].data[:, c, d]

                Σ[name].data[:, a, b] = acc[name].data[:, a, b]

        return Σ

    def static(self, G, V):
        rho = self.density(G)

        static = BlockGf(mesh = self.fmesh, gf_struct = self.gf_struct, target_rank = 2)
        
        for name, g in static:
            indices = [0, 1]
            for a, b in product(indices, indices):
                static[name].data[:-1, a, b] = self.v[name].data[:, a, b, a, b] * rho[name].data[:, b, a]
                static[name].data[-1, a, b] = self.v[name].data[-1, a, b, a, b] * rho[name].data[-1, b, a]
           
        return static

    def hartree(self, v, G):
        rho = self.density(G)
    
        hartree = BlockGf(mesh = self.fmesh, gf_struct = self.gf_struct, target_rank = 2)
        cpy = hartree.copy()
            
        for name, g in hartree:
            indices = [0, 1]
            for a, b in product(indices, indices):
                acc = cpy.copy()          
                indices = [0, 1]
                for c, d in product(indices, indices):
                    acc[name].data[:-1, a, b] += v[name].data[:, a, b, c, d] * rho[name].data[:, c, d]
                    acc[name].data[-1, a, b] += v[name].data[-1, a, b, c, d] * rho[name].data[-1, c, d]

                hartree[name].data[:, a, b] = acc[name].data[:, a, b]
                
        return hartree

    def fock(self, v, G):
        rho = self.density(G)
    
        fock = BlockGf(mesh = self.fmesh, gf_struct = self.gf_struct, target_rank = 2)
        cpy = fock.copy()
            
        for name, g in fock:
            indices = [0, 1]
            for a, b in product(indices, indices):
                acc = cpy.copy()          
                indices = [0, 1]
                for c, d in product(indices, indices):
                    acc[name].data[:-1, a, b] += v[name].data[:, a, c, d, b] * rho[name].data[:, d, c]
                    acc[name].data[-1, a, b] += v[name].data[-1, a, c, d, b] * rho[name].data[-1, d, c]

                fock[name].data[:, a, b] = -acc[name].data[:, a, b]
                
        return fock
        
    
    def greenFunction(self, G0, Σ):
        return (G0.inverse() - Σ).inverse()


    def polarizationCheck(self, P):
        flag = 1
        for name, g in P:
            flag *= int(np.allclose(P[name].data[:, 0, 0, 0, 0], self.ExactP()[name].data[:, 0,0]))
            flag *= int(np.allclose(P[name].data[:, 1, 1, 0, 0], self.ExactP()[name].data[:, 0,1]))
            
        print("Polarization : Pass") if flag == 1 else print("Polarization : Fail")

    def screenedPotentialCheck(self, W):
        flag = 1
        for name, g in W:
            flag *= int(np.allclose(W[name].data[:, 0, 0, 0, 0], self.ExactW()[name].data[:, 0,0]))
            flag *= int(np.allclose(W[name].data[:, 1, 1, 0, 0], self.ExactW()[name].data[:, 0,1]))
        print("Screened Potential : Pass") if flag == 1 else print("Screened Potential : Fail")
        
    def selfEnergyCheck(self, Σ):
        flag = 1
        for name, g in Σ:
            flag *= int(np.allclose(Σ[name].data[:, 0, 0], self.ExactΣ()[name].data[:, 0,0]))
            flag *= int(np.allclose(Σ[name].data[:, 0, 1], self.ExactΣ()[name].data[:, 0,1]))
        print("Self-Energy : Pass") if flag == 1 else print("Self-Energy : Fail")

    def GWCheck(self, G):
        flag = 1
        for name, g in G:
            flag *= int(np.allclose(G[name].data[:, 0, 0], self.ExactGW()[name].data[:, 0,0]))
            flag *= int(np.allclose(G[name].data[:, 0, 1], self.ExactGW()[name].data[:, 0,1]))
        print("GW Green function : Pass") if flag == 1 else print("GW Green function : Fail")
    
    def sanityCheck(self):
        self.polarizationCheck(self.P_w)
        self.screenedPotentialCheck(self.W_w)
        self.selfEnergyCheck(self.sigma_w)
        self.GWCheck(self.gw_w)
        print("Tests complete.")