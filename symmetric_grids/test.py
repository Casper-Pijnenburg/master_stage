from triqs.gf import Gf, make_gf_dlr, make_gf_dlr_imfreq
from triqs.gf.meshes import MeshDLRImTime

tau_mesh = MeshDLRImTime(beta = 20, statistic = 'Boson',  w_max = 2.0, eps = 1e-10, symmetrize = True)

G_tau = Gf(mesh = tau_mesh, target_shape = [])

G_dlr = make_gf_dlr(G_tau)
G_iw = make_gf_dlr_imfreq(G_dlr)