import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

np.random.seed(1)

flat_pts, slip_vecs, flat_gfs, flat_surf, fault = np.load('data/hill_ss/flat_ss_gfs.npy')
hill_pts, _, hill_gfs, hill_surf, _ = np.load('data/hill_ss/hill_ss_gfs.npy')

# choose some random obs pts?
# obs_pt_idxs = np.random.randint(0, flat_pts.shape[0], size = (250,))
# or all obs pts?
obs_pt_idxs = np.arange(flat_pts.shape[0])

# use only ux, uy, one row for each observation point.
which_dims = [0,1]
# 3.5m of strike-slip
slip = np.array([3.5,0,0])

def plot_situation():
    w = np.max(hill_pts[:,0])
    xi = np.linspace(-w,w,100)
    yi = np.linspace(-w,w,100)

    z = hill_pts[:, 2]
    zi = griddata((hill_pts[:,0], hill_pts[:,1]), z, (xi[None,:], yi[:,None]), method='cubic')

    min_z, max_z = np.min(z), np.max(z)
    levels = np.linspace(min_z, max_z, 15)[1:]
    CS = plt.contour(xi,yi,zi,levels = levels, linewidths=0.5, colors='k')

    min_faultx = np.min(fault[0][:,0])
    max_faultx = np.max(fault[0][:,0])
    plt.plot([min_faultx, max_faultx], [0,0], 'r-')
    plt.plot(flat_pts[obs_pt_idxs,0], flat_pts[obs_pt_idxs,1], 'o')
    plt.show()
# plot_situation()


rhs_G = hill_gfs
b = np.zeros(obs_pt_idxs.shape[0] * len(which_dims))
for i in range(rhs_G.shape[0]):
    s = slip_vecs[i,0,:]
    # project the desired slip vector onto the gf slip vectors.
    proj = s.dot(slip) / np.linalg.norm(s)
    # sum to resulting rhs
    b += proj * rhs_G[i,obs_pt_idxs,:][:,which_dims].flatten()

# G = hill_gfs
G = flat_gfs

reg_param = 0.03125
xs = []
# for reg_param in (2.0 ** (-np.arange(16))):
for G in [flat_gfs, hill_gfs]:
    A = G[:,obs_pt_idxs][:,:,which_dims].reshape((G.shape[0], -1)).T
    A += reg_param * np.eye(*A.shape)
    x = np.linalg.lstsq(A, b)[0]
    xs.append(x)
    plt.figure()
    plt.tripcolor(fault[0][:,0], fault[0][:,2], fault[1], x[1::2], shading = 'flat')
    plt.colorbar()
    # plt.savefig(str(reg_param))
    print(reg_param, x)
plt.show()

np.save('data/hill_flat_inversion_' + str(reg_param) + '.npy', xs)
