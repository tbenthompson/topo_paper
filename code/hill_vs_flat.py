import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

flat_pts, flat_gfs, flat_surf, fault = np.load('flat_gfs.npy')
hill_pts, hill_gfs, hill_surf, _ = np.load('hill_gfs.npy')

w = 10
xi = np.linspace(-w,w,100)
yi = np.linspace(-w,w,100)

z = hill_pts[:, 2]
zi = griddata((hill_pts[:,0], hill_pts[:,1]), z, (xi[None,:], yi[:,None]), method='cubic')

min_z, max_z = np.min(z), np.max(z)
print(max_z)
levels = np.linspace(min_z, max_z, 15)[1:]
CS = plt.contour(xi,yi,zi,levels = levels, linewidths=0.5, colors='k')

min_faultx = np.min(fault[0][:,0])
max_faultx = np.max(fault[0][:,0])
plt.plot([min_faultx, max_faultx], [0,0], 'r-')
plt.show()

# G = hill_gfs
G = flat_gfs
# reg_param = 0.0#025
which_dims = [0,1]
for reg_param in (2.0 ** (-np.arange(16))):
    A = G[:,:,which_dims].reshape((G.shape[0], -1)).T
    A += reg_param * np.eye(*A.shape)
    b = np.sum(hill_gfs, axis = 0)[:,:2].flatten()
    x = np.linalg.lstsq(A, b)[0]
    # import ipdb; ipdb.set_trace()
    plt.tripcolor(fault[0][:,0], fault[0][:,2], fault[1], x, shading = 'flat')
    plt.colorbar()
    # plt.savefig(str(reg_param))
    plt.show()
    print(reg_param, x)

# use only ux, uy, one row for each observation point.
# from build_gf import plot_gf
# plot_gf(hill_pts, hill_surf[1], hill_gfs[0])
