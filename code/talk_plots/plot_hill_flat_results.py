import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm

reg_param = 0.03125
xs = np.load('data/hill_ss/hill_flat_inversion_' + str(reg_param) + '.npy')
flat_pts, slip_vecs, flat_gfs, flat_surf, fault = np.load('data/hill_ss/flat_ss_gfs.npy')

x = -xs[0][1::2]

# fault_tris = fault[0][fault[1]]
# tri_centers = np.mean(fault_tris, axis = 1)
# plt.plot(tri_centers[:,0], tri_centers[:,2], 'o')
# plt.show()
#
# xs = np.linspace(-35000, 35000, 200)
# zs = np.linspace(-15000, -3000, 200)
# X, Z = np.meshgrid(xs, zs)
# SS = scipy.interpolate.griddata(tri_centers[:,[0,2]], x, (X, Z))
# SS = np.where(np.isnan(SS), 0, SS)
#
# cntf = plt.contourf(X, Z, SS)
# plt.contour(X, Z, SS)
# plt.colorbar(cntf)
# plt.show()

vert_tris = [[] for i in range(fault[0].shape[0])]
for i in range(fault[1].shape[0]):
    for d in range(3):
        vert_tris[fault[1][i,d]].append(i)
vert_n_tris = [len(ts) for ts in vert_tris]
vert_vals = np.zeros(fault[0].shape[0])
for i in range(fault[1].shape[0]):
    for d in range(3):
        vert_vals[fault[1][i,d]] += x[i]
vert_vals /= vert_n_tris

triang = tri.Triangulation(fault[0][:,0] / 1000.0, fault[0][:,2] / 1000.0, fault[1])
refiner = tri.UniformTriRefiner(triang)
tri_refi, z_test_refi = refiner.refine_field(vert_vals, subdiv=3)

plt.figure(figsize = (15, 3.5))
ax = plt.gca()
ax.set_aspect('equal')
# plt.triplot(triang, lw = 0.5, color = 'white')

levels = np.linspace(0.0, 4.5, 10)
# cmap = cm.get_cmap(name='terrain', lut=None)
cmap = cm.get_cmap(name='winter')
cntf = plt.tricontourf(tri_refi, z_test_refi, levels=levels, cmap=cmap)
plt.tricontour(
    tri_refi, z_test_refi, levels=levels,
    linestyles = 'solid',
    colors=['k'],
    linewidths=[0.5]
)


plt.xticks(np.linspace(-35.0, 35.0, 15))
plt.yticks(np.linspace(-3.0, -15.0, 5))
plt.xlabel('$x$ (km)')
plt.ylabel('$z$ (km)')

plt.title('')

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1.5%", pad=0.5)
cbar = plt.colorbar(cntf, cax = cax)
cbar.set_label('$s$ (m)')
# cbar.set_ticks(levels[::2])
# cbar.set_ticklabels(levels[::2])

plt.show()
