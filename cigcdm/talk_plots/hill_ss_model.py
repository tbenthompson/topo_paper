import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.cm as cm
from cigcdm.hill_ss import topo

matplotlib.rcParams.update({'font.size': 20})

hill_pts, slip_vecs, hill_gfs, hill_surf, fault = np.load('data/hill_ss/hill_ss_gfs.npy')

x = np.linspace(-40000, 40000)
y = 0 * x
z = topo(x, y, flat = False)

fig = plt.figure(figsize = (15, 8.5))
ax1 = plt.subplot(211)
# ax1.set_aspect('equal')
triang = tri.Triangulation(fault[0][:,0] / 1000.0, fault[0][:,2] / 1000.0, fault[1])
plt.triplot(triang)
plt.xticks(np.linspace(-35.0, 35.0, 15))
plt.yticks(np.linspace(3.0, -15.0, 7))
plt.ylim([-16, 3])
# plt.xlabel('$x$ (km)')
plt.ylabel('$z$ (km)')
plt.plot(x / 1000, z / 1000, 'k-')
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="3%", pad=0.5)
cax.axis('off')
plt.setp(ax1.get_xticklabels(), visible=False)

x, y, z = hill_surf[0][:,0], hill_surf[0][:,1], hill_surf[0][:, 2]
nx = 400
ny = 400
x_new = np.linspace(np.min(x), np.max(x), nx)
y_new = np.linspace(np.min(y), np.max(y), ny)
X, Y = np.meshgrid(x_new, y_new)
Z = topo(X, Y, flat = False)
# Z = scipy.interpolate.griddata(
#     (x, y), z, (X, Y)
# )

ax2 = plt.subplot(212, sharex = ax1)
# ax2.set_aspect('equal')

cmap = cm.get_cmap(name='PuOr_r')
cntf = plt.contourf(X / 1000, Y / 1000, Z, cmap = cmap)
try:
    plt.contour(
        X / 1000, Y / 1000, Z,
        linestyles = 'solid', colors=['k'], linewidths=[1.5]
    )
except ValueError:
    pass
plt.plot([-35.0, 35.0], [0.0, 0.0], 'r-', zorder = 1000)
plt.xlabel('$x$ (km)')
plt.ylabel('$y$ (km)')
plt.xlim([-40, 40])
plt.ylim([-15, 15])

divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="3%", pad=0.5)
cbar = plt.colorbar(cntf, cax = cax)
cbar.set_label('$h$ (m)')
plt.savefig('hill_ss_mapview.pdf')
plt.show()
