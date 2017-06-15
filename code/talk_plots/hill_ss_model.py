import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cigcdm.hill_ss import topo

hill_pts, slip_vecs, hill_gfs, hill_surf, fault = np.load('data/hill_ss/hill_ss_gfs.npy')

x = np.linspace(-40, 40)
y = 0 * x
z = topo(x, y)

plt.figure(figsize = (15, 3.5))
ax = plt.gca()
ax.set_aspect('equal')
triang = tri.Triangulation(fault[0][:,0] / 1000.0, fault[0][:,2] / 1000.0, fault[1])
plt.triplot(triang)
plt.xticks(np.linspace(-35.0, 35.0, 15))
plt.yticks(np.linspace(-3.0, -15.0, 5))
plt.ylim([-16, 0])
plt.xlabel('$x$ (km)')
plt.ylabel('$z$ (km)')
plt.plot(x, z, 'k-')
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.savefig('hill_ss_xsec_tris.pdf')
plt.show()

x, y, z = hill_surf[0][:,0], hill_surf[0][:,1], hill_surf[0][:, 2]
nx = 200
ny = 200
x_new = np.linspace(np.min(x), np.max(x), nx)
y_new = np.linspace(np.min(y), np.max(y), ny)
X, Y = np.meshgrid(x_new, y_new)
Z = scipy.interpolate.griddata(
    (x, y), z, (X, Y)
)

plt.figure(figsize = (10,5))
ax = plt.gca()
ax.set_aspect('equal')

cntf = plt.contourf(X / 1000, Y / 1000, Z)
try:
    plt.contour(X / 1000, Y / 1000, Z)
except ValueError:
    pass
plt.plot([-35.0, 35.0], [0.0, 0.0], 'r-', zorder = 1000)
plt.xlabel('$x$ (km)')
plt.ylabel('$y$ (km)')
plt.xlim([-40, 40])
plt.ylim([-15, 15])

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.5)
cbar = plt.colorbar(cntf, cax = cax)
cbar.set_label('$h$ (m)')
plt.savefig('hill_ss_mapview.pdf')
plt.show()
