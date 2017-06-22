import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cigcdm.plot_disp import interp_for_plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri
import matplotlib.cm as cm

matplotlib.rcParams.update({'font.size': 22})

surf_pts, flat_disp = np.load('data/hill_ss/superhires/flat_ss_disp.npy')
_, hill_disp = np.load('data/hill_ss/superhires/hill_ss_disp.npy')
# convert to kms
surf_pts /= 1000.0

d = 0

fields = [flat_disp[:,d], hill_disp[:,d], hill_disp[:,d] - flat_disp[:,d]]
field_names = ['flat', 'hill', 'diff']
vmin = [-0.2, -0.2, -0.05]
vmax = [0.2, 0.2, 0.05]
for field_idx in range(3):
    field = fields[field_idx]
    triang = tri.Triangulation(surf_pts[:,0], surf_pts[:,1])
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, interp_vals = refiner.refine_field(field, subdiv=2)

    dimname = ['x', 'y', 'z'][d]
    plt.figure(figsize = (12,8))
    ax = plt.gca()
    ax.set_aspect('equal')
    levels = np.linspace(vmin[field_idx], vmax[field_idx], 17)
    cmap = cm.get_cmap(name='PuOr_r')
    cntf = plt.tricontourf(tri_refi, interp_vals, levels = levels, extend = 'both', cmap = cmap)
    plt.tricontour(tri_refi, interp_vals, levels = levels, colors = '#333333', linestyles = 'solid', linewidths = 1.5)
    plt.plot([-35,35], [0,0], 'k-', linewidth = 8)
    plt.plot([-26.8,-30,-24], [2.4,1.3,1.3], 'k-', linewidth = 4)
    plt.plot([-27.2,-24,-30], [-2.4,-1.3,-1.3], 'k-', linewidth = 4)
    plt.plot([27.2,24,30], [2.4,1.3,1.3], 'k-', linewidth = 4)
    plt.plot([26.8,30,24], [-2.4,-1.3,-1.3], 'k-', linewidth = 4)
    plt.xlabel('$x$ (km)')
    plt.ylabel('$y$ (km)')
    plt.xlim([-50, 50])
    plt.ylim([-20,20])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    cbar = plt.colorbar(cntf, cax = cax)
    cbar.set_label('$u_' + dimname + '$ (m)')
    cbar.set_ticks(levels[::2])
    cbar.set_ticklabels(['{:.2f}'.format(l) for l in levels[::2]])
    plt.savefig(field_names[field_idx] + '_ux.pdf', bbox_inches = 'tight')

    plt.show()
