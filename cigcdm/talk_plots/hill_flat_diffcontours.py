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

all_axes = []
plt.figure(figsize = (12,10))
for d in range(2):
    subplot_idx = int('21' + str(d + 1))
    if d == 0:
        ax = plt.subplot(subplot_idx)
    else:
        ax = plt.subplot(subplot_idx, sharex = all_axes[0])
    all_axes.append(ax)
    ax.set_aspect('equal')

    dimname = ['x', 'y', 'z'][d]
    test = False

    field = (np.abs(hill_disp[:,d] - flat_disp[:,d]) / 0.2) * 100

    cmap = cm.get_cmap(name='hot_r')
    levels = [0, 1, 2, 4, 8, 16, 32]

    subdiv = 2

    if not test:
        triang = tri.Triangulation(surf_pts[:,0], surf_pts[:,1])
        refiner = tri.UniformTriRefiner(triang)
        tri_refi, interp_vals = refiner.refine_field(field, subdiv = subdiv)
        cntf = plt.tricontourf(tri_refi, interp_vals, levels = levels, cmap = cmap)
        plt.tricontour(tri_refi, interp_vals, levels = levels, colors = '#333333', linestyles = 'solid', linewidths = 1.5)

    plt.plot([-35,35], [0,0], 'k-', linewidth = 8)
    plt.plot([-26.8,-30,-24], [2.4,1.3,1.3], 'k-', linewidth = 4)
    plt.plot([-27.2,-24,-30], [-2.4,-1.3,-1.3], 'k-', linewidth = 4)
    plt.plot([27.2,24,30], [2.4,1.3,1.3], 'k-', linewidth = 4)
    plt.plot([26.8,30,24], [-2.4,-1.3,-1.3], 'k-', linewidth = 4)
    plt.ylabel('$y$ (km)')
    plt.ylim([-20, 20])

    if d < 1:
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        plt.xlabel('$x$ (km)')
        plt.xlim([-50, 50])

    if not test:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5)
        cbar = plt.colorbar(cntf, cax = cax)
        cbar.set_label('\% difference $u_' + dimname + '$')
        cbar.set_ticks(levels[::1])
        cbar.set_ticklabels(['{:.0f}%'.format(l) for l in levels[::1]])

plt.savefig('hill_flat_diffcontours_uxy.pdf', bbox_inches = 'tight')
plt.show()
