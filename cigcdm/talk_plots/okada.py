import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams.update({'font.size': 22})

def okada_plots():
    soln, vals, obs_pts, surface_tris, fault_L, top_depth, sm, pr = np.load('data/okada/okada.npy')

    for d in range(3):
        dimname = ['x', 'y', 'z'][d]
        triang = tri.Triangulation(obs_pts[:,0], obs_pts[:,1], surface_tris)
        refiner = tri.UniformTriRefiner(triang)
        tri_refi, z_test_refi = refiner.refine_field(vals[:,d], subdiv=2)
        vmin = np.min(z_test_refi)
        vmax = np.max(z_test_refi)

        plt.figure(figsize = (10, 10))
        ax = plt.gca()
        ax.set_aspect('equal')
        # plt.triplot(triang, lw = 0.5)
        # plt.show()

        levels = np.linspace(vmin, vmax, 19)
        cmap = cm.get_cmap(name='winter')
        cntf = plt.tricontourf(tri_refi, z_test_refi, levels=levels, cmap=cmap)
        plt.tricontour(
            tri_refi, z_test_refi, levels=levels,
            linestyles = 'solid',
            colors=['k'],
            linewidths=[1.0]
        )
        plt.xlim([-5,5])
        plt.ylim([-5,5])
        plt.xlabel('$x$ (km)')
        plt.ylabel('$y$ (km)')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5)
        cbar = plt.colorbar(cntf, cax = cax)
        cbar.set_label('$u_' + dimname + '$ (m)')
        cbar.set_ticks(levels[::2])
        cbar.set_ticklabels(["{:.3f}".format(x) for x in levels[::2]])
        plt.savefig('okada_u' + dimname + '.pdf', bbox_inches='tight')


def convergence():
    n = np.array([8, 16, 32, 64, 128, 256])
    l2 = np.array([0.0149648012534, 0.030572079265, 0.00867837671259, 0.00105034618493, 6.66984415273e-05, 4.07689295549e-06])
    linf = np.array([0.008971091166208367, 0.014749192806577716, 0.0093510756645549115, 0.0042803891552975898, 0.0013886177492512669, 0.000338113427521])

    h = 20 / n
    plt.figure(figsize = (10,10))
    plt.loglog(h[1:], l2[1:], 'r-o', label = '$L^2$', linewidth = 4, markersize = 10)
    plt.loglog(h[1:], linf[1:], 'b-*', label = '$L^{\infty}$', linewidth = 4, markersize = 10)
    # plt.loglog(h[1:], 0.1 * h[1:] ** 2, 'k-')
    # plt.loglog(h[1:-1], 0.001 * h[1:-1] ** 4, 'k-')
    plt.xlim([10 ** -1.5, 10 ** 1])
    plt.ylabel('Error')
    plt.xlabel('$h$')
    plt.legend()
    plt.savefig('okada_convergence.pdf')
    plt.show()

if __name__ == '__main__':
    convergence()
