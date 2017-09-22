import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tectosaur.geometry import tri_normal, projection, tri_area
from tectosaur.mesh import remove_duplicate_pts

from cigcdm.connected_components import get_connected_components
from cigcdm.talk_plots.hill_flat_results import get_vert_vals
from cigcdm.wenchuan_compare_inverse import get_dip_strike_dirs

matplotlib.rcParams.update({'font.size': 16})

def make_inversion_plot(xs, fault, slip_vecs, field_idx, comp_idx, which_dir):
    x = xs[field_idx]

    fault = remove_duplicate_pts(fault)

    dip_slip = np.empty(fault[1].shape[0])
    for i in range(fault[1].shape[0]):
        dip_dir, strike_dir = get_dip_strike_dirs(fault[0][fault[1][i,:]])
        v = strike_dir if which_dir == 'strike' else dip_dir
        s1 = slip_vecs[2 * i]
        s2 = slip_vecs[2 * i + 1]
        ds = v.dot(s1[0,:] * x[2 * i] + s2[0,:] * x[2 * i + 1])
        dip_slip[i] = ds
    vert_vals = get_vert_vals(fault, dip_slip)

    cc = get_connected_components(fault[1])
    subfault_tri_idx = np.arange(fault[1].shape[0])[cc[1] == comp_idx]
    subfault_tris = fault[1][subfault_tri_idx]
    subfault_vert_idx = np.unique(subfault_tris)
    dip_dir, strike_dir = get_dip_strike_dirs(fault[0][subfault_tris[0]])
    normal_dir = np.cross(dip_dir, strike_dir)
    # fault_center = fault[0][subfault[0]][0,:]
    fault_center = np.mean(np.mean(fault[0][subfault_tris], axis = 1), axis = 0)
    # fault_center[2] = 0
    from_center = fault[0] - fault_center
    dip_dist = from_center.dot(dip_dir)
    strike_dist = from_center.dot(strike_dir)
    normal_dist = from_center.dot(normal_dir)

    dip_dist -= np.max(dip_dist[subfault_vert_idx])
    strike_dist -= np.max(strike_dist[subfault_vert_idx])
    strike_dist *= -1

    # plt.triplot(strike_dist, dip_dist, subfault)
    # plt.show()

    subfault_verts = fault[0].copy()
    subfault_verts[:,0] = strike_dist
    subfault_verts[:,2] = dip_dist
    subfault_verts = subfault_verts[subfault_vert_idx]
    mapped_subfault_tris = np.zeros_like(subfault_tris)
    for i in range(subfault_vert_idx.shape[0]):
        mapped_subfault_tris += np.where(subfault_tris == subfault_vert_idx[i], i, 0)
    subfault = (subfault_verts, mapped_subfault_tris)
    subfault = remove_duplicate_pts(subfault)

    subfault_vert_vals = vert_vals[subfault_vert_idx]

    triang = tri.Triangulation(subfault[0][:,0] / 1000.0, subfault[0][:,2] / 1000.0)
    include = []
    for i in range(triang.triangles.shape[0]):
        x = triang.x[triang.triangles[i]]
        y = triang.y[triang.triangles[i]]
        T = np.array([x,y,0*y]).T
        if tri_area(T) > 0.1:
            include.append(i)
    triang = tri.Triangulation(
        subfault[0][:,0] / 1000.0, subfault[0][:,2] / 1000.0, triang.triangles[include]
    )

    refiner = tri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(subfault_vert_vals, subdiv=3)

    plt.figure(figsize = (15, 3.5))
    ax = plt.gca()
    ax.set_aspect('equal')

    slip_min = -0.5 if which_dir == 'strike' else 0.0
    slip_max = 0.5 if which_dir == 'strike' else 2.0

    levels = np.linspace(slip_min, slip_max, 21)
    # cmap = cm.get_cmap(name='terrain', lut=None)
    cmap = cm.get_cmap(name='PuOr_r')
    cntf = plt.tricontourf(tri_refi, z_test_refi, levels=levels, cmap=cmap, extend = 'both')
    plt.tricontour(
        tri_refi, z_test_refi, levels=levels,
        linestyles = 'solid',
        colors=['k'],
        linewidths=[0.5]
    )


    # plt.xticks(np.linspace(-35.0, 35.0, 15))
    # plt.yticks(np.linspace(-3.0, -15.0, 5))
    plt.xlabel('along strike (km)')
    plt.ylabel('along dip (km)')
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.title('')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.5%", pad=0.5)
    cbar = plt.colorbar(cntf, cax = cax)
    cbar.set_label('$s$ (m)')
    cbar.set_ticks(levels[::4])
    cbar.set_ticklabels(levels[::4])

    field_name = 'topo' if field_idx == 1 else 'flat'
    comp_name = 'beichuan' if comp_idx == 0 else 'pengguan'
    plt.savefig(
        'wenchuan_inv_' + which_dir + '_' + comp_name + '_' + field_name + '.pdf',
        bbox_inches = 'tight'
    )


def main():
    reg_param = 0.025
    xs = np.load('data/wenchuan/narrow/wenchuan_inversion_' + str(reg_param) + '.npy')
    flat_pts, slip_vecs, flat_gfs, flat_surf, fault = np.load('data/wenchuan/narrow/flat_wenchuan_gfs.npy')
    for dir in ['strike', 'dip']:
        for comp_idx in range(2):
            for field_idx in range(2):
                make_inversion_plot(xs, fault, slip_vecs, field_idx, comp_idx, dir)


if __name__ == '__main__':
    main()
