import os, subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri
import matplotlib.cm as cm
import pyproj
import scipy.interpolate
import scipy.ndimage.filters

matplotlib.rcParams.update({'font.size': 22})

surf_pts, topo_disp = np.load('data/wenchuan/narrow/wenchuan_disp.npy')
_, flat_disp = np.load('data/wenchuan/narrow/wenchuan_disp_flat.npy')
# surf_pts, topo_disp = np.load('data/wenchuan/narrow_hires/wenchuan_disp.npy')
# _, flat_disp = np.load('data/wenchuan/narrow_hires/wenchuan_disp_flat.npy')

wgs84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
proj = pyproj.Proj("+proj=utm +zone=48R, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

LON,LAT = pyproj.transform(proj, wgs84, surf_pts[:,0], surf_pts[:,1])

cool_diff = True

def get_filename(d):
    dimname = ['x', 'y', 'z'][d]
    if cool_diff:
        return 'wenchuan_log_diff' + dimname + '.pdf'
    else:
        return 'wenchuan_topo_disp' + dimname + '.pdf'

def run(d):
    dimname = ['x', 'y', 'z'][d]

    plt.figure(figsize = (14,14))
    ax = plt.gca()
    ax.set_aspect('equal')

    if cool_diff:
        field = np.abs(topo_disp[:,d] - flat_disp[:,d])
        cmap = cm.get_cmap(name='hot_r')
        levels = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128]) / 100 * 0.2
        extend = 'neither'
    else:
        field = topo_disp[:,d]
        cmap = cm.get_cmap(name='PuOr_r')
        extend = 'both'
        levels = np.linspace(np.min(field), np.max(field), 17)

    # interpolate onto a regular grid.
    new_lon = np.linspace(np.min(LON), np.max(LON), 300)
    new_lat = np.linspace(np.min(LAT), np.max(LAT), 300)
    NEWLON,NEWLAT = np.meshgrid(new_lon, new_lat)
    field_interp = scipy.interpolate.griddata((LON, LAT), field, (NEWLON, NEWLAT))
    smoothing = 0.4
    field_interp = scipy.ndimage.filters.gaussian_filter(field_interp, smoothing)

    cntf = plt.contourf(NEWLON, NEWLAT, field_interp, levels = levels, extend = extend, cmap = cmap)
    plt.contour(NEWLON, NEWLAT, field_interp, levels = levels, colors = '#333333', linestyles = 'solid', linewidths = 2.0)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim([102, 105.5])
    plt.ylim([30.5,33])
    plt.xticks([102,103,104,105])
    plt.yticks([31,32,33])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    cbar = plt.colorbar(cntf, cax = cax)
    if cool_diff:
        cbar.set_label('difference $u_' + dimname +'$ (m)')
    else:
        cbar.set_label('$u_' + dimname + '$ (m)')
    plt.savefig(get_filename(d), bbox_inches = 'tight')
    # cbar.set_ticks(levels[::2])
    # cbar.set_ticklabels(['{:.2f}'.format(l) for l in levels[::2]])
    # plt.show()

for d in range(3):
    run(d)
    subprocess.call(['mv', get_filename(d), 'talk_figs'])
    os.chdir('./talk_figs')
    subprocess.call(['python', 'converttopng.py', get_filename(d)])
    os.chdir(os.pardir)

# fields = [topo_disp[:,d], flat_disp[:,d], topo_disp[:,d] - flat_disp[:,d]]
# field_names = ['topo', 'flat', 'diff']
# vmin = [-0.2, -0.2, -0.05]
# vmax = [0.2, 0.2, 0.05]


# triang = tri.Triangulation(LON, LAT)
# refiner = tri.UniformTriRefiner(triang)
# tri_refi, interp_vals = refiner.refine_field(field, subdiv=0)
# cntf = plt.tricontourf(tri_refi, interp_vals, levels = levels, extend = extend, cmap = cmap)
# plt.tricontour(tri_refi, interp_vals, levels = levels, colors = '#333333', linestyles = 'solid', linewidths = 1.5)
# plt.xlabel('$x$ (km)')
# plt.ylabel('$y$ (km)')
# plt.xlim([102, 105.5])
# plt.ylim([30.5,33])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.5)
# cbar = plt.colorbar(cntf, cax = cax)
# if cool_diff:
#     cbar.set_label('difference (m)')
# else:
#     cbar.set_label('$u_' + dimname + '$ (m)')
# # cbar.set_ticks(levels[::2])
# # cbar.set_ticklabels(['{:.2f}'.format(l) for l in levels[::2]])
#
# plt.show()
