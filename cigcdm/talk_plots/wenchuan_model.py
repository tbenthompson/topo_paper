import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np

matplotlib.rcParams.update({'font.size': 20})

LON, LAT, DEM = np.load('data/wenchuan/narrow/lonlatdem.npy')
trace_vs, trace_segs = np.load('data/wenchuan/wenchuan_trace.npy', encoding='latin1')

cmap = cm.get_cmap(name='PuOr_r')

ax = plt.gca()
# levels = np.linspace(np.min(DEM), np.max(DEM), 15)
levels = np.linspace(0,5000,11)
cntf = plt.contourf(LON, LAT, DEM, levels = levels, cmap = cmap)
plt.contour(LON, LAT, DEM, levels = levels, linestyles = 'solid', colors = 'k', linewidths = 1.0)
cbar = plt.colorbar(cntf)
cbar.set_ticks(levels[::2])
cbar.set_ticklabels(['{:.0f}'.format(l) for l in levels[::2]])
cbar.set_label('Elevation (m)')
for i in range(trace_segs.shape[0]):
    vs = trace_vs[trace_segs[i,:]]
    plt.plot(vs[:,1], vs[:,0], color = 'k', linewidth = 4.5)
plt.xlim([102, 105.5])
plt.ylim([30.5,33])
plt.xticks([102,103,104,105])
plt.yticks([31,32,33])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# ax.spines["top"].set_visible(False)
# ax.spines["bottom"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_visible(False)
plt.savefig('wenchuan_dem.pdf', bbox_inches = 'tight')
plt.show()

# plt.imshow(np.flipud(DEM))
# plt.show()
