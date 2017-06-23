import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np

matplotlib.rcParams.update({'font.size': 20})

LON, LAT, DEM = np.load('data/wenchuan/narrow/lonlatdem.npy')
trace_vs, trace_segs = np.load('data/wenchuan/wenchuan_trace.npy', encoding='latin1')

cmap = cm.get_cmap(name='PuOr_r')

ax = plt.gca()
levels = np.linspace(np.min(DEM), np.max(DEM), 15)
cntf = plt.contourf(LON, LAT, DEM, levels = levels, cmap = cmap)
plt.contour(LON, LAT, DEM, levels = levels, linestyles = 'solid', colors = 'k', linewidths = 0.5)
plt.colorbar(cntf)
for i in range(trace_segs.shape[0]):
    vs = trace_vs[trace_segs[i,:]]
    plt.plot(vs[:,1], vs[:,0], color = 'k', linewidth = 2.5)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.show()

# plt.imshow(np.flipud(DEM))
# plt.show()
