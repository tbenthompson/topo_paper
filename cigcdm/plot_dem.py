import matplotlib.pyplot as plt
import numpy as np

LON, LAT, DEM = np.load('lonlatdem.npy')
trace_vs, trace_segs = np.load('wenchuan_trace.npy', encoding='latin1')

levels = np.linspace(np.min(DEM), np.max(DEM), 15)
plt.contour(LON, LAT, DEM, levels = levels, linestyles = 'solid', colors = 'k', linewidths = 0.5)
cntf = plt.contourf(LON, LAT, DEM, levels = levels)
plt.colorbar(cntf)
for i in range(trace_segs.shape[0]):
    vs = trace_vs[trace_segs[i,:]]
    plt.plot(vs[:,1], vs[:,0], color = 'k', linewidth = 2.5)
plt.show()

# plt.imshow(np.flipud(DEM))
# plt.show()
