import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

from cigcdm.talk_plots.hill_flat_results import get_vert_vals

reg_param = 0.03125
xs = np.load('data/hill_ss/hill_flat_inversion_' + str(reg_param) + '.npy')
flat_pts, slip_vecs, flat_gfs, flat_surf, fault = np.load('data/hill_ss/flat_ss_gfs.npy')

s = xs[0][1::2]

vert_vals = get_vert_vals(fault, s)

z = np.linspace(-3 * 1000, -15 * 1000, 13)
x = 0 * z

SS = scipy.interpolate.griddata(fault[0][:,[0,2]], vert_vals, (x, z))
plt.plot(-SS, z / 1000.0, 'k-o')
plt.ylim([-15,-3])
plt.xlim([0, 4.0])
plt.xlabel('$s$ (m)')
plt.ylabel('$z$ (km)')
plt.yticks(np.linspace(-3.0, -15.0, 5))

plt.savefig('data/hill_vertical_xsec.pdf')
