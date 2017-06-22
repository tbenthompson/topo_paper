import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib

from cigcdm.talk_plots.hill_flat_results import get_vert_vals
matplotlib.rcParams['font.size'] = 18

reg_param = 0.03125
xs = np.load('data/hill_ss/hill_flat_inversion_' + str(reg_param) + '.npy')
flat_pts, slip_vecs, flat_gfs, flat_surf, fault = np.load('data/hill_ss/flat_ss_gfs.npy')

s = xs[0][1::2]

vert_vals = get_vert_vals(fault, s)

z = np.linspace(-3 * 1000, -15 * 1000, 13)
x = 0 * z

plt.figure(figsize = (7.5,8.6))
SS = scipy.interpolate.griddata(fault[0][:,[0,2]], vert_vals, (x, z))
plt.plot(-SS / np.max(np.abs(SS)), z / 1000.0, 'b-o', linewidth = 4, markersize = 10)
plt.ylim([-12,0])
plt.xlim([0, 1.0])
plt.xlabel('Normalized moment')
plt.ylabel('$z$ (km)')
plt.yticks(np.linspace(-0.0, -15.0, 6))

plt.savefig('hill_vertical_xsec.pdf', bbox_inches='tight')
plt.show()
