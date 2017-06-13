import numpy as np
from solve import solve_bem
import tectosaur
from slip_vectors import get_slip_vectors
import pyproj

surf = np.load('data/refined_sphere.npy')

fault_top_depth = 4 * 1000
fault_bottom_depth = 15 * 1000
fault_length = 1.0
fault_n = 20

corners = [
    [-fault_length / 2, 0, -fault_top_depth],
    [fault_length / 2, 0, -fault_top_depth],
    [-fault_length / 2, 0, -fault_bottom_depth],
    [fault_length / 2, 0, -fault_bottom_depth]
]
fault = tectosaur.make_rect(fault_n, fault_n, corners)


wgs84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
proj = pyproj.Proj('+proj=geocent +datum=WGS84 +units=m +no_defs')
x_new,y_new,z_new = pyproj.transform(wgs84, proj, fault[0][:,0], fault[0][:,1], fault[0][:,2])
projected_pts = np.vstack((x_new,y_new,z_new)).T
fault_projected = (projected_pts, fault[1])
v1,v2 = get_slip_vectors(projected_pts[fault[1][0,:]])
# print(v1,v2)
fault_slip = np.array([[0, 1, 0]] * 3 * fault[1].shape[0]).flatten()
#project fault and fault slip

surf_pts, surf_disp = solve_bem(surf, fault_projected, fault_slip, True)
np.save('data/sphere_ss.npy', (surf_pts, surf_disp))
