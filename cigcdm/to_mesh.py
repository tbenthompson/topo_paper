import sys
import numpy as np
import pyproj
import scipy.spatial
# import matplotlib.pyplot as plt

LON, LAT, DEM = [arr.flatten() for arr in np.load(sys.argv[1], encoding = 'latin1')]
tris = scipy.spatial.Delaunay(np.vstack((LON, LAT)).T).simplices.copy()
# f_pts, f_tris = np.load('data/without_detachment.npy', encoding = 'latin1')

# plt.figure()
# plt.triplot(LON, LAT, tris)
# plt.triplot(f_pts[:,0], f_pts[:,1], f_tris)

proj_name = sys.argv[2]

wgs84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
if proj_name == 'ellps':
    proj = pyproj.Proj('+proj=geocent +datum=WGS84 +units=m +no_defs')
elif proj_name == 'utm':
    proj = pyproj.Proj("+proj=utm +zone=48R, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

x,y,z = pyproj.transform(wgs84, proj, LON, LAT, DEM)
# fx,fy,fz = pyproj.transform(wgs84, proj, f_pts[:,0], f_pts[:,1], f_pts[:,2])

projected_pts = np.vstack((x,y,z)).T
# projected_f_pts = np.vstack((fx,fy,fz)).T

np.save('data/' + proj_name + '_surf_mesh.npy', (projected_pts, tris))
# np.save('data/' + proj_name + '_fault_mesh.npy', (projected_f_pts, f_tris))


# plt.figure()
# plt.triplot(projected_pts[:,0], projected_pts[:,1], tris)
# plt.triplot(projected_f_pts[:,0], projected_f_pts[:,1], f_tris)
# plt.show()

# from mayavi import mlab
# mlab.triangular_mesh(
#     projected_pts[:,0], projected_pts[:,1], projected_pts[:,2], tris,
#     representation = 'wireframe'
# )
# mlab.triangular_mesh(projected_f_pts[:,0], projected_f_pts[:,1], projected_f_pts[:,2], f_tris)

