import sys
import numpy as np
import pyproj
import scipy.spatial
import scipy.sparse.csgraph as graph
import matplotlib.pyplot as plt
from tectosaur.mesh import remove_duplicate_pts

LON, LAT, DEM = [arr.flatten() for arr in np.load('lonlatdem.npy', encoding = 'latin1')]
tris = scipy.spatial.Delaunay(np.vstack((LON, LAT)).T).simplices.copy()
f_lon, f_lat, f_z, f_tris = [np.array(arr) for arr in np.load('wenchuan_fault_surf.npy', encoding = 'latin1')]
f_pts = np.vstack((f_lon,f_lat,f_z)).T
f_pts, f_tris = remove_duplicate_pts((f_pts, f_tris))

plt.figure()
plt.triplot(LON, LAT, tris)
plt.triplot(f_pts[:,0], f_pts[:,1], f_tris)

proj_name = sys.argv[1]

wgs84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
if proj_name == 'ellps':
    proj = pyproj.Proj('+proj=geocent +datum=WGS84 +units=m +no_defs')
elif proj_name == 'utm':
    proj = pyproj.Proj("+proj=utm +zone=48R, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

x,y,z = pyproj.transform(wgs84, proj, LON, LAT, DEM)
fx,fy,fz = pyproj.transform(wgs84, proj, f_pts[:,0], f_pts[:,1], f_pts[:,2])

projected_pts = np.vstack((x,y,z)).T
projected_f_pts = np.vstack((fx,fy,fz)).T

np.save(proj_name + '_surf_mesh.npy', (projected_pts, tris))
np.save(proj_name + '_fault_mesh.npy', (projected_f_pts, f_tris))


plt.figure()
plt.triplot(projected_pts[:,0], projected_pts[:,1], tris)
#remove triangles with all z < 19800
plt.triplot(projected_f_pts[:,0], projected_f_pts[:,1], f_tris)
plt.show()
