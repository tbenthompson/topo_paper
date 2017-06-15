import numpy as np
import tectosaur
from tectosaur.geometry import unscaled_normals
import matplotlib.pyplot as plt
import scipy.spatial
import subprocess
import pyproj

def plot_mesh(m):
    plt.figure()
    plt.triplot(m[0][:,0], m[0][:,1], m[1])

r = 6371000
orig_surf = tectosaur.make_sphere([0,0,0], r, 1)

focal_point = [r, 0, 0]
threshold = 1e4
iters = 0
max_iters = 100
inner_ring = 100000.0
inner_ring_area = 100000000.0
m = orig_surf
while iters < max_iters:
    iters += 1
    tri_pts = m[0][m[1]]
    tri_centroids = np.mean(tri_pts, axis = 1)
    dist = np.linalg.norm(tri_centroids - focal_point, axis = 1)
    area = np.linalg.norm(unscaled_normals(tri_pts), axis = 1) / 2.0
    print(np.max(area / dist))
    refine = area / dist > threshold
    refine = np.where(dist < inner_ring, area > inner_ring_area, refine)
    if np.all(refine == False):
        break
    new_mesh = tectosaur.selective_refine(m, refine)
    new_pts = r * (new_mesh[0] / np.linalg.norm(new_mesh[0], axis = 1)[:, np.newaxis])
    new_tris = scipy.spatial.ConvexHull(new_pts).simplices
    m = (new_pts, new_tris)
    print('at iter = ' + str(iters) + ' mesh has ' + str(m[0].shape[0]) + ' pts and ' + str(m[1].shape[0]) + ' tris.')

print("done")
# plot_mesh(m)
# plt.show()

# convert to WGS84 ellipsoid.
x = m[0][:,0]
y = m[0][:,1]
z = m[0][:,2]
phi = np.arccos(z / r)
latitude = np.rad2deg(phi) - 90
theta = np.arctan2(y, x)
longitude = np.rad2deg(theta)
plt.plot(longitude, latitude, '.')
plt.show()

wgs84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
proj = pyproj.Proj('+proj=geocent +datum=WGS84 +units=m +no_defs')
x_new,y_new,z_new = pyproj.transform(wgs84, proj, longitude, latitude, 0 * longitude)
projected_pts = np.vstack((x_new,y_new,z_new)).T
m_wgs84 = (projected_pts, m[1])

np.save('data/refined_sphere.npy', m_wgs84)
subprocess.Popen(['python2', 'code/plot_3d_mesh.py', 'data/refined_sphere.npy'])
