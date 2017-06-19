import numpy as np
import matplotlib.pyplot as plt
import tectosaur
from tectosaur.geometry import unscaled_normals
import scipy.spatial

def plot_mesh(m):
    plt.figure()
    plt.triplot(m[0][:,0], m[0][:,1], m[1])

w = 1e7
n = 5

corners = [[-w, -w, 0], [-w, w, 0], [w, w, 0], [w, -w, 0]]
orig_surf = tectosaur.make_rect(n, n, corners)

threshold = 1e4
iters = 0
max_iters = 100
inner_ring = 100000.0
inner_ring_area = 10000000.0
m = orig_surf
while iters < max_iters:
    iters += 1
    tri_pts = m[0][m[1]]
    tri_centroids = np.mean(tri_pts, axis = 1)
    dist = np.linalg.norm(tri_centroids, axis = 1)
    area = np.linalg.norm(unscaled_normals(tri_pts), axis = 1) / 2.0
    refine = area / dist > threshold
    refine = np.where(dist < inner_ring, area > inner_ring_area, refine)
    if np.all(refine == False):
        break
    new_mesh = tectosaur.selective_refine(m, refine)
    new_tris = scipy.spatial.Delaunay(new_mesh[0][:, :2]).simplices
    if new_tris.shape[0] == m[1].shape[0]:
        import ipdb; ipdb.set_trace()
    m = (new_mesh[0], new_tris)
    print('at iter = ' + str(iters) + ' mesh has ' + str(m[0].shape[0]) + ' pts and ' + str(m[1].shape[0]) + ' tris.')

print("done")
plot_mesh(m)
plt.show()
