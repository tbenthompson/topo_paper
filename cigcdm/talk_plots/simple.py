import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt


# plt.figure()
# plt.plot([-10, 10], [0,0], 'k-')
# plt.plot([
# ax = plt.gca()
# ax.spines["top"].set_visible(False)
# ax.spines["bottom"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_visible(False)
# plt.show()
#
# import sys; sys.exit()


def remove_duplicate_pts(m):
    threshold = (np.max(m[0]) - np.min(m[0])) * 1e-13
    idx_map = dict()
    next_idx = 0
    for i in range(m[0].shape[0]):
        dists = np.sum((m[0][:i] - m[0][i,:]) ** 2, axis = 1)
        close = dists < threshold ** 2
        if np.sum(close) > 0:
            replacement_idx = np.argmax(close)
            idx_map[i] = idx_map[replacement_idx]
        else:
            idx_map[i] = next_idx
            next_idx += 1

    n_pts_out = np.max(list(idx_map.values())) + 1
    out_pts = np.empty((n_pts_out, 3))
    for i in range(m[0].shape[0]):
        out_pts[idx_map[i],:] = m[0][i,:]

    out_tris = np.empty_like(m[1])
    for i in range(m[1].shape[0]):
        for d in range(3):
            out_tris[i,d] = idx_map[m[1][i,d]]

    return out_pts, out_tris

def refine(m):
    pts, tris = m
    c0 = pts[tris[:,0]]
    c1 = pts[tris[:,1]]
    c2 = pts[tris[:,2]]
    midpt01 = (c0 + c1) / 2.0
    midpt12 = (c1 + c2) / 2.0
    midpt20 = (c2 + c0) / 2.0
    new_pts = np.vstack((pts, midpt01, midpt12, midpt20))
    new_tris = []
    first_new = pts.shape[0]
    ntris = tris.shape[0]
    for i, t in enumerate(tris):
        new_tris.append((t[0], first_new + i, first_new + 2 * ntris + i))
        new_tris.append((t[1], first_new + ntris + i, first_new + i))
        new_tris.append((t[2], first_new + 2 * ntris + i, first_new + ntris + i))
        new_tris.append((first_new + i, first_new + ntris + i, first_new + 2 * ntris + i))
    new_tris = np.array(new_tris)
    return remove_duplicate_pts((new_pts, new_tris))

pts = np.array([[0,0,0], [1,0,0],[0,1,-1], [1,1,-1]], dtype = np.float64)
tris = np.array([[0,1,2],[2,1,3]])

for i in range(3):
    pts, tris = refine((pts, tris))

mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], tris, color = (0,0,0))
mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], tris, representation = 'wireframe', color = (1,1,1))

w = 5
pts = np.array([[-w,-w,0], [w,-w,0],[-w,w,0], [w,w,0]], dtype = np.float64)
tris = np.array([[0,1,2],[2,1,3]])

for i in range(5):
    pts, tris = refine((pts, tris))

mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], tris, color = (0,0,0))
mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], tris, representation = 'wireframe', color = (1,1,1))

mlab.show()
