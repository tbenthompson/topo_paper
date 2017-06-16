import sys
import numpy as np

root = sys.argv[1]
n_chunks = int(sys.argv[2])

for i in range(n_chunks):
    filename = root + str(i) + '.npy'
    surf_pts, chunk_slip_vecs, chunk_gfs, surf, fault, chunk_indices = np.load(filename)
    if i == 0:
        slip_vecs = np.empty((fault[1].shape[0] * 2, 3, 3))
        gfs = np.empty((fault[1].shape[0] * 2, surf_pts.shape[0], 3))
    gf_indices = np.tile((2 * np.array(chunk_indices))[:,np.newaxis], (1,2))
    gf_indices[:,1] += 1
    gf_indices = gf_indices.flatten()
    slip_vecs[gf_indices] = chunk_slip_vecs
    gfs[gf_indices] = chunk_gfs
out_filename = root + '.npy'
np.save(out_filename, (surf_pts, slip_vecs, gfs, surf, fault))
