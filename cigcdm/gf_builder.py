import sys
import numpy as np
import tectosaur
import tectosaur.util.gpu as gpu
from cigcdm.slip_vectors import get_slip_vectors
from cigcdm.solve import solve_bem
from cigcdm.multi_gpu import how_many_gpus, use_gpu

def make_tri_greens_functions(surf, fault, fault_refine_size, basis_idx, i):
    gfs = []
    slip_vecs = []
    subfault_pts = fault[0][fault[1][i,:]]
    subfault_tris = [[0,1,2]]
    subfault_unrefined = [np.array(subfault_pts), np.array(subfault_tris)]
    for s in get_slip_vectors(subfault_pts):
        print(subfault_pts)
        print('slip vector is ' + str(s))
        slip = np.zeros((1,3,3))
        if basis_idx is None:
            slip[0,:,:] = s
        else:
            slip[0,basis_idx,:] = s

        subfault, refined_slip = tectosaur.refine_to_size(
            subfault_unrefined, fault_refine_size,
            [slip[:,:,0], slip[:,:,1], slip[:,:,2]]
        )
        print(
            'Building GFs for fault triangle ' + str(i) +
            ' with ' + str(subfault[1].shape[0]) + ' subtris.'
        )
        full_slip = np.concatenate([s[:,:,np.newaxis] for s in refined_slip], 2)
        result = solve_bem(surf, subfault, full_slip.flatten())
        surf_pts = result[0]
        slip_vecs.append(slip[0,:,:])
        gfs.append(result[1])
    return surf_pts, slip_vecs, gfs

def split(a, n):
    # From https://stackoverflow.com/questions/2130016/splitting-a-list-of-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def build_greens_functions(surf, fault, fault_refine_size, basis_idx):
    proc_idx = int(sys.argv[1])
    n_procs = int(sys.argv[2])
    print("Building GFs in " + str(proc_idx) + "/" + str(n_procs))

    if not gpu.gpu_initialized:
        try:
            gpu_idx = proc_idx % how_many_gpus()
            print('using gpu #' + str(gpu_idx))
            use_gpu(gpu_idx)
        except:
            pass

    indices = list(list(split(range(fault[1].shape[0]), n_procs))[proc_idx])
    print(indices)
    results = [
        make_tri_greens_functions(surf, fault, fault_refine_size, basis_idx, i)
        for i in indices
    ]

    surf_pts = results[0][0]
    slip_vecs = np.array([r[1] for r in results]).reshape((-1, 3, 3))
    gfs = np.array([r[2] for r in results]).reshape((-1, surf_pts.shape[0], 3))
    return surf_pts, slip_vecs, gfs, indices

def build_save_tri_greens_functions(surf, fault, fault_refine_size, fileroot):
    surf_pts, slip_vecs, gfs, indices = build_greens_functions(surf, fault, fault_refine_size, None)
    filename = fileroot + str(proc_idx) + '.npy'
    np.save(filename, (surf_pts, slip_vecs, gfs, surf, fault, indices))

# def build_basis_greens_functions(surf, fault, fault_refine_size):
#     surf_pts, slip_vecs0, gfs0 = build_greens_functions(surf, fault, fault_refine_size, 0)
#     _, slip_vecs1, gfs1 = build_greens_functions(surf, fault, fault_refine_size, 1)
#     _, slip_vecs2, gfs2 = build_greens_functions(surf, fault, fault_refine_size, 2)

