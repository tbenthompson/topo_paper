import numpy as np
import tectosaur
import tectosaur.util.gpu as gpu
from slip_vectors import get_slip_vectors
from solve import solve_bem
from multi_gpu import how_many_gpus, use_gpu
import multiprocessing

def make_tri_greens_functions(args):
    proc_idx = int(multiprocessing.current_process().name.split('-')[1]) - 1
    if not gpu.gpu_initialized:
        gpu_idx = proc_idx % how_many_gpus()
        print('using gpu #' + str(gpu_idx))
        use_gpu(gpu_idx)
    surf, fault, fault_refine_size, basis_idx, i = args
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

def build_greens_functions(surf, fault, fault_refine_size, basis_idx):
    pool = multiprocessing.Pool(6)
    input = [(surf, fault, fault_refine_size, basis_idx, i) for i in range(fault[1].shape[0])]
    results = list(pool.map(make_tri_greens_functions, input))

    surf_pts = results[0][0]
    slip_vecs = np.array([r[1] for r in results]).reshape((-1, 3, 3))
    gfs = np.array([r[2] for r in results]).reshape((-1, surf_pts.shape[0], 3))
    return surf_pts, slip_vecs, gfs

def build_tri_greens_functions(surf, fault, fault_refine_size):
    surf_pts, slip_vecs, gfs = build_greens_functions(surf, fault, fault_refine_size, None)
    return surf_pts, slip_vecs, gfs, surf, fault

# def build_basis_greens_functions(surf, fault, fault_refine_size):
#     surf_pts, slip_vecs0, gfs0 = build_greens_functions(surf, fault, fault_refine_size, 0)
#     _, slip_vecs1, gfs1 = build_greens_functions(surf, fault, fault_refine_size, 1)
#     _, slip_vecs2, gfs2 = build_greens_functions(surf, fault, fault_refine_size, 2)

