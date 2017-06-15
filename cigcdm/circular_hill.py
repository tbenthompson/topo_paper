import numpy as np
import matplotlib.pyplot as plt

import tectosaur
from tectosaur.geometry import tri_normal
import solve

def build_greens_functions(surf, fault, fault_refine_size, basis_idx):
    gfs = []

    if basis_idx is None:
        slip = np.zeros((1,3,3))
        slip[0,:,0] = 1
    else:
        slip = np.zeros((1, 3, 3))
        slip[0,basis_idx,0] = 1

    for i in range(fault[1].shape[0]):
        subfault_pts = fault[0][fault[1][i,:]]
        subfault_tris = [[0,1,2]]
        subfault_unrefined = [np.array(subfault_pts), np.array(subfault_tris)]
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
        gfs.append(result[1])
    gfs = np.array(gfs)
    return surf_pts, gfs

def build_save_greens_functions(filename, surf, fault, fault_refine_size):
    # surf_pts, gfs0 = build_greens_functions(surf, fault, fault_refine_size, 0)
    # _, gfs1 = build_greens_functions(surf, fault, fault_refine_size, 1)
    # _, gfs2 = build_greens_functions(surf, fault, fault_refine_size, 2)
    surf_pts, gfs = build_greens_functions(surf, fault, fault_refine_size, None)
    np.save(filename, [surf_pts, gfs, surf, fault])

def main():
    # import logging
    # tectosaur.logger.setLevel(logging.INFO)

    w = 10
    n = 41

    flat_surf_corners = [[-w, -w, 0], [-w, w, 0], [w, w, 0], [w, -w, 0]]
    flat_surf = tectosaur.make_rect(n, n, flat_surf_corners)
    hill_surf = (flat_surf[0].copy(), flat_surf[1].copy())
    x = hill_surf[0][:,0]
    y = hill_surf[0][:,1]
    hill_surf[0][:,2] = 0.1 * np.exp(-(x ** 2 + y ** 2))

    L = 2.0
    top_depth = -0.5
    nx = 6
    fault_refined_size = 0.005
    fault = tectosaur.make_rect(nx, nx, [
        [-L, 0, top_depth], [-L, 0, top_depth - 1],
        [L, 0, top_depth - 1], [L, 0, top_depth]
    ])
    normal = tri_normal(fault[0][fault[1][0,:]], normalize = True)

    print('Building greens functions for ' + str(fault[1].shape[0]) + ' tris')

    build_save_greens_functions('flat_gfs.npy', flat_surf, fault, fault_refined_size)
    build_save_greens_functions('hill_gfs.npy', hill_surf, fault, fault_refined_size)

    # For each observation point, find the surface triangle that it is within
    # Get the reference triangle coordinates for the observation point
    # Interpolate the green's function from that surface triangle onto the true pt
    # Add that as a row in the inversion matrix
    # Invert!

if __name__ == '__main__':
    main()
