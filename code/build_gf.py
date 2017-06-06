import numpy as np
import matplotlib.pyplot as plt

import tectosaur
from tectosaur.geometry import tri_normal
import solve

def plot_gf(pts, tris, disp):
    vmax = np.max(disp)
    for d in range(3):
        plt.figure()
        plt.tripcolor(
            pts[:,0], pts[:, 1], tris, disp[:,d], #shading='gouraud',
            cmap = 'PuOr', vmin = -vmax, vmax = vmax
        )
        plt.title("u " + ['x', 'y', 'z'][d])
        plt.colorbar()
    plt.show()

def solve_bem(surf, fault, fault_slip, should_plot = False):
    sm = 1.0
    pr = 0.25

    m = tectosaur.CombinedMesh([('surf', surf), ('fault', fault)])

    cs = tectosaur.continuity_constraints(
        m.get_piece_tris('surf'), m.get_piece_tris('fault'), m.pts
    )
    cs.extend(tectosaur.all_bc_constraints(
        m.get_start('fault'), m.get_past_end('fault'), fault_slip
    ))
    cs.extend(tectosaur.free_edge_constraints(m.get_piece_tris('surf')))

    iop = tectosaur.SparseIntegralOp(
        [], 0, 0, 6, 2, 6, 4.0,
        'H', sm, pr, m.pts, m.tris,
        use_tables = True,
        remove_sing = True
    )
    soln = solve.iterative_solve(iop, cs)

    surf_pts, surf_disp = m.extract_pts_vals('surf', soln)

    if should_plot:
        plot_gf(surf_pts, m.get_piece_tris('surf'), surf_disp)

    return surf_pts, surf_disp

def build_greens_functions(surf, fault, fault_refine_size, basis_idx):
    gfs = []

    #TODO: Here, I could find the two fault parallel vector directions by:
    # -- dot product [1, 0, 0] and [0, 1, 0] with the normal to check that they aren't the same vector
    # -- cross [1,0,0] with normal to find the first surface parallel direction
    # -- cross surface parallel direction 1 with normal to get surface parallel direction 2.
    # -- normalize everything.
    # slip = np.array([[1, 0, 0]] * 3 * subfault[1].shape[0]).flatten()
    # print(normal.dot(slip))

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
