import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt

import tectosaur

def iterative_linsolve(iop, constraints, rhs = None):
    cm, c_rhs = tectosaur.build_constraint_matrix(constraints, iop.shape[1])
    cm = cm.tocsr()
    cmT = cm.T
    if rhs is None:
        rhs_constrained = cmT.dot(-iop.dot(c_rhs))
    else:
        rhs_constrained = cmT.dot(rhs - iop.dot(c_rhs))
    n = rhs_constrained.shape[0]

    P = scipy.sparse.linalg.spilu(cmT.dot(iop.nearfield_no_correction_dot(cm)))
    def prec_f(x):
        return P.solve(x)
        # return x
    M = scipy.sparse.linalg.LinearOperator((n, n), matvec = prec_f)

    iter = [0]
    def mv(v):
        iter[0] += 1
        out = cmT.dot(iop.dot(cm.dot(v)))
        return out
    A = scipy.sparse.linalg.LinearOperator((n, n), matvec = mv)

    def report_res(R):
        tectosaur.logger.debug('residual at iteration ' + str(iter[0]) + ': ' + str(R))
        pass
    soln = scipy.sparse.linalg.gmres(
        A, rhs_constrained, M = M, tol = 1e-14, callback = report_res, restart = 200
    )
    return cm.dot(soln[0]) + c_rhs

def plot_result(pts, tris, disp):
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
    soln = iterative_linsolve(iop, cs)

    surf_pts, surf_disp = m.extract_pts_vals('surf', soln)

    if should_plot:
        plot_result(surf_pts, m.get_piece_tris('surf'), surf_disp)

    return surf_pts, surf_disp
