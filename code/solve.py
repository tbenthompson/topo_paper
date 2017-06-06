import numpy as np
import scipy.sparse.linalg

import tectosaur

def iterative_solve(iop, constraints, rhs = None):
    cm, c_rhs = tectosaur.build_constraint_matrix(constraints, iop.shape[1])
    cm = cm.tocsr()
    cmT = cm.T
    if rhs is None:
        rhs_constrained = cmT.dot(-iop.dot(c_rhs))
    else:
        rhs_constrained = cmT.dot(rhs - iop.dot(c_rhs))

    n = rhs_constrained.shape[0]

    iter = [0]
    def mv(v):
        iter[0] += 1
        out = cmT.dot(iop.dot(cm.dot(v)))
        return out

    P = scipy.sparse.linalg.spilu(cmT.dot(iop.nearfield_no_correction_dot(cm)))
    def prec_f(x):
        return P.solve(x)
        # return x
    M = scipy.sparse.linalg.LinearOperator((n, n), matvec = prec_f)
    A = scipy.sparse.linalg.LinearOperator((n, n), matvec = mv)

    def report_res(R):
        tectosaur.logger.debug('residual at iteration ' + str(iter[0]) + ': ' + str(R))
        pass
    soln = scipy.sparse.linalg.gmres(
        A, rhs_constrained, M = M, tol = 1e-8, callback = report_res, restart = 200
    )
    return cm.dot(soln[0]) + c_rhs
