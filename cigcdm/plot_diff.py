import sys
import numpy as np
from solve import plot_result
from plot_disp import interp_for_plotting, plot_disp

def main():
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    pts, disp1 = np.load(filename1)
    pts2, disp2 = np.load(filename2)
    np.testing.assert_almost_equal(pts[:,0], pts2[:,0])
    np.testing.assert_almost_equal(pts[:,1], pts2[:,1])
    X, Y, interp_vals = interp_for_plotting(pts, disp1 - disp2)
    plot_disp(X, Y, interp_vals, filename_prefix = 'diff')

if __name__ == '__main__':
    main()
