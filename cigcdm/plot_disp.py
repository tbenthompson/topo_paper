import sys
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

def interp_for_plotting(pts, vals):
    xs = np.linspace(np.min(pts[:,0]), np.max(pts[:,0]), 500)[1:-1]
    ys = np.linspace(np.min(pts[:,1]), np.max(pts[:,1]), 500)[1:-1]
    X,Y = np.meshgrid(xs, ys)

    interp_vals = []
    for d in range(3):
        interp_vals.append(scipy.interpolate.griddata((pts[:,0], pts[:,1]), vals[:,d], (X, Y)))
    interp_vals = np.array(interp_vals)
    interp_vals[np.isnan(interp_vals)] = 0.0
    return X, Y, interp_vals

def plot_disp(X, Y, interp_vals):
    for d in range(3):
        plt.figure(figsize = (12,12))
        levels = np.linspace(np.min(interp_vals[d]), np.max(interp_vals[d]), 21)
        cntf = plt.contourf(X, Y, interp_vals[d], levels = levels)
        plt.contour(X, Y, interp_vals[d], levels = levels, colors = '#333333', linestyles = 'solid', linewidths = 0.25)
        plt.colorbar(cntf)
        plt.savefig('data/cntf_u_' + ['x', 'y', 'z'][d] + '.pdf')
    plt.show()

def main():
    filename = sys.argv[1]
    surf_pts, disp = np.load(filename)
    X, Y, interp_vals = interp_for_plotting(surf_pts, disp)
    plot_disp(X, Y, interp_vals)

if __name__ == '__main__':
    main()
