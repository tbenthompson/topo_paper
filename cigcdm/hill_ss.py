import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from cigcdm.solve import solve_bem

from tectosaur.mesh.mesh_gen import make_rect
from tectosaur_topo import solve_topo
# from cigcdm.gf_builder import build_save_tri_greens_functions

def plot_model(x, y, z, fault, nx = 100, ny = 100):
    x_new = np.linspace(np.min(x), np.max(x), nx)
    y_new = np.linspace(np.min(y), np.max(y), ny)
    X, Y = np.meshgrid(x_new, y_new)
    Z = scipy.interpolate.griddata(
        (x, y), z, (X, Y)
    )
    plt.figure(figsize = (10,10))

    # cntf = plt.contourf(X, Y, Z)
    # try:
    #     plt.contour(X, Y, Z)
    # except ValueError:
    #     pass
    # plt.colorbar(cntf)
    plt.triplot(fault[0][:,0], fault[0][:,1], fault[1], linewidth = 1.5, color = 'r', zorder = 1000)
    plt.savefig('hill_ss_model.pdf')
    plt.show()

def topo(x, y, flat):
    z = np.zeros_like(x)
    if flat:
        return z

    C = [0, 0]
    hill_height = 2 * 1000
    hill_xradius = 7 * 1000
    hill_yradius = 7 * 1000

    z += hill_height * np.exp(
        -(((x - C[0]) / hill_xradius) ** 2 + ((y - C[1]) / hill_yradius) ** 2)
    )

    return z

def make_surface(flat = False):
    wx = 100 * 1000
    wy = 100 * 1000
    n = 81
    # n = 201
    surf_corners = [[-wx, -wy, 0], [-wx, wy, 0], [wx, wy, 0], [wx, -wy, 0]]
    surf = make_rect(n, n, surf_corners)
    surf[0][:,2] = topo(surf[0][:,0], surf[0][:,1], flat)
    return surf

def make_fault():
    fault_top_z = -3 * 1000
    fault_bottom_z = -15 * 1000
    fault_length = 70 * 1000
    fault_nx = int(fault_length / 2800)
    fault_ny = int((fault_top_z - fault_bottom_z) / 2000)
    # fault_nx = 40
    # fault_ny = 40

    corners = [
        [-fault_length / 2, 0, fault_bottom_z],
        [fault_length / 2, 0, fault_bottom_z],
        [fault_length / 2, 0, fault_top_z],
        [-fault_length / 2, 0, fault_top_z],
    ]
    fault = make_rect(fault_nx, fault_ny, corners)
    fault_slip = np.array([[1, 0, 0]] * 3 * fault[1].shape[0]).flatten()
    return fault, fault_slip

def forward_model(model):
    surf = make_surface(model == 'flat')
    fault, fault_slip = make_fault()
    plot_model(surf[0][:,0], surf[0][:,1], surf[0][:,2], fault)
    # surf_pts, surf_disp = solve_bem(surf, fault, fault_slip, True)
    surf_pts, surf_disp, soln = solve_topo(surf, fault, fault_slip, 1.0, 0.25)
    np.save('data/' + model + '_ss_disp.npy', (surf_pts, surf_disp))

def main():
    for model in ['flat','hill']:
        forward_model(model)
        # surf = make_surface(model == 'flat')
        # fault, fault_slip = make_fault()
        # # plot_model(surf[0][:,0], surf[0][:,1], surf[0][:,2], fault)
        # np.save('data/' + model + '_ss_gfs.npy', build_tri_greens_functions(surf, fault, 500000))

if __name__ == '__main__':
    main()
