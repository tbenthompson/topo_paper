import numpy as np
import tectosaur
import scipy.interpolate
import matplotlib.pyplot as plt
from solve import solve_bem

def plot_model(surf, fault, nx = 100, ny = 100):
    x_new = np.linspace(np.min(surf[0][:,0]), np.max(surf[0][:,0]), nx)
    y_new = np.linspace(np.min(surf[0][:,1]), np.max(surf[0][:,1]), ny)
    X, Y = np.meshgrid(x_new, y_new)
    Z = scipy.interpolate.griddata(
        (surf[0][:,0], surf[0][:,1]), surf[0][:,2], (X, Y)
    )
    Z = np.where(np.isnan(Z), 0, Z)
    plt.figure(figsize = (10,10))
    levels = np.linspace(np.min(Z) - 1, np.max(Z), 12)
    cntf = plt.contourf(X, Y, Z, levels = levels)
    plt.contour(X, Y, Z, levels = levels)
    plt.colorbar(cntf)
    plt.triplot(fault[0][:,0], fault[0][:,1], fault[1], linewidth = 0.5, color = 'k', zorder = 1000)
    plt.savefig('subduction_model.pdf')

    plt.figure(figsize = (10,10))
    cntf = plt.contourf(X, Y, Z, levels = levels)
    plt.contour(X, Y, Z, levels = levels)
    plt.colorbar(cntf)
    plt.triplot(surf[0][:,0], surf[0][:,1], surf[1], linewidth = 0.5, color = 'k', zorder = 1000)
    plt.savefig('subduction_model_surf_mesh.pdf')

    plt.figure(figsize = (10,10))
    plt.plot(surf[0][:,1], surf[0][:,2], '.')
    plt.triplot(fault[0][:,1], fault[0][:,2], fault[1], linewidth = 0.5, color = 'k', zorder = 1000)
    plt.savefig('subduction_xsec.pdf')

    plt.figure(figsize = (10,10))
    plt.contourf(X, Y, Z > 0)
    plt.savefig('subduction_abovesealevel.pdf')
    plt.show()

no_topo = True
island_centers = []#[[0,0]]
island_height = 1000
island_xradius = 0.9e5
island_yradius = 5e4

mainland_shallowness = 10 * 1000
mainland_elevation = 1000.0
mainland_edge = 150 * 1000

ocean_edge = -70 * 1000
ocean_elevation = 2000
ocean_shallowness = 10 * 1000

fault_dip_deg = 10
fault_surface_intersection_y = ocean_edge
fault_trench_creep_end = 15 * 1000
fault_strike_width = 300 * 1000
fault_dip_width = 250 * 1000
fault_n = 30

def topo(x, y):
    z = np.zeros_like(x)
    if no_topo:
        return z
    z += mainland_elevation * (np.arctan((y - mainland_edge) / mainland_shallowness) / np.pi)
    z += ocean_elevation * (np.arctan((y - ocean_edge) / ocean_shallowness) / np.pi - 0.5)
    for C in island_centers:
        z += island_height * np.exp(
            -(((x - C[0]) / island_xradius) ** 2 + ((y - C[1]) / island_yradius) ** 2)
        )

    return z

def make_fault():
    fault_dip_rad = np.deg2rad(fault_dip_deg)

    trench_z = topo(-fault_strike_width / 2, fault_surface_intersection_y)
    upper_y = fault_surface_intersection_y + np.cos(fault_dip_rad) * fault_trench_creep_end
    upper_z = trench_z - np.sin(fault_dip_rad) * fault_trench_creep_end
    upper_west_corner = [-fault_strike_width / 2, upper_y, upper_z]
    upper_east_corner = [fault_strike_width / 2, upper_y, upper_z]

    bottom_y = fault_surface_intersection_y + np.cos(fault_dip_rad) * fault_dip_width
    bottom_z = trench_z - np.sin(fault_dip_rad) * fault_dip_width
    bottom_west_corner = [-fault_strike_width / 2, bottom_y, bottom_z]
    bottom_east_corner = [fault_strike_width / 2, bottom_y, bottom_z]

    corners = np.array([bottom_west_corner, upper_west_corner, upper_east_corner, bottom_east_corner])

    fault = tectosaur.make_rect(fault_n, fault_n, corners)
    fault_slip = np.array(
        [[0, -np.cos(fault_dip_rad), np.sin(fault_dip_rad)]] * 3 * fault[1].shape[0]
    ).flatten()
    return fault, fault_slip

def make_surface():
    wx = 1000 * 1000
    wy = 1000 * 1000
    n = 121
    surf_corners = [[-wx, -wy, 0], [-wx, wy, 0], [wx, wy, 0], [wx, -wy, 0]]
    surf = tectosaur.make_rect(n, n, surf_corners)
    surf[0][:,2] = topo(surf[0][:,0], surf[0][:,1])
    return surf

def main():
    surf = make_surface()
    print(surf[1].shape)
    fault, fault_slip = make_fault()
    plot_model(surf, fault)
    surf_pts, surf_disp = solve_bem(surf, fault, fault_slip, True)
    np.save('data/subduction_disp.npy', (surf_pts, surf_disp))

if __name__ == '__main__':
    main()
