import numpy as np
import tectosaur
import scipy.interpolate
import matplotlib.pyplot as plt

def plot_model(x, y, z, fault, nx = 100, ny = 100):
    x_new = np.linspace(np.min(x), np.max(x), nx)
    y_new = np.linspace(np.min(y), np.max(y), ny)
    X, Y = np.meshgrid(x_new, y_new)
    Z = scipy.interpolate.griddata(
        (x, y), z, (X, Y)
    )
    plt.figure(figsize = (10,10))
    cntf = plt.contourf(X, Y, Z)
    plt.contour(X, Y, Z)
    plt.colorbar(cntf)
    plt.triplot(fault[0][:,0], fault[0][:,1], fault[1], linewidth = 0.5, color = 'k', zorder = 1000)
    plt.savefig('subduction_model.pdf')

    plt.figure(figsize = (10,10))
    plt.contourf(X, Y, Z > 0)
    plt.savefig('subduction_abovesealevel.pdf')

def topo(x, y):
    island_centers = [[-300 * 1000,0], [0,0], [300 * 1000,0]]
    island_height = 1000
    island_xradius = 1e5
    island_yradius = 3e4

    mainland_shallowness = 10 * 1000
    mainland_elevation = 1000.0
    mainland_edge = 150 * 1000

    z = np.zeros_like(x)
    z += mainland_elevation * (np.arctan((y - mainland_edge) / mainland_shallowness) / np.pi)
    for C in island_centers:
        z += island_height * np.exp(
            -(((x - C[0]) / island_xradius) ** 2 + ((y - C[1]) / island_yradius) ** 2)
        )

    return z


def make_fault():
    fault_dip_deg = 15
    fault_surface_intersection_y = -90 * 1000
    fault_strike_width = 1000 * 1000
    fault_dip_width = 250 * 1000
    fault_n = 5

    fault_dip_rad = np.deg2rad(fault_dip_deg)

    upper_west_corner = [-fault_strike_width / 2, fault_surface_intersection_y, 0]
    upper_west_corner[2] = topo(*upper_west_corner[:2])

    upper_east_corner = [fault_strike_width / 2, fault_surface_intersection_y, 0]
    upper_east_corner[2] = topo(*upper_east_corner[:2])

    bottom_y = fault_surface_intersection_y + np.cos(fault_dip_rad) * fault_dip_width
    bottom_west_z = upper_west_corner[2] - np.sin(fault_dip_rad) * fault_dip_width
    bottom_west_corner = [-fault_strike_width / 2, bottom_y, bottom_west_z]

    bottom_east_z = upper_east_corner[2] - np.sin(fault_dip_rad) * fault_dip_width
    bottom_east_corner = [fault_strike_width / 2, bottom_y, bottom_east_z]

    corners = np.array([bottom_west_corner, upper_west_corner, upper_east_corner, bottom_east_corner])
    return tectosaur.make_rect(fault_n, fault_n, corners)



def make_surface():
    wx = 1000 * 1000
    wy = 1000 * 1000
    n = 81
    surf_corners = [[-wx, -wy, 0], [-wx, wy, 0], [wx, wy, 0], [wx, -wy, 0]]
    surf = tectosaur.make_rect(n, n, surf_corners)
    surf[0][:,2] = topo(surf[0][:,0], surf[0][:,1])
    return surf


def main():
    surf = make_surface()
    fault = make_fault()
    plot_model(surf[0][:,0], surf[0][:,1], surf[0][:,2], fault)

if __name__ == '__main__':
    main()
