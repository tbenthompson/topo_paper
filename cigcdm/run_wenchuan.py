import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import tectosaur
from tectosaur.geometry import projection, tri_normal
from cigcdm.slip_vectors import get_slip_vectors
from cigcdm.solve import solve_bem

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
    plt.triplot(surf[0][:,0], surf[0][:,1], surf[1], linewidth = 0.5, color = 'k', zorder = 1000)
    plt.triplot(fault[0][:,0], fault[0][:,1], fault[1], linewidth = 0.5, color = 'k', zorder = 1000)
    # plt.savefig('subduction_model.pdf')

    plt.figure()
    plt.triplot(surf[0][:,0], surf[0][:,2], surf[1])
    plt.plot(fault[0][:,0], fault[0][:,2], 'o')
    # plt.figure(figsize = (10,10))
    # cntf = plt.contourf(X, Y, Z, levels = levels)
    # plt.contour(X, Y, Z, levels = levels)
    # plt.colorbar(cntf)
    # plt.triplot(surf[0][:,0], surf[0][:,1], surf[1], linewidth = 0.5, color = 'k', zorder = 1000)
    # # plt.savefig('subduction_model_surf_mesh.pdf')
    plt.show()

def plot_3d_model(surf, fault):
    from mayavi import mlab
    mlab.triangular_mesh(surf[0][:,0], surf[0][:,1], surf[0][:,2], surf[1], representation = 'wireframe')
    mlab.triangular_mesh(fault[0][:,0], fault[0][:,1], fault[0][:,2], fault[1], representation = 'wireframe')
    mlab.show()

def get_fault_slip(fault):
    n_tris = fault[1].shape[0]
    fault_slip = np.empty((n_tris, 3, 3))
    vertical = np.array([0,0,1])
    for i in range(n_tris):
        tri = fault[0][fault[1][i,:]]
        #TODO: THIS IS WRONG!
        s = projection(vertical, tri_normal(tri, normalize = True))
        # v1, v2 = get_slip_vectors(tri)
        fault_slip[i] = s
    return fault_slip


def main():
    for flat in [False, True]:
        fault = np.load('data/wenchuan/planar_utm_fault_mesh.npy')
        fault[0][:,2] = np.where(fault[0][:,2] > 0, fault[0][:,2] - 5000, fault[0][:,2])
        import ipdb; ipdb.set_trace()
        print(np.max(fault[0][:,2]))
        slip = get_fault_slip(fault)
        fault_refined, refined_slip = tectosaur.refine_to_size(
            fault, 70000000,
            [slip[:,:,0], slip[:,:,1], slip[:,:,2]]
        )
        full_slip = np.concatenate([s[:,:,np.newaxis] for s in refined_slip], 2)
        # print(fault_refined[1].shape[0])
        surf = np.load('data/wenchuan/narrow/utm_surf_mesh.npy')
        if flat:
            surf[0][:,2] = 0
        # plot_3d_model(surf, fault)
        # plot_model(surf, fault_refined)
        surf_pts, surf_disp = solve_bem(surf, fault_refined, full_slip.flatten(), True)
        filename = 'data/wenchuan_disp.npy'
        if flat:
            filename = 'data/wenchuan_disp_flat.npy'
        np.save(filename, (surf_pts, surf_disp))

if __name__ == '__main__':
    main()
