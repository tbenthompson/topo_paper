import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import tectosaur

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

    # plt.figure(figsize = (10,10))
    # cntf = plt.contourf(X, Y, Z, levels = levels)
    # plt.contour(X, Y, Z, levels = levels)
    # plt.colorbar(cntf)
    # plt.triplot(surf[0][:,0], surf[0][:,1], surf[1], linewidth = 0.5, color = 'k', zorder = 1000)
    # # plt.savefig('subduction_model_surf_mesh.pdf')
    plt.show()

def main():
    fault = np.load('data/wenchuan/planar_utm_fault_mesh.npy')
    fault_refined,_ = tectosaur.refine_to_size(fault, 70000000)
    print(fault_refined[1].shape[0])
    surf = np.load('data/wenchuan/high_res/utm_surf_mesh.npy')
    plot_model(surf, fault_refined)

if __name__ == '__main__':
    main()
