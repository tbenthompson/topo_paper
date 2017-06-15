import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from connected_components import get_connected_components

pts, tris = np.load('data/utm_fault_mesh.npy')

cc = get_connected_components(tris)
planar_pts = []
planar_tris = []
for comp_idx in [0,1]:
    comp_pts = pts[tris[np.where(cc[1] == comp_idx)[0]]].reshape((-1, 3))
    upper_left_idx = np.argmin(comp_pts[:,1])
    bottom_left_idx = np.argmin(comp_pts[:,0])
    bottom_right_idx = np.argmax(comp_pts[:,1])
    upper_right_idx = np.argmax(comp_pts[:,0])


    #need 4 points to define a rectangle, the upper right, bottom right, upper left, bottom left
    corners = np.array([
        comp_pts[bottom_left_idx,:],
        comp_pts[upper_left_idx,:],
        comp_pts[upper_right_idx,:],
        comp_pts[bottom_right_idx,:]
    ])
    planar_pts.append(corners)
    comp_planar_tris = np.array([[0,1,2], [0,2,3]])
    planar_tris.append(4 * comp_idx + comp_planar_tris)

    plt.figure()
    plt.triplot(pts[:,0], pts[:,1], tris[np.where(cc[1] == comp_idx)[0],:])
    plt.triplot(corners[:,0], corners[:,1], comp_planar_tris)
    plt.show()

planar_pts = np.array(planar_pts).reshape((-1, 3))
planar_tris = np.array(planar_tris).reshape((-1, 3))
plt.figure()
plt.triplot(pts[:,0], pts[:,1], tris)
plt.triplot(planar_pts[:,0], planar_pts[:,1], planar_tris)
plt.show()

np.save('data/planar_utm_fault_mesh.npy', (planar_pts, planar_tris))


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(pts[:,0], pts[:,1], tris, pts[:,2])
# plt.show()
