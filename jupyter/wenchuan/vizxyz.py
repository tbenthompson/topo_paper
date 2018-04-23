import sys
import numpy as np
from mayavi import mlab

points, tris = np.load(sys.argv[1])
dist_center = np.sqrt(np.sum(points ** 2, axis = 1))

# min_elev = np.min(dist_center)
# max_elev = np.max(dist_center)
# print(min_elev, max_elev, max_elev - min_elev)
# C = (dist_center - min_elev) / (max_elev - min_elev)
# # vertical exaggerate
# up_vector = points / np.linalg.norm(points, axis = 1)[:,np.newaxis]
# elev = dist_center - min_elev
# points += 4 * elev[:,np.newaxis] * up_vector
#
#
#
# mlab.triangular_mesh(
#     points[:,0], points[:,1], points[:,2], tris,
#     # scalars = dist_center,
#     scalars = C,
#     representation = 'wireframe'
# )
# mlab.show()

start_fault = 372864
surf_tris = tris[:start_fault]
fault_tris = tris[start_fault:]

min_elev = np.min(dist_center[surf_tris])
max_elev = np.max(dist_center[surf_tris])
print(min_elev, max_elev, max_elev - min_elev)
C = np.clip((dist_center - min_elev) / (max_elev - min_elev), 0, 1)

mlab.triangular_mesh(
    points[:,0], points[:,1], points[:,2], surf_tris,
    # scalars = dist_center,
    scalars = C,
    representation = 'wireframe'
)

mlab.triangular_mesh(
    points[:,0], points[:,1], points[:,2], fault_tris,
    # scalars = dist_center,
    # scalars = C,
    color = (1.0,1.0,1.0),
    representation = 'surface'
)
mlab.show()
