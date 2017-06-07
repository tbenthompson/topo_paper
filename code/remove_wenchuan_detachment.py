import sys
import numpy as np
from tectosaur.mesh import remove_duplicate_pts
from connected_components import get_connected_components

filename = sys.argv[1]
lon,lat,z,tris = [np.array(arr) for arr in np.load(filename, encoding = 'latin1')]
pts = np.vstack((lon,lat,z)).T
pts, tris = remove_duplicate_pts((pts, tris))
which_comp = get_connected_components(tris)[1]
# get average depth of each component
average_depth = np.zeros(3)
average_depth = np.mean(pts[tris,2], axis = 1)

component_depth = np.array([np.mean(average_depth[which_comp == d]) for d in range(3)])
detachment_component = np.where(component_depth < -21000)[0][0]
nondetachment_tris = tris[np.where(which_comp != detachment_component)[0]]

#remove unused pts
referenced_pts = np.unique(nondetachment_tris)
new_pts = pts[referenced_pts,:]
new_indices = np.empty(pts.shape[0], dtype = np.int64)
new_indices[referenced_pts] = np.arange(referenced_pts.shape[0])
new_tris = new_indices[nondetachment_tris]

np.save('data/without_detachment.npy', (new_pts, new_tris))
