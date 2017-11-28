import numpy as np
import scipy.interpolate
import collect_dem
def remove_unused_pts(m):
    referenced_pts = np.unique(m[1])
    new_pts = m[0][referenced_pts,:]
    new_indices = np.empty(m[0].shape[0], dtype = np.int64)
    new_indices[referenced_pts] = np.arange(referenced_pts.shape[0])
    new_tris = new_indices[m[1]]
    return (new_pts, new_tris)

def get_surf_fault_edges(surf_tris, fault_tris):
    surf_verts = np.unique(surf_tris)
    surf_fault_edges = []
    for i, t in enumerate(fault_tris):
        in_surf = []
        for d in range(3):
            if t[d] in surf_verts:
                in_surf.append((i, d))
        if len(in_surf) == 2:
            surf_fault_edges.append(in_surf)
    return surf_fault_edges

def get_surf_fault_pts(surf_tris, fault_tris):
    surf_fault_pts = []
    for e in get_surf_fault_edges(surf_tris, fault_tris):
        for j in range(2):
            t_idx, d = e[j]
            surf_fault_pts.append(fault_tris[1][t_idx, d])
    surf_fault_pts = np.unique(surf_fault_pts)
    return surf_fault_pts