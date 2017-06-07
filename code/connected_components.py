import numpy as np
import scipy.sparse.csgraph as graph

def get_connected_components(tris):
    n_tris = tris.shape[0]
    touching = [[] for i in range(np.max(tris) + 1)]
    for i in range(n_tris):
        for d in range(3):
            touching[tris[i,d]].append(i)
    connectivity = np.zeros((n_tris, n_tris))
    for i in range(len(touching)):
        for row in touching[i]:
            for col in touching[i]:
                connectivity[row, col] = 1
    return graph.connected_components(connectivity)

