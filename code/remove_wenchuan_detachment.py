import sys


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
    res = graph.connected_components(connectivity)

filename = sys.argv[1]
pts, tris = np.load(filename)
res = get_connected_components(tris)
import ipdb; ipdb.set_trace()
