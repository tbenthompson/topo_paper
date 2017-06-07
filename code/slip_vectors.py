import numpy as np
from tectosaur.geometry import tri_normal

def get_slip_vectors(tri):
    n = tri_normal(tri, normalize = True)
    is_normal_e0 = np.abs(n[0]) >= 1.0
    if not is_normal_e0:
        start_vector = [1, 0, 0]
    else:
        start_vector = [0, 1, 0]
    v1 = np.cross(n, start_vector)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    v2 /= np.linalg.norm(v2)
    return v1, v2

def test_slip_vec_easy():
    v1, v2 = get_slip_vectors(np.array([[0,0,0],[1,0,0],[0,1,0]]))

def test_slip_vec_hard():
    v1, v2 = get_slip_vectors(np.array([[0,0,0],[0,1,0],[0,0,1]]))
    np.testing.assert_almost_equal(v1, [0,0,1])
    np.testing.assert_almost_equal(v2, [0,-1,0])

def test_slip_vec_harder():
    for i in range(10):
        # random triangles should still follow these rules:
        # vecs should be perpindicular to each other and the normal
        # and be normalized to unit length
        tri = np.random.rand(3,3)
        n = tri_normal(tri, normalize = True)
        v1, v2 = get_slip_vectors(tri)
        np.testing.assert_almost_equal(np.linalg.norm(v1), 1.0)
        np.testing.assert_almost_equal(np.linalg.norm(v2), 1.0)
        np.testing.assert_almost_equal(v1.dot(v2), 0.0)
        np.testing.assert_almost_equal(v1.dot(n), 0.0)
        np.testing.assert_almost_equal(v2.dot(n), 0.0)
