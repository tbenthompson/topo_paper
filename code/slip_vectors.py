import numpy as np
from tectosaur.geometry import tri_normal
    #TODO: Here, I could find the two fault parallel vector directions by:
    # -- dot product [1, 0, 0] and [0, 1, 0] with the normal to check that they aren't the same vector
    # -- cross [1,0,0] with normal to find the first surface parallel direction
    # -- cross surface parallel direction 1 with normal to get surface parallel direction 2.
    # -- normalize everything.

def get_slip_vectors(tri):
    n = tri_normal(tri, normalize = True)
    is_normal_e0 = np.abs(n[0]) >= 1.0
    print(is_normal_e0)
    if not is_normal_e0:
        start_vector = [1, 0, 0]
    else:
        start_vector = [0, 1, 0]
    print(start_vector,n)
    v1 = np.cross(start_vector, n)
    v2 = np.cross(n, v1)
    return v1, v2

def test_slip_vec_easy():
    v1, v2 = get_slip_vectors(np.array([[0,0,0],[1,0,0],[0,1,0]]))
    print(v1, v2)
    assert(np.all(v1 == [1,0,0]))
    assert(np.all(v2 == [0,1,0]))

def test_slip_vec_hard():
    v1, v2 = get_slip_vectors(np.array([[0,0,0],[0,1,0],[0,0,1]]))
    print(v1, v2)
