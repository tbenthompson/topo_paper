import sys
import numpy as np
from mayavi.mlab import *

m = np.load(sys.argv[1])
triangular_mesh(m[0][:,0], m[0][:,1], m[0][:,2], m[1], representation = 'surface')
triangular_mesh(m[0][:,0], m[0][:,1], m[0][:,2], m[1], representation = 'wireframe', color = (0,0,0))
show()
