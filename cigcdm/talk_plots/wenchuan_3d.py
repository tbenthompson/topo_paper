import sys
import numpy as np
from mayavi.mlab import *

surf_filename = 'data/wenchuan/narrow/utm_surf_mesh.npy'
fault_filename = 'data/wenchuan/utm_fault_mesh.npy'
m_surf = np.load(surf_filename)
m_fault = np.load(fault_filename)

figure(bgcolor = (0,0,0), fgcolor = (1,1,1))
triangular_mesh(m_fault[0][:,0], m_fault[0][:,1], m_fault[0][:,2], m_fault[1], representation = 'surface', colormap = 'Blues')
# triangular_mesh(m_surf[0][:,0], m_surf[0][:,1], m_surf[0][:,2], m_surf[1], representation = 'wireframe', color = (0,0,0))
# triangular_mesh(m_surf[0][:,0], m_surf[0][:,1], 0 * m_surf[0][:,2], m_surf[1], representation = 'surface', colormap = 'gray', opacity = 0.2)
triangular_mesh(m_surf[0][:,0], m_surf[0][:,1], m_surf[0][:,2], m_surf[1], representation = 'wireframe', color = (1,1,1))
show()
