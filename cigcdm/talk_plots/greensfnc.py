import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import matplotlib.cm as cm
import tectosaur.elastic as elastic
from sympy.utilities.lambdify import lambdify

matplotlib.rcParams.update({'font.size': 20})

K = elastic.T(0,1)['expr']
K = lambdify(elastic.all_args, K)


G = 1.0
nu = 0.25

obs = [0,0,0]
l = [0, 1, 0]

x = y = np.linspace(-3, 3, 500)
X, Y = np.meshgrid(x, y)
Xf = X.flatten()
Yf = Y.flatten()

out = np.empty_like(Xf)
for i in range(Xf.shape[0]):
    src = [Xf[i], Yf[i], 0]
    out[i] = K(G, nu, 0, 0, src[0], src[1], src[2], obs[0], obs[1], obs[2], l[0], l[1], l[2], 0, 0, 0)
out = out.reshape(X.shape)

levels = np.linspace(-5, 2, 9)
out = np.log10(np.abs(out))
cmap = cm.get_cmap(name='PuOr_r')

def go(include_tris):

    plt.figure(figsize = (12,10))
    ax = plt.gca()
    ax.set_aspect('equal')
    cntf = plt.contourf(X, Y, out, levels = levels, extend = 'both', cmap = cmap)
    plt.contour(X, Y, out, levels = levels, extend = 'both', linestyles = 'solid', colors = ['k'], linewidth = 1.5)

    tri1_verts = [(-2.3, -2), (-1.2, -2), (-2.1, -0.9)]
    tri2_verts = [(-0.3, -0.4), (0.9, -0.5), (0.1, 1.3)]
    tri1 = patches.Polygon(tri1_verts,linewidth=6,edgecolor='k',facecolor='none',zorder=1000)
    tri2 = patches.Polygon(tri2_verts,linewidth=6,edgecolor='k',facecolor='none',zorder=1000)

    plt.xlabel('$x$ (km)')
    plt.ylabel('$y$ (km)')

    cbar = plt.colorbar(cntf)
    cbar.set_label('$\log_{10}(|T^*_{xy}|)$')

    if include_tris:
        filename = 'greensfnc_tris.pdf'
        plt.gca().add_patch(tri1)
        plt.gca().add_patch(tri2)
    else:
        filename = 'greensfnc.pdf'

    plt.savefig(filename, bbox_inches = 'tight')
    plt.show()

go(False)
go(True)
