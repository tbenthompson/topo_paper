from __future__ import division
import math
import code
import datetime
import urllib
import utm
import os.path
import numpy as np
import matplotlib.pyplot as plt
#import mpl_toolkits.basemap.pyproj as pyproj
import scipy as sp
from okada_wrapper import dc3d0wrapper, dc3dwrapper
from matplotlib import cm
#from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
# For buffering around fault shape
from shapely.ops import cascaded_union
from shapely.ops import unary_union
#from shapely.ops import MultiPolygon
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pdb as check
from descartes import PolygonPatch
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
from geopy.distance import great_circle
import Readsrcmod
import matplotlib
from matplotlib.pyplot import contourf, contour, xlabel, ylabel, title, colorbar, show, savefig, plot, close, xlim, ylim
import Calculate
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors


def plotFault(filename):
    
    """
    Plots the fault
    """
    boundsize = 5.
    veclength = 30
    
    EventSrcmod = Readsrcmod(filename, 'FSPFiles')
    
    for ii in xrange(0,360,20):
      print ii
      fig = plt.figure(facecolor='white', figsize=(17, 6), dpi=100)
      for i in range(0, 3):
        print i
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        verts = []
        x1 = np.min(np.double(EventSrcmod['x1Utm']))/1000.-boundsize
        x2 = np.max(np.double(EventSrcmod['x1Utm']))/1000.+boundsize
        y1 = np.min(np.double(EventSrcmod['y1Utm']))/1000.-boundsize
        y2 = np.max(np.double(EventSrcmod['y1Utm']))/1000.+boundsize
        z1 = np.min(np.double(EventSrcmod['z1Utm']))/1000.-boundsize
        z2 = np.max(np.double(EventSrcmod['z1Utm']))/1000.+boundsize
        
        if i == 0:
            cmap = plt.get_cmap('binary')
            stringslip = 'slip'
            title('total slip')
        if i > 0: cmap = plt.get_cmap('bwr')
        if i == 1:
            stringslip = 'slipStrike'
            title('strike slip')
        if i == 2:
            stringslip = 'slipDip'
            title('dip slip')
                #check.set_trace()
        maxslip = np.max(np.abs(np.array(EventSrcmod['slip'])))
        for iPatch in range(0, len(EventSrcmod['x1Utm'])): # Plot the edges of each fault patch fault patches
            xvec = [np.double(EventSrcmod['x1Utm'][iPatch])/1000., np.double(EventSrcmod['x2Utm'][iPatch])/1000., np.double(EventSrcmod['x4Utm'][iPatch])/1000, np.double(EventSrcmod['x3Utm'][iPatch])/1000]
            yvec = [np.double(EventSrcmod['y1Utm'][iPatch])/1000, np.double(EventSrcmod['y2Utm'][iPatch])/1000, np.double(EventSrcmod['y4Utm'][iPatch])/1000, np.double(EventSrcmod['y3Utm'][iPatch])/1000]
            zvec = [np.double(EventSrcmod['z1Utm'][iPatch])/1000, np.double(EventSrcmod['z2Utm'][iPatch])/1000, np.double(EventSrcmod['z4Utm'][iPatch])/1000, np.double(EventSrcmod['z3Utm'][iPatch])/1000]
            verts = [zip(xvec, yvec, zvec)]
            frac = getfrac(maxslip, np.double(EventSrcmod[stringslip][iPatch]))

            if i>0:
                frac = frac/2.0+0.5
                polyies = Poly3DCollection(verts, facecolors = cmap(frac), linewidths = 0.1)
            if i==0:
                  polyies = Poly3DCollection(verts, facecolors = cmap(frac), linewidths = 0.1)

            polyies.set_edgecolor('k')
            ax.add_collection3d(polyies)
    
        # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x2-x1, y2-y1, z2-z1]).max()
#            Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x1 + x2)
#            Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y1 + y2)
#            Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z1 + z2)
#            for xb, yb, zb in zip(Xb, Yb, Zb):
#                ax.plot([xb], [yb], [zb], 'w')

        norm = colors.Normalize(vmin=0, vmax=1.0)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, cmap=cmap, norm=norm, orientation='horizontal', ticks = [0, 0.5, 1.0])
            #cb1 = plt.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
            #cb1.set_label('$\mathrm{slip} \, \mathrm{(m)}$')
                
                #plt.clim([0.0, 1.0])
        cbar.ax.set_xlabel('$\mathrm{slip} \, \mathrm{(m)}$')
        if i == 0: cbar.ax.set_xticklabels(['0', str(0.5*maxslip), str(1.0*maxslip)])
        else: cbar.ax.set_xticklabels([str(-1.0*maxslip), '0', str(1.0*maxslip)])
        #cbar.ax.tick_params(length=0)
        ax.set_xlim3d([x1, x2])
        ax.set_ylim3d([y1, y2])
        ax.set_zlim3d([z1, z2])
        ax.invert_zaxis()
        fig.suptitle(str(filename) + ', red dot blue = %1.4f' % np.dot(ns, n) + ', strike = ' + str(EventSrcmod['strikeMean']) + ', dip = ' + str(EventSrcmod['dipMean']) + ', dip calculated = %1.1f' % math.degrees(np.arccos(np.dot([0,0,-1], n))) + ', rake = ' + str(EventSrcmod['rakeMean']))
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_zlabel('z (km)')
        ax.auto_scale_xyz([x1, x2], [y1, y2], [z1, z2])

        ax.view_init(elev=20., azim=ii)
        #ax.pbaspect = [1, 1, 0.25]
        #ax.auto_scale_xyz([minbound, maxbound], [minbound, maxbound], [minbound, maxbound])
#check.set_trace()

      plt.savefig('ReadSrcModFigs/%s_%d.png' % (directory, ii))
      plt.close()

    return