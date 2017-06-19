import os
import sys
import gdal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.interpolate

filename = sys.argv[1]
filebase, fileext = os.path.splitext(filename)
ds = gdal.Open(filename)
dem = ds.ReadAsArray()

msk = dem==-9999 # boolean array with True at elements to be masked
dem = np.ma.array(data=dem, mask=msk, fill_value=np.nan, dtype = np.float64)
# plt.imshow(dem)
# plt.show()

width = ds.RasterXSize
height = ds.RasterYSize
print(width,height)
gt = ds.GetGeoTransform()
# minlon = gt[0]
# maxlon = gt[0] + width * gt[1] + height * gt[2]
# minlat = gt[3] + width * gt[4] + height * gt[5]
# maxlat = gt[3]
xs = np.linspace(0, width - 1, width)
ys = np.linspace(0, height - 1, height)
X, Y = np.meshgrid(xs, ys)
lon = gt[0] + X * gt[1] + Y * gt[2]
lat = gt[3] + X * gt[4] + Y * gt[5]

assert(gt[2] == 0)
assert(gt[4] == 0)

minlat, minlon = 27.5, 100
maxlat, maxlon = 36.5, 108
n = int(sys.argv[2])
expand = 0
lons = np.linspace(minlon - expand, maxlon + expand, n)
lats = np.linspace(minlat - expand, maxlat + expand, n)
LON, LAT = np.meshgrid(lons, lats)

DEM = scipy.interpolate.griddata(
    (lon.flatten(), lat.flatten()), dem.flatten(),
    (LON, LAT)
)

np.save('data/lonlatdem.npy', [LON, LAT, DEM])
