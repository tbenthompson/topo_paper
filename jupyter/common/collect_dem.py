import os
import sys
import shutil
import gdal
import scipy.interpolate
import pyproj
import numpy as np


import download_dem

def get_dem(zoom, bounds, n_width, dest_dir = 'dem_download'):
    api_key = open('../common/mapzenapikey').read()
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    dest = os.path.join(dest_dir, 'raw_merc.tif')
    download_dem.download(dest, download_dem.tiles(zoom, *bounds), api_key, verbose = False)
    filebase, fileext = os.path.splitext(dest)
    dataset_merc = gdal.Open(dest)
    filename_latlon = os.path.join(dest_dir, 'latlon.tif')
    dataset_latlon = gdal.Warp(filename_latlon, dataset_merc, dstSRS = 'EPSG:4326')
    dem = dataset_latlon.ReadAsArray().astype(np.float64)
    width = dataset_latlon.RasterXSize
    height = dataset_latlon.RasterYSize
    gt = dataset_latlon.GetGeoTransform()
    xs = np.linspace(0, width - 1, width)
    ys = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(xs, ys)
    lon = gt[0] + X * gt[1] + Y * gt[2]
    lat = gt[3] + X * gt[4] + Y * gt[5]
    assert(gt[2] == 0)
    assert(gt[4] == 0)
    minlat, minlon = bounds[0], bounds[1]
    maxlat, maxlon = bounds[2], bounds[3]
    expand = 0
    LON, LAT = np.meshgrid(
        np.linspace(minlon - expand, maxlon + expand, n_width), 
        np.linspace(minlat - expand, maxlat + expand, n_width)
    )
    DEM = scipy.interpolate.griddata(
        (lon.flatten(), lat.flatten()), dem.flatten(),
        (LON, LAT)
    )
    return LON.flatten(), LAT.flatten(), DEM.flatten()

def project(lon, lat, dem, proj_name):
    wgs84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
    if proj_name == 'ellps':
        proj = pyproj.Proj('+proj=geocent +datum=WGS84 +units=m +no_defs')
    elif proj_name.startswith('utm'):
        zone = proj_name[3:]
        print(zone)
        proj = pyproj.Proj("+proj=utm +zone=" + zone + ", +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    x,y,z = pyproj.transform(wgs84, proj, lon, lat, dem)
    projected_pts = np.vstack((x,y,z)).T.copy()
    return projected_pts