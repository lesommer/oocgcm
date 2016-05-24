#!/usr/bin/env python
# coding=utf-8
#
""" Create a movie of a 2D variable
    requires: cartopy, ffmpeg
    to do:
        - automatically choose lon,lat bounds and ticks
        - fix loading of the library
        - improve treatment of datetime64 in titles
"""


# - Modules
#
import sys, os
import xarray as xr
import time, calendar
import matplotlib.pyplot as plt
from matplotlib import animation
import cartopy.crs as ccrs

sys.path.append("../../../oocgcm/")
from oocgcm.oceanmodels.nemo import grids

# - Parameter
natl60_path = '/home7/pharos/othr/NATL60/'
coordfile = natl60_path + 'NATL60-I/NATL60_coordinates_v4.nc'
maskfile = natl60_path + 'NATL60-I/NATL60_v4.1_cdf_byte_mask.nc'
#filenatl60 = natl60_path+'NATL60-MJM155-S/5d/2008/NATL60-MJM155_y2008m01*gridT.nc'
filenatl60 = natl60_path + 'NATL60-MJM155-S/5d/2008/NATL60-MJM155_y2008*gridT.nc'

chunks = (1727, 2711)
xr_chunks = {'x': chunks[-1], 'y': chunks[-2]}
xr_chunks_t = {'x': chunks[-1], 'y': chunks[-2], 'time_counter':1}

# - creating the grid object
grd = grids.nemo_2d_grid(nemo_coordinate_file=coordfile, nemo_byte_mask_file=maskfile, chunks=xr_chunks)

# - defining a 2D xarray
v_xr = xr.open_mfdataset(filenatl60, chunks=xr_chunks_t, lock=False)['sossheig'][:]
# http://xarray.pydata.org/en/stable/io.html
print v_xr

# - plot a 2D maps for warm up
#
plt.figure(figsize=(8,3))
ax=plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-90, 15, 25, 70],ccrs.PlateCarree())
v_xr[0, ...].plot.pcolormesh('nav_lon','nav_lat',ax=ax, 
                             vmin=-1.0, vmax=1.3, transform=ccrs.PlateCarree())
plt.title(v_xr[0, ...].coords['time_centered'].values, size=10) # to modify the title
#ax.title.set_size(10)
lon_tcks = range(-80,20,20)
lat_tcks = range(30,70,10)
ax.set_xticks(lon_tcks, crs=ccrs.PlateCarree())
ax.set_yticks(lat_tcks, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m') # Currently can be one of “110m”, “50m”, and “10m”
ax.gridlines()
plt.savefig('figs.jpg',dpi=300)

#sys.exit()


#-create the movie
#

# create repertory if need be
if not os.path.exists('./movies'): os.makedirs('./movies')
os.system("rm -rf movies/*.jpg >& /dev/null")

vmin, vmax=(-1.0,1.3)

# setup the plot
fig = plt.figure(figsize=(8,3))
ax=plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-90, 15, 25, 70],ccrs.PlateCarree())
plt_hdl = v_xr[0, ...].plot.pcolormesh('nav_lon','nav_lat',ax=ax, 
                                       vmin=-1.0, vmax=1.3, transform=ccrs.PlateCarree())
plt.title(v_xr[0, ...].coords['time_centered'].values, size=10) # to modify the title
#ax.title.set_size(10)
lon_tcks = range(-80,20,20)
lat_tcks = range(30,70,10)
ax.set_xticks(lon_tcks, crs=ccrs.PlateCarree())
ax.set_yticks(lat_tcks, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m') # Currently can be one of “110m”, “50m”, and “10m”
ax.gridlines()

for i in xrange(v_xr.shape[0]):
    plt_hdl.remove()
    plt_hdl = v_xr[i, ...].plot.pcolormesh('nav_lon','nav_lat',ax=ax,
                                                vmin=vmin, vmax=vmax,
                                                transform=ccrs.PlateCarree(),
                                                add_colorbar=False)
    plt.title(v_xr[i, ...].coords['time_centered'].values, size=10)
    plt.savefig('movies/fig%04d.jpg'%i, dpi=100)
    print i,'/',v_xr.shape[0]-1

os.system("ffmpeg -y -r 4 -i movies/fig%04d.jpg  movies/movie_cartopy_ffmpeg_ssh.mp4")



print "make_movie_cartopy_ffmpeg.py done"

