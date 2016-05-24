#!/usr/bin/env python
# coding=utf-8
#
""" Create a movie of a 2D variable
    requires: cartopy
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


# - Class file for animation
#
class animationGeneral(object):
# First set up the figure, the axis, and the plot element we want to animate
    def __init__(self, var_xarray, Nsteps, vmin, vmax, animationname):
        # Global definitions
        self.var = var_xarray
        self.Nsteps = Nsteps
        self.vmin = vmin
        self.vmax = vmax

        # Figure and axes
        self.fig = plt.figure(figsize=(8,3))
        self.ax=plt.axes(projection=ccrs.PlateCarree())
        self.ax.set_extent([-90, 15, 25, 70],ccrs.PlateCarree())
        self.plt_hdl = self.var[0, ...].plot.pcolormesh('nav_lon','nav_lat',ax=self.ax,
                                                    vmin=self.vmin, vmax=self.vmax,
                                                    transform=ccrs.PlateCarree())
        plt.title(self.var[0, ...].coords['time_centered'].values, size=10) # to modify the title
        #self.ax.title.set_size(10)
        lon_tcks = range(-80,20,20)
        lat_tcks = range(30,70,10)
        self.ax.set_xticks(lon_tcks, crs=ccrs.PlateCarree())
        self.ax.set_yticks(lat_tcks, crs=ccrs.PlateCarree())
        self.ax.coastlines(resolution='50m') # Currently can be one of “110m”, “50m”, and “10m”
        self.ax.gridlines()
        
        # Animation definitio
        anim = animation.FuncAnimation(self.fig, self.update, xrange(Nsteps), interval=1, blit=False)

        # Writer FFMpeg needed for mp4 film
        mywriter = animation.FFMpegWriter(bitrate=500)
        anim.save(animationname, writer=mywriter, fps=1, extra_args=['-vcodec', 'libx264'], dpi=300)
        #anim.save(animationname, writer=mywriter, fps=1, extra_args=['-vcodec', 'libx264'])

    # No init_plot needed
    def setup_plot(self):
        pass
        
    # animation function to be called nsteps times plus the initialisation since init_func isn't defined
    def update(self, i):
        print i+1, '/', self.Nsteps
        
        self.plt_hdl.remove()
        self.plt_hdl = self.var[i, ...].plot.pcolormesh('nav_lon','nav_lat',ax=self.ax,
                                                        vmin=self.vmin, vmax=self.vmax,
                                                        transform=ccrs.PlateCarree(),
                                                        add_colorbar=False)
        plt.title(self.var[i, ...].coords['time_centered'].values, size=10) # to modify the title
        #self.ax.title.set_size(10)        

    def show(self):
        plt.show()


# - Actually
#
if not os.path.exists('./movies'): os.makedirs('./movies')
a = animationGeneral(v_xr, v_xr.shape[0], -1, 1.3, 'movies/movie_crt_ssh.mp4')

print "make_movie_cartopy.py done"

