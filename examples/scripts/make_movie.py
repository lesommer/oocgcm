#!/usr/bin/env python
# coding=utf-8
#
""" Create a movie of a 2D variable
"""


# - Modules
#
import sys, os
import xarray as xr
import time, calendar
import matplotlib.pyplot as plt
from matplotlib import animation

sys.path.append("../../../oocgcm/")
from oocgcm.oceanmodels.nemo import grids

# - Parameter
natl60_path = '/home7/pharos/othr/NATL60/'
coordfile = natl60_path + 'NATL60-I/NATL60_coordinates_v4.nc'
maskfile = natl60_path + 'NATL60-I/NATL60_v4.1_cdf_byte_mask.nc'
filenatl60 = natl60_path+'NATL60-MJM155-S/5d/2008/NATL60-MJM155_y2008m01*gridT.nc'
#filenatl60 = natl60_path + 'NATL60-MJM155-S/5d/2008/NATL60-MJM155_y2008*gridT.nc'

chunks = (1727, 2711)
xr_chunks = {'x': chunks[-1], 'y': chunks[-2]}
xr_chunks_t = {'x': chunks[-1], 'y': chunks[-2], 'time_counter':1}

# - creating the grid object
grd = grids.nemo_2d_grid(nemo_coordinate_file=coordfile, nemo_byte_mask_file=maskfile, chunks=xr_chunks)

# - defining a 2D xarray
v_xr = xr.open_mfdataset(filenatl60, chunks=xr_chunks_t, lock=False)['sossheig'][:]
# http://xarray.pydata.org/en/stable/io.html
print v_xr.shape


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
        self.fig = plt.figure()
        # self.ax = self.fig.axes()
        
        # Animation definitio
        anim = animation.FuncAnimation(self.fig, self.update, xrange(Nsteps), interval=1, blit=False)

        # Writer FFMpeg needed for mp4 film
        mywriter = animation.FFMpegWriter(bitrate=500)
        anim.save(animationname, writer=mywriter, fps=1, extra_args=['-vcodec', 'libx264'], dpi=300)

    # No init_plot needed
    def setup_plot(self):
        pass
        
    # animation function to be called nsteps times plus the initialisation since init_func isn't defined
    def update(self, i):
        print i, '/', self.Nsteps

        self.fig.clf()
        self.var[i, ...].plot.pcolormesh(vmin=self.vmin, vmax=self.vmax)
        plt.title(var[i, ...].coords['time_centered'].values, size=10) # to modify the title

    def show(self):
        plt.show()


# - Actually
#
if not os.path.exists('./movies'): os.makedirs('./movies')
a = animationGeneral(v_xr, v_xr.shape[0], -1, 1.3, 'movies/movie_ssh.mp4')

print "make_movie.py done"

