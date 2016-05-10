#!/usr/bin/env python
# coding=utf-8
#
""" Create a movie of a 2D variable 
    Requires: ffmpeg
"""


#- Modules
#
import sys, os
import xarray as xr
import time, calendar
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, pyproj

sys.path.append("../../../oocgcm/")
from oocgcm.oceanmodels.nemo import grids

#- Parameter
natl60_path='/home7/pharos/othr/NATL60/'
coordfile  = natl60_path+'NATL60-I/NATL60_coordinates_v4.nc'
maskfile = natl60_path+'NATL60-I/NATL60_v4.1_cdf_byte_mask.nc'
#filenatl60 = natl60_path+'NATL60-MJM155-S/5d/2008/NATL60-MJM155_y2008m01*gridT.nc'
filenatl60 = natl60_path+'NATL60-MJM155-S/5d/2008/NATL60-MJM155_y2008*gridT.nc'

chunks = (1727,2711)
xr_chunks = {'x': chunks[-1], 'y': chunks[-2]}
xr_chunks_t = {'x': chunks[-1], 'y': chunks[-2],'time_counter':1}

#- creating the grid object
grd = grids.nemo_2d_grid(nemo_coordinate_file=coordfile,nemo_byte_mask_file=maskfile,chunks=xr_chunks)
print "Grid has been loaded"

#- defining a 2D xarray
v_xr = xr.open_mfdataset(filenatl60,chunks=xr_chunks_t,lock=False)['sossheig'][:]
# http://xarray.pydata.org/en/stable/io.html
print v_xr.shape

# print figure
if not os.path.exists('./movies'): os.makedirs('./movies')
os.system("rm -rf movies/*.jpg >& /dev/null")
    
#- plot the horizontal gradient
plt.figure(figsize=(15,10))

for i in xrange(v_xr.shape[0]):
    plt.clf()
    v_xr[i,...].plot.pcolormesh(vmin=-1.,vmax=1.3)
    plt.savefig('movies/fig%04d.jpg'%i, dpi=100)
    print i,'/',v_xr.shape[0]-1

os.system("ffmpeg -y -r 4 -i movies/fig%04d.jpg  movies/movie_ssh.mp4")



print "make_movie.py done"





sys.exit()



#############################################################################"
### old code


import time, calendar
import modelgrids as mgd
from contextlib import contextmanager
from netCDF4 import Dataset, MFDataset
from mfdataset2 import MFDataset2
import dask.array as da
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from glob import glob

from mpl_toolkits.basemap import Basemap, pyproj

#- Timing tool
#
@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('{} takes {} ms'.format(name, int(elapsedTime * 1000)))

#
# directories and variable name
#
dir_natl60='/home7/pharos/othr/NATL60/'
#files_natl60=dir_natl60+'NATL60-MJM155-S/5d/2008/NATL60-MJM155_y2008m01*gridT.nc' # for faster result
files_natl60=dir_natl60+'NATL60-MJM155-S/5d/2008/NATL60-MJM155_y2008*gridT.nc'
coordfile=dir_natl60+'NATL60-I/NATL60_coordinates_v4.nc'
vname='sossheig'

# with dask from netcdf
chunks = (3454,5422)
grd = mgd.nemo_grid_with_dask_arrays(coordfile=coordfile,chunks=chunks,
                                     array_type='dask_from_netcdf')
print('The grid shape is ' + str(grd.e1u.shape))

#ds = MFDataset(files_natl60).variables[vname] # does not work
v = MFDataset2(files_natl60).variables[vname]
#print v
vd = da.from_array(v, chunks=(1,)+ chunks)
#print vd
#print('The array shape is ' + str(vd.shape))

# load time
ti = MFDataset2(files_natl60).variables['time_centered'][:]

# with basemap
def plot_map():
    lon=grd.lon; lat=grd.lat
    lat_ts=((lat[0,0]+lat[-1,0])*0.5)
    cbox=[lon[0, 0],lat[0, 0],lon[-1, -1],lat[-1, -1]]
    map = Basemap(projection='merc', resolution='l',
                    lat_ts=lat_ts,
                    llcrnrlon=cbox[0], llcrnrlat=cbox[1],
                    urcrnrlon=cbox[2], urcrnrlat=cbox[3])
    map.drawmapboundary(fill_color='white')
    map.drawcoastlines()
    map.fillcontinents(color='#cc9966', lake_color='#99ffff')
    lon_tcks = np.arange(-100.,30.,20.)
    lat_tcks = np.arange(0.,70.,10.)
    map.drawmeridians(lon_tcks, labels=[True, False, False, True])
    map.drawparallels(lat_tcks, labels=[False, True, True, False])
    # get projected coordinates
    x,y=map(lon,lat)
    return map, x, y

def print_time(ti):
    t0 = calendar.timegm(time.strptime("1 Jan 1958", "%d %b %Y"))
    tstruct=time.gmtime(ti+t0)
    #ti_str=time.strftime('%Y/%m/%d',tstruct)
    ti_str=time.strftime('%Y/%m/%d/%Hh',tstruct)
    #ti_str=time.strftime('%d',tstruct)+str(tstruct[3]/24.)[1:]+time.strftime('/%m/%Y',tstruct)

    return ti_str




# make movie of ssh

flag_ssh_movie=False

if flag_ssh_movie:
    
    if not os.path.exists('./mv_figs'): os.makedirs('./mv_figs')
    os.system("rm -rf mv_figs/* >& /dev/null")
    if not os.path.exists('./movies'): os.makedirs('./movies')
    
    #for i in range(len(ti[:5])):
    for i in range(len(ti)):
    
        plt.close('all')
        fig = plt.figure(figsize=(8,5))
        map, x, y = plot_map()
        
        # load snapshot of ssh
        toplt=vd[i,:,:].compute()
        lvls=20 
        lvls=np.arange(-1.,1.3,.1)
        cs=map.contourf(x,y,toplt,lvls,cmap=plt.cm.RdBu_r,animated=True)
        # add colorbar
        cb = map.colorbar(cs,"bottom", size="5%", pad="8%")
        cb.set_label('ssh [m], '+print_time(ti[i]))
    
        plt.savefig('mv_figs/fig%04d.jpg'%i, dpi=300)
        
    os.system("ffmpeg -y -r 2 -i mv_figs/fig%04d.jpg  movies/ssh.mp4")








