#!/usr/bin/env python
#
"""testing_grids_with_dask script.
Test the performance of modelgrids.nemo_grid objects.
"""
# from http://stackoverflow.com/questions/5478351/python-time-measure-function


#- Modules
#
import time
import modelgrids as mgd
from contextlib import contextmanager
from netCDF4 import Dataset
import dask.array as da
import numpy as np
import xarray as xr

#- Timing tool
#
@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('{} takes {} ms'.format(name, int(elapsedTime * 1000)))

#- Parameter
coordfile  = '/Users/lesommer/data/NATL60/NATL60-I/NATL60_coordinates_v4.nc'
filenatl60 = '/Users/lesommer/data/NATL60/NATL60-MJM155-S/1d/2008/NATL60-MJM155_y2008m01.1d_BUOYANCYFLX.nc'

chunks = (3454,5422)
#chunks = (1727,2711)
#chunks = (500,500)
xr_chunks = {'x': chunks[-1], 'y': chunks[-2]}

with_numpy     = True
with_dask_np   = False 
with_dask_nc   = False
with_xarray    = True

#- Actual code
#
# with numpy :
#

if with_numpy is True:
    print('\n')
    print('with numpy : ')

    with timeit_context('The creation of grid object and loading a 2D t-file'):
        grd = mgd.nemo_grid_with_numpy_arrays(coordfile=coordfile)
        sig0 = Dataset(filenatl60).variables['vosigma0'][0]
	    #sig0 = Dataset(filenatl60).variables['vosigma0'] # test
        print('The grid shape is ' + str(grd.e1u.shape))
        print('The array shape is ' + str(sig0.shape))

    with timeit_context('The computation of the horizontal gradient'):
        np_gradsig = grd.gradh(sig0)
        print('The output array shape is ' + str(np_gradsig[0].shape))


#
# with dask :
#

if with_dask_np is True:
    print('\n')
    print('with dask (from array): ')

    with timeit_context('The creation of grid object and loading a 2D t-file'):
        grd = mgd.nemo_grid_with_dask_arrays(coordfile=coordfile,chunks=chunks,array_type='dask_from_numpy')
        sig0 = da.from_array(np.array(Dataset(filenatl60).variables['vosigma0'][0]),chunks=chunks)
	#sig0 = da.from_array(np.array(Dataset(filenatl60).variables['vosigma0']),chunks=chunks) # test
        #ds = da.open_dataset(filenatl60)
        #sig0 = ds.variables['vosigma0']
        print('The grid shape is ' + str(grd.e1u.shape))
        print('The array shape is ' + str(sig0.shape))

    with timeit_context('The computation of the horizontal gradient'):
        gradsig = grd.gradh(sig0)
        danp_gradsig = (gradsig[0].compute(),gradsig[1].compute())
        print('The output array shape is ' + str(danp_gradsig[0].shape))

if with_dask_nc is True:
    print('\n')
    print('with dask (from netcdf): ')

    with timeit_context('The creation of grid object and loading a 2D t-file'):
        grd = mgd.nemo_grid_with_dask_arrays(coordfile=coordfile,chunks=chunks,
                                             array_type='dask_from_netcdf')
        #sig0 = da.from_array(np.array(Dataset(filenatl60).variables
        #                              ['vosigma0'][0]),chunks=chunks)
	ds = Dataset(filenatl60).variables['vosigma0']
        sig0 = da.from_array(ds, chunks=(1,)+ chunks)[0]
        print('The grid shape is ' + str(grd.e1u.shape))
        print('The array shape is ' + str(sig0.shape))

    with timeit_context('The computation of the horizontal gradient'):
        gradsig = grd.gradh(sig0)
        danc_gradsig = (gradsig[0].compute(),gradsig[1].compute())
        print('The output array shape is ' + str(danc_gradsig[0].shape))

#
# with xarray :
#

if with_xarray is True:
    print('\n')
    print('with xarray : ')

    with timeit_context('The creation of grid object and loading a 2D t-file'):
        grd = mgd.nemo_grid_with_xarray(coordfile=coordfile,chunks=xr_chunks)
        ds = xr.open_dataset(filenatl60,chunks=xr_chunks,lock=False)
        sig0 = ds.variables['vosigma0'][0]
        print('The grid shape is ' + str(grd.e1u.shape))
        print('The array shape is ' + str(sig0.shape))

    with timeit_context('The computation of the horizontal gradient'):
        gradsig = grd.gradh(sig0)
        #xr_gradsig = (gradsig[0].to_masked_array(),gradsig[1].to_masked_array())
        xr_gradsig = (gradsig[0].values,gradsig[1].values)
        #xr_gradsig = (gradsig[0],gradsig[1])
        print('The output array shape is ' + str(xr_gradsig[0].shape))
