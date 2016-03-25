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

#- Timing tool
#
@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))

#- Parameter
coordfile  = '/Users/lesommer/data/NATL60/NATL60-I/NATL60_coordinates_v4.nc'
filenatl60 = '/Users/lesommer/data/NATL60/NATL60-MJM155-S/1d/2008/NATL60-MJM155_y2008m01.1d_BUOYANCYFLX.nc' 
#chunks = (1000,1000)
chunks = (3454,5422)
#chunks = (1727,2711)

#- Actual code
#
# with numpy : 

with timeit_context('Creation of grid object and loading a 2D t-file'):
    grd = mgd.nemo_grid_with_numpy(coordfile=coordfile)
    sig0 = Dataset(filenatl60).variables['vosigma0'][0]
    print('The grid shape is ' + str(grd.e1u.shape))
    print('The array shape is ' + str(sig0.shape))

with timeit_context('Computation of the horizontal gradient'):
    gradsig = grd.gradh(sig0)
    print('The output array shape is ' + str(gradsig[0].shape))


print('')

# with dask :  
with timeit_context('Creation of grid object and loading a 2D t-file'):
   
    grd = mgd.nemo_grid_with_dask(coordfile=coordfile,chunks=chunks)
    sig0 = da.from_array(np.array(Dataset(filenatl60).variables['vosigma0'][0]),chunks=chunks)
    #ds = da.open_dataset(filenatl60)
    #sig0 = ds.variables['vosigma0']
    print('The grid shape is ' + str(grd.e1u.shape))
    print('The array shape is ' + str(sig0.shape))


with timeit_context('Computation of the horizontal gradient'):
    gradsig = grd.gradh(sig0)
    gradsig = (gradsig[0].compute(),gradsig[1].compute())
    print('The output array shape is ' + str(gradsig[0].shape))
