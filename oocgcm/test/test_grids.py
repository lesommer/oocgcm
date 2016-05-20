import os
import numpy as np

import xarray as xr
import xarray.ufuncs as xu

from . import TestCase

from oocgcm.core import grids
from oocgcm.griddeddata import grids as agrids
from oocgcm.oceanmodels.nemo import grids as fgrids

def path_to_test_coord():
    return open_dataset(os.path.join(os.path.dirname(__file__), 'data', name),
                        *args, **kwargs)

class TestCoreDifferenceOperators(TestCase):
    def setUp(self):
        self.x = np.arange(start=0, stop=101, step=10,dtype=float)
        self.y = np.arange(start=0, stop=101, step=10,dtype=float)
        self.xx,self.yy = np.meshgrid(self.x, self.y)
        self.xrx = xr.DataArray(self.xx,dims=['y','x'])
        self.xry = xr.DataArray(self.yy,dims=['y','x'])
        self.xrt = xu.sin(self.xrx/30.) + xu.cos(self.xry/30.)
        self.t = np.sin(self.xx/30.) + np.cos(self.yy/30.)

    def test_di(self):
        di = np.roll(self.t,-1,axis=-1) - self.t
        self.assertArrayClose(grids._di(self.xrt).to_masked_array(),di)

    def test_dj(self):
        dj = np.roll(self.t,-1,axis=-2) - self.t
        self.assertArrayClose(grids._dj(self.xrt).to_masked_array(),dj)

    def test_mi(self):
        mi = (np.roll(self.t,-1,axis=-1) + self.t) / 2.
        self.assertArrayClose(grids._mi(self.xrt).to_masked_array(),mi)

    def test_mj(self):
        mj = (np.roll(self.t,-1,axis=-2) + self.t) / 2.
        self.assertArrayClose(grids._mj(self.xrt).to_masked_array(),mj)

class TestGrid2d_Array(TestCase):
    def setUp(self):
        self.x = np.arange(start=-180, stop=181, step=10,dtype=float)
        self.y = np.arange(start=-90, stop=91, step=10,dtype=float)
        self.xx,self.yy = np.meshgrid(self.x, self.y)

    def test_grid2d_from_array1d(self):
        grd = agrids.latlon_2d_grid(latitudes=self.y,longitudes=self.x)
        xx = grd.arrays['projection_x_coordinate_at_t_location']
        yy = grd.arrays['projection_y_coordinate_at_t_location']
        self.assertArrayEqual(xx.to_masked_array(),self.xx)
        self.assertArrayEqual(yy.to_masked_array(),self.yy)

    def test_grid2d_from_array2d(self):
        grd = agrids.latlon_2d_grid(latitudes=self.yy,longitudes=self.xx)
        xx = grd.arrays['projection_x_coordinate_at_t_location']
        yy = grd.arrays['projection_y_coordinate_at_t_location']
        self.assertArrayEqual(xx.to_masked_array(),self.xx)
        self.assertArrayEqual(yy.to_masked_array(),self.yy)
        self.assertArrayEqual(xx.to_masked_array()[0,:],self.x)
        self.assertArrayEqual(yy.to_masked_array()[:,0],self.y)

class TestGrid2d_NEMO(TestCase):
    def setUp(self):
        self.coordfile = os.path.join(os.path.dirname(__file__), 'data', \
                                 'nemo-oocgcm-testdata_coordinates.nc')
        self.maskfile  = os.path.join(os.path.dirname(__file__), 'data', \
                                 'nemo-oocgcm-testdata_byte_mask.nc')
    def test_grid2d_from_netCDF(self):
        fgrd =  fgrids.nemo_2d_grid(nemo_coordinate_file=self.coordfile,\
                                    nemo_byte_mask_file=self.maskfile)
