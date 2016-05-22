import os
import numpy as np

import xarray as xr
import xarray.ufuncs as xu

from . import TestCase, print_array_around

from oocgcm.core import grids
from oocgcm.griddeddata import grids as agrids
from oocgcm.oceanmodels.nemo import grids as fgrids
from oocgcm.parameters.mathematicalparameters import deg2rad

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
        self.assertArray2dCloseInside(grids._di(self.xrt).to_masked_array(),di)

    def test_dj(self):
        dj = np.roll(self.t,-1,axis=-2) - self.t
        self.assertArray2dCloseInside(grids._dj(self.xrt).to_masked_array(),dj)

    def test_mi(self):
        mi = (np.roll(self.t,-1,axis=-1) + self.t) / 2.
        self.assertArray2dCloseInside(grids._mi(self.xrt).to_masked_array(),mi)

    def test_mj(self):
        mj = (np.roll(self.t,-1,axis=-2) + self.t) / 2.
        self.assertArray2dCloseInside(grids._mj(self.xrt).to_masked_array(),mj)

class TestGrid2d_Array(TestCase):
    def setUp(self):
        self.x = np.arange(start=-180, stop=181, step=10,dtype=float)
        self.y = np.arange(start=-90, stop=91, step=10,dtype=float)
        self.xx,self.yy = np.meshgrid(self.x, self.y)
        self.t = np.sin(self.xx/30.) + np.cos(self.yy/30.)

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

class TestGrid2d_DifferentialOperators(TestCase):
    def setUp(self):
        self.x = np.arange(start=0, stop=90, step=1,dtype=float)
        self.y = np.arange(start=0, stop=45, step=1,dtype=float)
        lons,lats = np.meshgrid(self.x, self.y)
        self.grd = agrids.latlon_2d_grid(latitudes=lats,longitudes=lons)
        self.xt,self.yt = self.grd.get_projection_coordinates_in_meters()
        self.xu = self.grd.change_grid_location_t_to_u(self.xt)
        self.yu = self.grd.change_grid_location_t_to_u(self.yt)
        self.xv = self.grd.change_grid_location_t_to_v(self.xt)
        self.yv = self.grd.change_grid_location_t_to_v(self.yt)
        self.scale = 2. * 3.14159 / 5.e6 # value adapted to the grid step
        self.tvar = xu.sin(self.xt * self.scale) + xu.cos(self.yt * self.scale)


    def test_horizontal_gradient(self):
        tols = {'rtol':1e-3,'atol':1e-3}
        gradvar =  self.grd.horizontal_gradient(self.tvar)
        gradx   =   self.grd.horizontal_gradient(self.xt)
        dxdy = gradx.y_component
        s = self.scale
        actual_gx = gradvar.x_component.to_masked_array()
        actual_gy = gradvar.y_component.to_masked_array()
        expected_gx = (xu.cos(self.xu * s) * s).to_masked_array()
        expected_gy = (- xu.sin(self.yv * s) * s
                       + xu.cos(self.xv * s) * s * dxdy
                       ).to_masked_array()
        self.assertArray2dCloseInside(actual_gx / s,expected_gx / s ,
                                      depth=2,**tols)
        print dxdy[:,20].values
        #print self.grd.arrays['cell_y_size_at_t_location'][:,20].values
        #print self.grd.arrays['cell_y_size_at_v_location'][:,20].values
        #print_array_around(expected=expected_gy/s,actual=actual_gy/s)
        self.assertArray2dCloseInside(actual_gy / s,expected_gy /s ,
                                      depth=2, **tols)

    #def test_horizontal_laplacian(self):
    #    l =  self.grd.horizontal_laplacian(self.tvar)
    #    s = self.scale
    #    tols = {'rtol':1e-3,'atol':1e-3}
    #    actual_lap = l.to_masked_array()
    #    expected_lap = (-1. *  xu.sin(self.xt * s) * s * s).to_masked_array()
    #    expected_lap + (-1. *  xu.cos(self.xt * s) * s * s).to_masked_array()
    #    self.assertArray2dCloseInside(actual_lap/s**2,expected_lap/s**2,**tols)


class TestGrid2d_NEMO(TestCase):
    def setUp(self):
        self.coordfile = os.path.join(os.path.dirname(__file__), 'data', \
                                 'nemo-oocgcm-testdata_coordinates.nc')
        self.maskfile  = os.path.join(os.path.dirname(__file__), 'data', \
                                 'nemo-oocgcm-testdata_byte_mask.nc')
    def test_grid2d_from_netCDF(self):
        fgrd =  fgrids.nemo_2d_grid(nemo_coordinate_file=self.coordfile,\
                                    nemo_byte_mask_file=self.maskfile)
