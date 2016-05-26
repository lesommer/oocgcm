import os
import numpy as np

import xarray as xr
import xarray.ufuncs as xu

from . import TestCase, print_array_around, assert_equal

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


class TestCreationGrid2d_from_arrays(TestCase):
    def setUp(self):
        self.x = np.arange(start=-180, stop=181, step=30,dtype=float)
        self.y = np.arange(start=-90, stop=91, step=10,dtype=float)
        self.xx,self.yy = np.meshgrid(self.x, self.y)
        self.t = np.sin(self.xx/30.) + np.cos(self.yy/30.)

    def test_grid2d_from_latlon_array1d(self):
        grd = agrids.latlon_2d_grid(latitudes=self.y,longitudes=self.x)
        xx = grd.arrays['longitude_at_t_location']
        yy = grd.arrays['latitude_at_t_location']
        self.assertArrayEqual(xx.to_masked_array(),self.xx)
        self.assertArrayEqual(yy.to_masked_array(),self.yy)

    def test_grid2d_from_latlon_array2d(self):
        grd = agrids.latlon_2d_grid(latitudes=self.yy,longitudes=self.xx)
        xx = grd.arrays['longitude_at_t_location']
        yy = grd.arrays['latitude_at_t_location']
        self.assertArrayEqual(xx.to_masked_array(),self.xx)
        self.assertArrayEqual(yy.to_masked_array(),self.yy)
        self.assertArrayEqual(xx.to_masked_array()[0,:],self.x)
        self.assertArrayEqual(yy.to_masked_array()[:,0],self.y)

    def test_grid2d_from_plane_coordinate_array1d(self):
        grd = agrids.plane_2d_grid(xcoord=self.x,ycoord=self.y)
        xx = grd.arrays['plane_x_coordinate_at_t_location']
        yy = grd.arrays['plane_y_coordinate_at_t_location']
        self.assertArrayEqual(xx.to_masked_array(),self.xx)
        self.assertArrayEqual(yy.to_masked_array(),self.yy)

    def test_grid2d_from_plane_coordinate_array2d(self):
        grd = agrids.plane_2d_grid(xcoord=self.x,ycoord=self.y)
        xx = grd.arrays['plane_x_coordinate_at_t_location']
        yy = grd.arrays['plane_y_coordinate_at_t_location']
        self.assertArrayEqual(xx.to_masked_array(),self.xx)
        self.assertArrayEqual(yy.to_masked_array(),self.yy)
        self.assertArrayEqual(xx.to_masked_array()[0,:],self.x)
        self.assertArrayEqual(yy.to_masked_array()[:,0],self.y)


class TestCreationGrid2d_from_NEMO(TestCase):
    def setUp(self):
        self.coordfile = os.path.join(os.path.dirname(__file__), 'data', \
                                 'nemo-oocgcm-testdata_coordinates.nc')
        self.maskfile  = os.path.join(os.path.dirname(__file__), 'data', \
                                 'nemo-oocgcm-testdata_byte_mask.nc')
    def test_grid2d_from_netCDF(self):
        fgrd =  fgrids.nemo_2d_grid(nemo_coordinate_file=self.coordfile,\
                                    nemo_byte_mask_file=self.maskfile)
        assert(isinstance(fgrd,grids.generic_2d_grid))


class TestGrid2d_slicing_chunking(TestCase):
    def setUp(self):
        self.x = np.arange(start=0, stop=100, step=10,dtype=float)
        self.y = np.arange(start=0, stop=90, step=10,dtype=float)
        self.grd = agrids.latlon_2d_grid(latitudes=self.y,longitudes=self.x)

    def test_grid2d_slicing(self):
        sgrd = self.grd[:5,:5]
        assert(isinstance(sgrd,grids.generic_2d_grid))
        assert(sgrd.shape==(5,5))

    def test_grid2d_chunking(self):
        self.grd.chunk({'x':5,'y':5})
        assert(self.grd.chunks is not None)


class TestGrid2d_spatial_integration(TestCase):
    def setUp(self):
        x = np.arange(start=0, stop=1.e7, step=1.e6,dtype=float)
        y = np.arange(start=0, stop=1.2e7, step=1.e6,dtype=float)
        self.xx, self.yy = np.meshgrid(x,y)
        self.grd = agrids.plane_2d_grid(ycoord=self.yy,xcoord=self.xx)
        self.xax = self.grd.arrays["plane_x_coordinate_at_t_location"]
        self.xay = self.grd.arrays["plane_y_coordinate_at_t_location"]

    def test_integrate_dxdy(self):
        integral = self.grd.integrate_dxdy(self.xax>self.xay,grid_location='t')
        assert_equal(integral.values, 2.8e13)

    def test_spatial_average_xy(self):
        average = self.grd.spatial_average_xy(self.xax>self.xay,grid_location='t')
        assert_equal(average.values,0.35)


class TestGrid2d_DifferentialOperators(TestCase):
    def setUp(self):
        # only testing on a regular grid
        x = np.arange(start=0, stop=1.e7, step=1.e5,dtype=float)
        y = np.arange(start=0, stop=1.2e7, step=1.e5,dtype=float)
        x,y = np.meshgrid(x,y)
        self.grd = agrids.plane_2d_grid(ycoord=y,xcoord=x)
        self.scale = 2. * 3.14159 / 1.e7
        self.xt = self.grd.arrays['plane_x_coordinate_at_t_location']
        self.yt = self.grd.arrays['plane_y_coordinate_at_t_location']
        self.xu = self.grd.change_grid_location_t_to_u(self.xt)
        self.yu = self.grd.change_grid_location_t_to_u(self.yt)
        self.xv = self.grd.change_grid_location_t_to_v(self.xt)
        self.yv = self.grd.change_grid_location_t_to_v(self.yt)
        self.tvar1 = xu.sin(self.xt * self.scale) + xu.cos(self.yt * self.scale)
        self.uvar1 = xu.sin(self.xu * self.scale) + xu.cos(self.yu * self.scale)
        self.vvar1 = xu.sin(self.xv * self.scale) + xu.cos(self.yv * self.scale)
        self.vector = grids.VectorField2d(self.uvar1 ,self.vvar1,
                                          x_component_grid_location = 'u',
                                          y_component_grid_location = 'v')
        self.tvar2 = xu.cos(self.yt * self.scale)

    def test_horizontal_gradient(self):
        tols = {'rtol':1e-3,'atol':1e-3}
        gradvar =  self.grd.horizontal_gradient(self.tvar1)
        gradx   =   self.grd.horizontal_gradient(self.xt)
        #dxdy = gradx.y_component
        s = self.scale
        actual_gx = gradvar.x_component.to_masked_array()
        actual_gy = gradvar.y_component.to_masked_array()
        expected_gx = (xu.cos(self.xu * s) * s).to_masked_array()
        expected_gy = (- xu.sin(self.yv * s) * s
        #               + xu.cos(self.xv * s) * s * dxdy
                       ).to_masked_array()
        self.assertArray2dCloseInside(actual_gx / s,expected_gx / s ,
                                      depth=2,**tols)
        #print dxdy[:,20].values
        #print self.grd.arrays['cell_y_size_at_t_location'][:,20].values
        #print self.grd.arrays['cell_y_size_at_v_location'][:,20].values
        #print_array_around(expected=expected_gy/s,actual=actual_gy/s)
        self.assertArray2dCloseInside(actual_gy / s,expected_gy /s ,
                                      depth=2, **tols)

    def test_curl_of_horizontal_gradient(self):
        tols = {'rtol':1e-3,'atol':1e-3}
        s = self.scale
        grad = self.grd.horizontal_gradient(self.tvar1)
        curlgrad = self.grd.vertical_component_of_curl(grad)
        expected_curl = (0. * self.tvar1).to_masked_array()
        actual_curl = curlgrad.to_masked_array()
        #print grad.x_component.values[20:23,20:23]
        #print grad.y_component.values[20:23,20:23]
        #print_array_around(expected=expected_curl/s/s,actual=actual_curl/s/s)
        self.assertArray2dCloseInside(actual_curl/s/s,expected_curl/s/s,
                                         depth=4,**tols)

    def test_horizontal_divergence(self):
        # TODO : in 3D test divergence of curl is zero.
        tols = {'rtol':1e-3,'atol':1e-3}
        s = self.scale
        divvar = self.grd.horizontal_divergence(self.vector)
        dxdy = grids._dj(self.xv).shift(y=1) \
              / self.grd.arrays["cell_y_size_at_t_location"] # custom derivative
        actual_div = divvar.to_masked_array()
        expected_div = (xu.cos(self.xt * s) * s
                      - xu.sin(self.yt * s) * s
                      + xu.cos(self.xt * s) * s * dxdy
                        ).to_masked_array()
        #print_array_around(expected=expected_div/s,actual=actual_div/s)
        self.assertArray2dCloseInside(actual_div/s,expected_div/s,
                                      depth=2,**tols)

    def test_horizontal_laplacian(self):
        tols = {'rtol':1e-3,'atol':1e-3}
        l =  self.grd.horizontal_laplacian(self.tvar2)
        s = self.scale
        print s
        actual_lap = l.to_masked_array()
        expected_lap = (-1. *  xu.cos(self.yt * s) * s * s).to_masked_array()
        #print_array_around(expected=expected_lap/s**2,actual=actual_lap/s**2)
        self.assertArray2dCloseInside(actual_lap/s**2,expected_lap/s**2,
                                      depth=4,**tols)
