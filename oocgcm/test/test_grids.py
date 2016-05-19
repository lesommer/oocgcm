
import numpy as np

from . import TestCase

from oocgcm.griddeddata import grids

class TestGrid2d(TestCase):
    def setUp(self):
        self.x = np.arange(start=-180, stop=181, step=10,dtype=float)
        self.y = np.arange(start=-90, stop=91, step=10,dtype=float)
        self.xx,self.yy = np.meshgrid(self.x, self.y)

    def test_grid2d_from_array1d(self):
        grd = grids.latlon_2d_grid(latitudes=self.y,longitudes=self.x)
        xx = grd.arrays['projection_x_coordinate_at_t_location']
        yy = grd.arrays['projection_y_coordinate_at_t_location']
        self.assertArrayEqual(xx.to_masked_array(),self.xx)
        self.assertArrayEqual(yy.to_masked_array(),self.yy)

    def test_grid2d_from_array2d(self):
        grd = grids.latlon_2d_grid(latitudes=self.yy,longitudes=self.xx)
        xx = grd.arrays['projection_x_coordinate_at_t_location']
        yy = grd.arrays['projection_y_coordinate_at_t_location']
        self.assertArrayEqual(xx.to_masked_array(),self.xx)
        self.assertArrayEqual(yy.to_masked_array(),self.yy)
        self.assertArrayEqual(xx.to_masked_array()[0,:],self.x)
        self.assertArrayEqual(yy.to_masked_array()[:,0],self.y)
