#!/usr/bin/env python
#
"""oocgcm.griddeddata.grids
Define tools for building grid objects from array of latitude and
longitude.

"""
import xarray as xr
import numpy as np

from ...core.grids import generic_2d_grid
from ...core.io import return_xarray_dataarray

#==================== Variables holders for NEMO ===============================
#
def is_numpy(array):
    """Return True if array ia a numpy
    """
    test =  bool( isinstance(array,np.ndarray)
                  + isinstance(array,np.ma.masked_array) )
    return test


class variables_holder_for_2d_grid_from_latlon_arrays:
    """This class create the dictionnary of variables to be used for creating a
    oocgcm.core.grids.generic_2d_grid from arrays of latitude and longitude.
    """
    def __init__(self,latitudes=None,longitudes=None,mask=None,chunks=None):
        """This holder will create grid metrics and masks corresponding to a
        Arakawa C-grid assuming that latitudes, longitudes and mask refer to
        the centers of the cells (t-points).

        Parameters
        ----------
        latitudes : array-like (numpy array or xarray)
            array of latitudes from which to build the grid metrics. This can be
            either a one-dimensionnal or a two-dimensional array.
            For 2d arrays, assume that the order of the dimensions is ('y','x').
        longitudes : array-like (numpy array or xarray)
            array of longitudes from which to build the grid metrics. This can
            be either a one-dimensionnal or a two-dimensional array.
            For 2d arrays, assume that the order of the dimensions is ('y','x').
        mask : boolean array-like (numpy array or xarray)
            two-dimensional array describing the inner domain of the grid.
            mask==1 within the ocean, mask==0 on land.
        chunks : dict-like
            dictionnary of sizes of chunk for creating xarray.DataArray.
        """
        #
        self.variables = {}
        if len(latitudes.shape) != len(longitudes.shape):
            raise Exception('longitudes and latitudes arrays have'
                            + 'similar dimensions')
        if len(latitudes.shape) == 1 and len(longitudes.shape) == 1:
            latitudes = np.array(latitudes)
            longitudes = np.array(longitudes)
            longitudes, latitudes = np.meshgrid(longitudes,latitudes)
        if is_numpy(latitudes):
            latitudes = xr.Dataarray(latitudes,dims=['y','x'])
        if is_numpy(longitudes):
            longitudes = xr.Dataarray(longitudes,dims=['y','x'])
        self.variables\
            ["projection_x_coordinate_at_t_location"] = longitudes.chunk(chunks)
        self.variables\
            ["projection_y_coordinate_at_t_location"] = latitudes.chunk(chunks)
