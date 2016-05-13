#!/usr/bin/env python
#
"""oocgcm.griddeddata.grids
Define tools for building grid objects from array of latitude and
longitude.

"""
import xarray as xr
import numpy as np

from ...core.grids import generic_2d_grid
from ...core.utils import add_extra_attrs_to_dataarray


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
            raise Exception('longitudes and latitudes arrays should have '
                            + 'identical shapes')
        if len(latitudes.shape) == 1 and len(longitudes.shape) == 1:
            latitudes = np.array(latitudes)
            longitudes = np.array(longitudes)
            longitudes, latitudes = np.meshgrid(longitudes,latitudes)
        if is_numpy(latitudes):
            latitudes = xr.Dataarray(latitudes,dims=['y','x'])
        if is_numpy(longitudes):
            longitudes = xr.Dataarray(longitudes,dims=['y','x'])
        if is_numpy(mask):
            mask = xr.Dataarray(mask,dims=['y','x'])

        self.shape = longitudes.shape

        longitudes = add_extra_attrs_to_dataarray(longitudes,grid_location='t')
        latitudes = add_extra_attrs_to_dataarray(latitudes,grid_location='t')
        self.variables\
            ["projection_x_coordinate_at_t_location"] = longitudes.chunk(chunks)
        self.variables\
            ["projection_y_coordinate_at_t_location"] = latitudes.chunk(chunks)

        if mask is not None:
            self.has_mask = True
            mask = mask.chunk(chunks)
        else:
            self.has_mask = False
            mask = xr.DataArray(np.ones(self.shape,dtype=bool),chunks=chunks)
        mask = add_extra_attrs_to_dataarray(mask,grid_location='t')
        self.variables["sea_binary_mask_at_t_location"] = mask

        self._compute_projection_coordinates_at_all_locations()
        self._compute_horizontal_metrics_at_all_locations()
        self._compute_masks__at_all_locations()

        def _compute_projection_coordinates_at_all_locations(self):
            """Compute the projection coordinates at u,v,f location.
            This includes :
                - projection_x_coordinate_at_u_location,
                - projection_y_coordinate_at_u_location,
                - projection_x_coordinate_at_v_location,
                - projection_y_coordinate_at_v_location,
                - projection_x_coordinate_at_t_location,
                - projection_y_coordinate_at_u_location,

            Not implemented yet.

            """
            # not sure whether this is needed in generic_2d_grid
            pass

        def _compute_horizontal_metrics_at_all_locations(self):
            """Compute the horizontal metric terms (aka scale factors) (m).
            This includes :
                - cell_x_size_at_t_location, cell_y_size_at_t_location
                - cell_x_size_at_u_location, cell_y_size_at_u_location
                - cell_x_size_at_v_location, cell_y_size_at_v_location
                - cell_x_size_at_f_location, cell_y_size_at_f_location

            Not implemented yet.

            """
            pass

        def _compute_masks__at_all_locations(self):
            """Compute the mask (eg. ocean / land) at u,v,f locations.
            This includes :

            Not implemented yet.
                            
            """
            pass
