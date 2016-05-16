#!/usr/bin/env python
#
"""oocgcm.griddeddata.grids
Define tools for building grid objects from array of latitude and
longitude.

"""
import numpy as np
import xarray as xr

from xarray.ufuncs import sqrt, cos

from ..core.grids import generic_2d_grid,_mi,_mj,_horizontal_gradient
from ..core.utils import add_extra_attrs_to_dataarray, is_numpy
from ..parameters.physicalparameters import earthrad
from ..parameters.mathematicalparameters import deg2rad

#================= Preparing the variable for the grid =========================
#

def _compute_e1e2_from_latlon(latitudes,longitudes):
    """Return horizontal scale factors computed from lat, lon arrays.

        Parameters
        ----------
        latitudes : xarray dataarray
            array of latitudes from which to build the grid metrics.
            assume that the order of the dimensions is ('y','x').
        longitudes :xarray dataarray
            array of latitudes from which to build the grid metrics.
            assume that the order of the dimensions is ('y','x').

        Return
        ------
        e1 : xarray dataarray
            Array of grid cell width corresponding to cell_x_size_at_*_location
        e2 : xarray dataarray
            Array of grid cell width corresponding to cell_y_size_at_*_location
    """
    # Compute the centered first order derivatives of lat/lon arrays
    dlat_dj,dlat_di = _horizontal_gradient(latitudes)
    dlon_dj,dlon_di = _horizontal_gradient(longitudes)
    # Compute the approximate size of the cells in x and y direction
    e1 = earthrad * deg2rad \
          * sqrt(( dlon_di * cos( deg2rad * latitudes ) )**2. + dlat_di**2.)
    e2 = earthrad * deg2rad \
          * sqrt(( dlon_dj * cos( deg2rad * latitudes ) )**2. + dlat_dj**2.)
    return e1,e2


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
                            + 'the same number of dimensions')
        if len(latitudes.shape) == 1 and len(longitudes.shape) == 1:
            latitudes = np.array(latitudes)
            longitudes = np.array(longitudes)
            longitudes, latitudes = np.meshgrid(longitudes,latitudes)
        if is_numpy(latitudes):
            latitudes = xr.DataArray(latitudes,dims=['y','x'])
        if is_numpy(longitudes):
            longitudes = xr.DataArray(longitudes,dims=['y','x'])
        if is_numpy(mask):
            mask = xr.DataArray(mask,dims=['y','x'])

        self.shape = longitudes.shape

        longitudes = add_extra_attrs_to_dataarray(longitudes,grid_location='t')
        latitudes  = add_extra_attrs_to_dataarray(latitudes,grid_location='t')
        self.variables\
            ["projection_x_coordinate_at_t_location"] = longitudes.chunk(chunks)
        self.variables\
            ["projection_y_coordinate_at_t_location"] = latitudes.chunk(chunks)

        if mask is not None:
            self.has_mask = True
        else:
            self.has_mask = False
            mask = xr.DataArray(np.ones(self.shape,dtype=bool), dims=['y','x'])
        mask = mask.chunk(chunks)
        mask = add_extra_attrs_to_dataarray(mask,grid_location='t')
        self.variables["sea_binary_mask_at_t_location"] = mask

        self._compute_projection_coordinates_at_all_locations()
        self._compute_horizontal_metrics_at_all_locations()
        self._compute_masks__at_all_locations()

        self.parameters = {}
        self.parameters['chunks'] = chunks

    def _compute_projection_coordinates_at_all_locations(self):
        """Compute the projection coordinates at u,v,f location.
        This includes :
            - projection_x_coordinate_at_u_location,
            - projection_y_coordinate_at_u_location,
            - projection_x_coordinate_at_v_location,
            - projection_y_coordinate_at_v_location,
            - projection_x_coordinate_at_t_location,
            - projection_y_coordinate_at_u_location,
        """
        self.variables["projection_x_coordinate_at_u_location"] = _mi(
            self.variables["projection_x_coordinate_at_t_location"])
        self.variables["projection_x_coordinate_at_v_location"] = _mj(
            self.variables["projection_x_coordinate_at_t_location"])
        self.variables["projection_y_coordinate_at_u_location"] = _mi(
            self.variables["projection_y_coordinate_at_t_location"])
        self.variables["projection_y_coordinate_at_v_location"] = _mj(
            self.variables["projection_y_coordinate_at_t_location"])
        self.variables["projection_x_coordinate_at_f_location"] = _mj(
            self.variables["projection_x_coordinate_at_u_location"])
        self.variables["projection_y_coordinate_at_f_location"] = _mj(
            self.variables["projection_y_coordinate_at_u_location"])

    def _compute_horizontal_metrics_at_all_locations(self):
        """Compute the horizontal metric terms (aka scale factors) (m).
        This includes :
            - cell_x_size_at_t_location, cell_y_size_at_t_location
            - cell_x_size_at_u_location, cell_y_size_at_u_location
            - cell_x_size_at_v_location, cell_y_size_at_v_location
            - cell_x_size_at_f_location, cell_y_size_at_f_location

        """
        for grid_location in ['t','u','v','f']:
            latname = "projection_y_coordinate_at_" + grid_location \
                      + "_location"
            lonname = "projection_x_coordinate_at_" + grid_location \
                      + "_location"
            lat = self.variables[latname]
            lon = self.variables[lonname]
            e1name = 'cell_x_size_at_' +  grid_location + '_location'
            e2name = 'cell_y_size_at_' +  grid_location + '_location'

            e1,e2 = _compute_e1e2_from_latlon(lat,lon)
            self.variables[e1name] = e1
            self.variables[e2name] = e2

    def _compute_masks__at_all_locations(self):
        """Compute the mask (eg. ocean / land) at u,v,f locations.

        Namely, :
             - sea_binary_mask_at_u_location
             - sea_binary_mask_at_v_location
             - sea_binary_mask_at_f_location

        """
        tmask = self.variables["sea_binary_mask_at_t_location"]
        if not(self.has_mask):
            umask = add_extra_attrs_to_dataarray(tmask.copy(),
                                                 grid_location='u')
            vmask = add_extra_attrs_to_dataarray(tmask.copy(),
                                                 grid_location='v')
            fmask = add_extra_attrs_to_dataarray(tmask.copy(),
                                                 grid_location='f')
        else:
            umask = tmask  * tmask.shift(x=-1)
            vmask = tmask  * tmask.shift(y=-1)
            fmask = tmask \
                  * tmask.shift(x=-1) \
                  * tmask.shift(y=-1) \
                  * tmask.shift(x=-1).shift(y=-1)

        self.variables["sea_binary_mask_at_u_location"] = umask
        self.variables["sea_binary_mask_at_v_location"] = vmask
        self.variables["sea_binary_mask_at_f_location"] = fmask

#================== Defining the grid from generic grids ==============================
#

def latlon_2d_grid(latitudes=None,longitudes=None,mask=None,chunks=None):
    """Return a generic 2d grid build from arrays of lat,lon.

    The grid object corresponds to a Arakawa C-grid assuming that
    latitudes, longitudes and mask refer to the centers of the cells.

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

    Returns
    -------
    grid : oocgcm.core.grids.generic_2d_grid
        grid object corresponding to the input lat,lon arrays.
    """
    variables = variables_holder_for_2d_grid_from_latlon_arrays(
                     longitudes= longitudes,latitudes=latitudes,
                     mask= mask, chunks=chunks)
    grid = generic_2d_grid(variables=variables.variables,
                           parameters= variables.parameters)
    return grid
