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
from ..core.utils import _append_dataarray_extra_attrs, is_numpy
from ..parameters.physicalparameters import earthrad
from ..parameters.mathematicalparameters import deg2rad

#================= Utilities for building grid descriptors =====================
#

def _horizontal_metrics_from_coordinates(xcoord,ycoord):
    """Return horizontal scale factors computed from arrays of projection
    coordinates.

        Parameters
        ----------
        xcoord : xarray dataarray
            array of x_coordinate used to  build the grid metrics.
            either plane_x_coordinate or projection_x_coordinate
            assume that the order of the dimensions is ('y','x').
        ycoord :xarray dataarray
            array of y_coordinate used to build the grid metrics.
            either plane_y_coordinate or projection_y_coordinate
            assume that the order of the dimensions is ('y','x').

        Return
        ------
        e1 : xarray dataarray
            Array of grid cell width corresponding to cell_x_size_at_*_location
        e2 : xarray dataarray
            Array of grid cell width corresponding to cell_y_size_at_*_location
    """
    #- Compute the centered first order derivatives of proj. coordinate arrays
    dy_dj,dy_di = _horizontal_gradient(ycoord)
    dx_dj,dx_di = _horizontal_gradient(xcoord)

    #- Compute the approximate size of the cells in x and y direction
    e1 = sqrt( dx_di**2. + dy_di**2. )
    e2 = sqrt( dx_dj**2. + dy_dj**2. )

    return e1,e2

def _horizontal_metrics_from_geographical_coordinates(latitudes,longitudes):
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
    #- Define the centered first order derivatives of lat/lon arrays
    dlat_dj,dlat_di = _horizontal_gradient(latitudes)
    dlon_dj,dlon_di = _horizontal_gradient(longitudes)

    #- Define the approximate size of the cells in x and y direction
    e1 = earthrad * deg2rad \
          * sqrt(( dlon_di * cos( deg2rad * latitudes ) )**2. + dlat_di**2.)
    e2 = earthrad * deg2rad \
          * sqrt(( dlon_dj * cos( deg2rad * latitudes ) )**2. + dlat_dj**2.)

    return e1,e2


def _masks_from_mask_at_t_location(tmask):
    """Return sea_binary_mask_at_[uvf]_location defined from
    sea_binary_mask_at_t_location.

    Parameters
    ----------
    tmask : xarray dataarray
        array of sea_binary_mask_at_t_location (mask==0 on land).
        assume that the order of the dimensions is ('y','x').

    Return
    ------
    masks : tuple of xarray dataarrays.
        umask refers to sea_binary_mask_at_u_location
        vmask refers to sea_binary_mask_at_v_location
        fmask refers to sea_binary_mask_at_f_location
    """
    umask = tmask  * tmask.shift(x=-1)
    vmask = tmask  * tmask.shift(y=-1)
    fmask = tmask \
              * tmask.shift(x=-1) \
              * tmask.shift(y=-1) \
              * tmask.shift(x=-1).shift(y=-1)
    return umask, vmask, fmask
#
#================= Defining variables for creating grid ========================
#
class variables_holder_for_2d_grid_from_plane_coordinate_arrays:
    """This class create the dictionnary of variables to be used for creating a
    oocgcm.core.grids.generic_2d_grid from arrays of plane coordinates.
    """
    def __init__(self,xcoord=None,ycoord=None,
                 mask=None,chunks=None,lat=45,lon=0):
        """This holder will create grid metrics and masks corresponding to a
        Arakawa C-grid assuming that xcoord, ycoord and mask refer to
        the centers of the cells (t-points). This class is meant to be used
        for idealized domains (eg.g on the f-plane of beta-plane)

        Parameters
        ----------
        xcoord : array-like (numpy array or xarray)
            array of x-coordinate from which to build the grid metrics.
            This can be either a one-dimensionnal or a two-dimensional array.
            For 2d arrays, assume that the order of the dimensions is ('y','x').
        ycoord : array-like (numpy array or xarray)
            array of y-coordinate from which to build the grid metrics.
            This can be either a one-dimensionnal or a two-dimensional array.
            For 2d arrays, assume that the order of the dimensions is ('y','x').
        mask : boolean array-like (numpy array or xarray)
            two-dimensional array describing the inner domain of the grid.
            mask==1 within the ocean, mask==0 on land.
        chunks : dict-like
            dictionnary of sizes of chunk for creating xarray.DataArray.
        lat : float or array-like
            latitude of t-points. This can be a float, a 1d or a 2d numpy array.
        lon : float or array-like
            longitude of t-points. This can be a float, a 1d or a 2d numpy array.

        """
        #
        self.variables = {}

        if len(xcoord.shape) != len(ycoord.shape):
            raise Exception('Arrays of coordinates should have '
                            + 'the same number of dimensions')
        if len(xcoord.shape) == 1 and len(ycoord.shape) == 1:
            xcoord = np.array(xcoord)
            ycoord = np.array(ycoord)
            xcoord, ycoord = np.meshgrid(xcoord,ycoord)
        if is_numpy(xcoord):
            xcoord = xr.DataArray(xcoord,dims=['y','x'])
        if is_numpy(ycoord):
            ycoord = xr.DataArray(ycoord,dims=['y','x'])
        self.shape = xcoord.shape

        xcoord = _append_dataarray_extra_attrs(xcoord,grid_location='t')
        ycoord = _append_dataarray_extra_attrs(ycoord,grid_location='t')
        self.variables\
            ["plane_x_coordinate_at_t_location"] = xcoord.chunk(chunks)
        self.variables\
            ["plane_y_coordinate_at_t_location"] = ycoord.chunk(chunks)

        if mask is not None:
            self.has_mask = True
            if is_numpy(mask):
                mask = xr.DataArray(mask,dims=['y','x'])
        else:
            self.has_mask = False
            mask = xr.DataArray(np.ones(self.shape,dtype=bool), dims=['y','x'])
        mask = mask.chunk(chunks)
        mask = _append_dataarray_extra_attrs(mask,grid_location='t')
        self.variables["sea_binary_mask_at_t_location"] = mask

        self._compute_plane_coordinates_at_all_locations()
        self._compute_horizontal_metrics_at_all_locations()
        self._compute_binary_masks_at_all_locations()

        if isinstance(lat,float):
            lat = np.array(lat)
        if len(lat.shape)==0:
            lat = float(lat) * np.ones(self.shape)
        elif len(lat.shape)==1:
            dum = np.zeros(self.shape[-1])
            lat = np.array(lat)
            dum, lat = np.meshgrid(dum,lat)
        if is_numpy(lat):
            lat = xr.DataArray(lat,dims=['y','x'])
        self.variables['latitude_at_t_location'] = lat

        if isinstance(lon,float):
            lon = np.array(lon)
        if len(lon.shape)==0:
            lon = float(lon) * np.ones(self.shape)
        elif len(lon.shape)==1:
            lon = np.array(lon)
            dum = np.zeros(self.shape[-2])
            lon,dum = np.meshgrid(lon,dum)
        if is_numpy(lon):
            lon = xr.DataArray(lon,dims=['y','x'])
        self.variables['longitude_at_t_location'] = lon

        for dataname in self.variables:
            data = self.variables[dataname]
            if isinstance(data, xr.DataArray):
                self.variables[dataname] = data.chunk(chunks)

        self.parameters = {}
        self.parameters['chunks'] = chunks

    def _compute_plane_coordinates_at_all_locations(self):
        """Define the plane coordinates at u,v,f location.

        Namely :
            - plane_x_coordinate_at_[u,v,f]_location,
            - plane_y_coordinate_at_[u,v,f]_location,

        """
        self.variables["plane_x_coordinate_at_u_location"] = _mi(
            self.variables["plane_x_coordinate_at_t_location"])
        self.variables["plane_x_coordinate_at_v_location"] = _mj(
            self.variables["plane_x_coordinate_at_t_location"])
        self.variables["plane_y_coordinate_at_u_location"] = _mi(
            self.variables["plane_y_coordinate_at_t_location"])
        self.variables["plane_y_coordinate_at_v_location"] = _mj(
            self.variables["plane_y_coordinate_at_t_location"])
        self.variables["plane_x_coordinate_at_f_location"] = _mj(
            self.variables["plane_x_coordinate_at_u_location"])
        self.variables["plane_y_coordinate_at_f_location"] = _mj(
            self.variables["plane_y_coordinate_at_u_location"])

    def _compute_horizontal_metrics_at_all_locations(self):
        """Define the horizontal metric terms (aka scale factors) (m).

        Namely :
            - cell_x_size_at_[tuvf]_location
            - cell_y_size_at_[tuvf]_location

        """
        for grid_location in ['t','u','v','f']:
            xname = "plane_x_coordinate_at_" + grid_location + "_location"
            yname = "plane_y_coordinate_at_" + grid_location + "_location"
            xcoord = self.variables[xname]
            ycoord = self.variables[yname]
            e1name = 'cell_x_size_at_' +  grid_location + '_location'
            e2name = 'cell_y_size_at_' +  grid_location + '_location'

            e1,e2 = _horizontal_metrics_from_coordinates(xcoord,ycoord)
            self.variables[e1name] = e1
            self.variables[e2name] = e2

    def _compute_binary_masks_at_all_locations(self):
        """Compute the mask (eg. ocean / land) at u,v,f locations.

        Namely, :
             - sea_binary_mask_at_u_location
             - sea_binary_mask_at_v_location
             - sea_binary_mask_at_f_location

        """
        tmask = self.variables["sea_binary_mask_at_t_location"]
        if not(self.has_mask): # no mask, only ocean points
            umask = _append_dataarray_extra_attrs(tmask.copy(),
                                                 grid_location='u')
            vmask = _append_dataarray_extra_attrs(tmask.copy(),
                                                 grid_location='v')
            fmask = _append_dataarray_extra_attrs(tmask.copy(),
                                                 grid_location='f')
        else:
            umask,vmask,tmask = _masks_from_mask_at_t_location(tmask)

        self.variables["sea_binary_mask_at_u_location"] = umask
        self.variables["sea_binary_mask_at_v_location"] = vmask
        self.variables["sea_binary_mask_at_f_location"] = fmask

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
        self.shape = longitudes.shape

        longitudes = _append_dataarray_extra_attrs(longitudes,grid_location='t')
        latitudes  = _append_dataarray_extra_attrs(latitudes,grid_location='t')
        self.variables\
            ["longitude_at_t_location"] = longitudes.chunk(chunks)
        self.variables\
            ["latitude_at_t_location"] = latitudes.chunk(chunks)

        if mask is not None:
            self.has_mask = True
            if is_numpy(mask):
                mask = xr.DataArray(mask,dims=['y','x'])
        else:
            self.has_mask = False
            mask = xr.DataArray(np.ones(self.shape,dtype=bool), dims=['y','x'])
        mask = mask.chunk(chunks)
        mask = _append_dataarray_extra_attrs(mask,grid_location='t')
        self.variables["sea_binary_mask_at_t_location"] = mask

        self._compute_geographical_coordinates_at_all_locations()
        self._compute_horizontal_metrics_at_all_locations()
        self._compute_binary_masks_at_all_locations()

        self.parameters = {}
        self.parameters['chunks'] = chunks

    def _compute_geographical_coordinates_at_all_locations(self):
        """Define the geographical coordinates at u,v,f location.

        Namely :
            - latitude_at_[u,v,f]_location,
            - longitude_at_[u,v,f]_location,

        """
        self.variables["longitude_at_u_location"] = _mi(
            self.variables["longitude_at_t_location"])
        self.variables["longitude_at_v_location"] = _mj(
            self.variables["longitude_at_t_location"])
        self.variables["latitude_at_u_location"] = _mi(
            self.variables["latitude_at_t_location"])
        self.variables["latitude_at_v_location"] = _mj(
            self.variables["latitude_at_t_location"])
        self.variables["longitude_at_f_location"] = _mj(
            self.variables["longitude_at_u_location"])
        self.variables["latitude_at_f_location"] = _mj(
            self.variables["latitude_at_u_location"])

    def _compute_horizontal_metrics_at_all_locations(self):
        """Define the horizontal metric terms (aka scale factors) (m).

        Namely :
            - cell_x_size_at_[tuvf]_location
            - cell_y_size_at_[tuvf]_location

        """
        for grid_location in ['t','u','v','f']:
            latname = "latitude_at_"  + grid_location + "_location"
            lonname = "longitude_at_" + grid_location + "_location"
            lat = self.variables[latname]
            lon = self.variables[lonname]
            e1name = 'cell_x_size_at_' +  grid_location + '_location'
            e2name = 'cell_y_size_at_' +  grid_location + '_location'

            e1,e2 = _horizontal_metrics_from_geographical_coordinates(lat,lon)
            self.variables[e1name] = e1
            self.variables[e2name] = e2

    def _compute_binary_masks_at_all_locations(self):
        """Compute the mask (eg. ocean / land) at u,v,f locations.

        Namely, :
             - sea_binary_mask_at_u_location
             - sea_binary_mask_at_v_location
             - sea_binary_mask_at_f_location

        """
        tmask = self.variables["sea_binary_mask_at_t_location"]
        if not(self.has_mask): # no mask, only ocean points
            umask = _append_dataarray_extra_attrs(tmask.copy(),
                                                 grid_location='u')
            vmask = _append_dataarray_extra_attrs(tmask.copy(),
                                                 grid_location='v')
            fmask = _append_dataarray_extra_attrs(tmask.copy(),
                                                 grid_location='f')
        else:
            umask,vmask,tmask = _masks_from_mask_at_t_location(tmask)

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
    grid = generic_2d_grid(arrays=variables.variables,
                           parameters= variables.parameters)
    return grid


def plane_2d_grid(xcoord=None,ycoord=None,mask=None,chunks=None,lat=45.,lon=0.):
    """Return a generic 2d grid build from arrays of coordinate.

    The grid object corresponds to a Arakawa C-grid assuming that
    xcoord, ycoord and mask refer to the centers of the cells.

    Parameters
    ----------
    xcoord : array-like (numpy array or xarray)
        array of x-coordinate from which to build the grid metrics.
        This can be either a one-dimensionnal or a two-dimensional array.
        For 2d arrays, assume that the order of the dimensions is ('y','x').
    ycoord : array-like (numpy array or xarray)
        array of y-coordinate from which to build the grid metrics.
        This can be either a one-dimensionnal or a two-dimensional array.
        For 2d arrays, assume that the order of the dimensions is ('y','x').
    mask : boolean array-like (numpy array or xarray)
        two-dimensional array describing the inner domain of the grid.
        mask==1 within the ocean, mask==0 on land.
    chunks : dict-like
        dictionnary of sizes of chunk for creating xarray.DataArray.
    lat : float or array-like
        latitude of t-points. This can be a float, a 1d or a 2d numpy array.
    lon : float or array-like
        longitude of t-points. This can be a float, a 1d or a 2d numpy array.

    Returns
    -------
    grid : oocgcm.core.grids.generic_2d_grid
        grid object corresponding to the input xcoord,ycoord arrays.
    """
    variables = variables_holder_for_2d_grid_from_plane_coordinate_arrays(
                     xcoord= xcoord,ycoord=ycoord,
                     mask= mask, chunks=chunks,
                     lat=lat,lon=lon)
    grid = generic_2d_grid(arrays=variables.variables,
                           parameters= variables.parameters)
    return grid
