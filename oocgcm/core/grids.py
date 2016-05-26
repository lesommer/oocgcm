#!/usr/bin/env python
#
"""oocgcm.core.grids
Define classes that give acces to grid metrics and differential operators.

"""
from collections import namedtuple # for vector data structures

import numpy as np
import xarray as xr
import dask.array as da
import xarray.ufuncs as xu         # ufuncs like np.sin for xarray


from .utils import is_numpy,is_xarray,has_chunks,add_extra_attrs_to_dataarray
from ..parameters.physicalparameters import coriolis_parameter, grav
from ..parameters.physicalparameters import earthrad
from ..parameters.mathematicalparameters import deg2rad

#
#==================== Differences and Averages =================================
#
def _horizontal_gradient(scalararray):
    """Return the gradient of a scalararray

    Parameters
    ----------
    scalararray : xarray.DataArray
        xarray that should be differentiated. So far scalararray should be 2d
        with dimensions ('y','x').

    Return
    ------
    da_dj,da_di : tuple of xarray.DataArray
        two arrays of the same shape as scalararray giving the derivative of
        scalararray with respect to each dimension ('y','x').

    Method
    ------
    Wrap numpy.gradient

    Caution
    -------
    Not fully functional yet : problem with boundary values
    see https://github.com/lesommer/oocgcm/issues/21
    """
    # TODO : solve https://github.com/lesommer/oocgcm/issues/21
    data = scalararray.data
    coords = scalararray.coords
    dims = scalararray.dims
    chunks = scalararray.chunks
    if is_numpy(data):
        da_dj,da_di = np.gradient(data)
    else:
        x_derivative = lambda arr:np.gradient(arr,axis=-1) # req. numpy > 1.11
        y_derivative = lambda arr:np.gradient(arr,axis=-2) # req. numpy > 1.11
        gx = data.map_overlap(x_derivative,depth=(0,1),boundary={1: np.nan})
        gy = data.map_overlap(y_derivative,depth=(1,0),boundary={0: np.nan})
    da_di = xr.DataArray(gx,coords,dims)
    da_dj = xr.DataArray(gy,coords,dims)
    return da_dj,da_di

def _di(scalararray):
    """Return the difference scalararray(i+1) - scalararray(i).

    A priori, for internal use only.

    Parameters
    ----------
    scalararray : xarray.DataArray
        xarray that should be differentiated.

    Returns
    -------
    di : xarray.DataArray
       xarray of difference, defined at point i+1/2
    """
    di = scalararray.shift(x=-1) - scalararray
    return di

def _dj(scalararray):
    """Return the difference scalararray(j+1) - scalararray(j)

    A priori, for internal use only.

    Parameters
    ----------
    scalararray : xarray.DataArray
        xarray that should be differentiated.

    Returns
    -------
    dj : xarray.DataArray
       xarray of difference, defined at point j+1/2
    """
    dj = scalararray.shift(y=-1) - scalararray
    return dj


def _mi(scalararray):
    """Return the average of scalararray(i+1) and scalararray(i)

    A priori, for internal use only.

    Parameters
    ----------
    scalararray : xarray.DataArray
        xarray that should be averaged at i+1/2.

    Returns
    -------
    mi : xarray.DataArray
       averaged xarray, defined at point i+1/2
    """
    mi = ( scalararray.shift(x=-1) + scalararray ) / 2.
    return mi

def _mj(scalararray):
    """Return the average of scalararray(j+1) and scalararray(j)

    A priori, for internal use only.

    Parameters
    ----------
    scalararray : xarray.DataArray
        xarray that should be averaged at j+1/2.

    Returns
    -------
    mj : xarray.DataArray
       averaged xarray, defined at point j+1/2
    """
    mj = (scalararray.shift(y=-1) + scalararray ) / 2.
    return mj

#
#==================== Methods for testing xarrays ==============================
#
def _finalize_dataarray_attributes(xarr,**kwargs):
    """Update the dictionary of attibutes of a xarray dataarray.

    Parameters
    ----------
    xarr : xarray.DataArray
       xarray dataarray which attributes will be updated.
    kwargs : dict-like object
       dictionnary of attributes

    Returns
    -------
    xarr : xarray.DataArray
    """
    if isinstance(xarr, xr.DataArray):
        xarr.attrs.update(kwargs)
    if xarr.attrs.has_key('short_name'):
        xarr.name = xarr.attrs['short_name']
    return xarr

def convert_dataarray_attributes_xderivative(attrs,grid_location=None):
    """Return the dictionary of attributes corresponding to the spatial
    derivative of a scalar field in the x-direction

    Parameters
    ----------
    attrs : dict-like object
       dictionnary of attributes
    grid_location : str
        string describing the grid location : eg 'u','v','t','f'...

    Returns
    -------
    new_attrsarr : dict-like object
        dictionnary of attributes of the spatial derivative.
    """
    new_attrs = attrs.copy()
    if attrs.has_key('long_name'):
        new_attrs['long_name'] = 'x-derivative of ' + attrs['long_name']
    if attrs.has_key('short_name'):
        new_attrs['short_name'] = 'd_' + attrs['short_name'] + '_dx'
    if attrs.has_key('units'):
        new_attrs['units'] = attrs['units'] + '/m'
    if grid_location is not None:
        new_attrs['grid_location'] = grid_location
    return new_attrs

def convert_dataarray_attributes_yderivative(attrs,grid_location=None):
    """Return the dictionary of attributes corresponding to the spatial
    derivative of a scalar field in the y-direction.

    Parameters
    ----------
    attrs : dict-like object
       dictionnary of attributes
    grid_location : str
        string describing the grid location : eg 'u','v','t','f'...

    Returns
    -------
    new_attrsarr : dict-like object
        dictionnary of attributes of the spatial derivative.
    """
    new_attrs = attrs.copy()
    if attrs.has_key('long_name'):
        new_attrs['long_name'] = 'y-derivative of ' + attrs['long_name']
    if attrs.has_key('short_name'):
        new_attrs['short_name'] = 'd_' + attrs['short_name'] + '_dy'
    if attrs.has_key('units'):
        new_attrs['units'] = attrs['units'] + '/m'
    if grid_location is not None:
        new_attrs['grid_location'] = grid_location
    return new_attrs


def convert_dataarray_attributes_laplacian(attrs,grid_location='t'):
    """Return the dictionary of attributes corresponding to the horizontal
    laplacian of a scalar field.

    Parameters
    ----------
    attrs : dict-like object
       dictionary of attributes
    grid_location : str
        string describing the grid location : eg 'u','v','t','f'...

    Returns
    -------
    new_attrsarr : dict-like object
        dictionnary of attributes of the laplacian.
    """
    new_attrs = attrs.copy()
    if attrs.has_key('long_name'):
        new_attrs['long_name'] = 'horizontal laplacian of ' + attrs['long_name']
    if attrs.has_key('short_name'):
        new_attrs['short_name'] = 'hlap_' + attrs['short_name']
    if attrs.has_key('units'):
        new_attrs['units'] = attrs['units'] + '/m2'
    if grid_location is not None:
        new_attrs['grid_location'] = grid_location
    return new_attrs

def convert_dataarray_attributes_divergence(attrs1,attrs2,grid_location='t'):
    """Return the dictionary of attributes corresponding to the divergence of a
    vector field.

    Parameters
    ----------
    attrs1 : dict-like object
       dictionnary of attributes of the x-component of a vector field
    attrs2 : dict-like object
       dictionnary of attributes of the y-component of a vector field
    grid_location : str
        string describing the grid location : eg 'u','v','t','f'...

    Returns
    -------
    new_attrsarr : dict-like object
        dictionnary of attributes of the divergence field.
    """
    new_attrs = attrs1.copy()
    if attrs1.has_key('long_name') and attrs2.has_key('long_name'):
        new_attrs['long_name'] = \
           'horizontal divergence of ('\
           + attrs1['long_name'] + ','\
           + attrs2['long_name'] + ')'
    if attrs1.has_key('short_name') and attrs2.has_key('short_name'):
            new_attrs['short_name'] = 'div_()' + attrs1['short_name'] + ','\
                                               + attrs1['short_name'] + ')'
    if attrs1.has_key('units'):
            new_attrs['units'] = attrs1['units'] + '/m'
    if grid_location is not None:
        new_attrs['grid_location'] = grid_location
    return new_attrs

def assert_chunks_are_compatible(chunks1=None,chunks2=None,ndims=None):
    """Return True when two chunks are aligned over their common dimensions.

    Parameters
    ----------
    chunks1 : list-like of list-like object
        chunks associated to a xarray data array
    chunks2 : list-like of list-like object
        chunks associated to a xarray data array
    ndims : int
        number of dimensions over which chunks should ne compared.

    Returns
    -------
    test : bool
        boolean value of the test.

    """
    # TODO : not clear whether to compare a priori description of chunks
    #        (dictionnaries) or a posteriori values (tuple of tuples).
    test = True
    if (chunks1 is None) or (chunks2 is None):
        if (chunks1 is None) and (chunks2 is None):
            return True
        else:
            return False
    for idim in range(ndims):
        test *= chunks1[-idim-1] == chunks2[-idim-1]
    return test

def assert_grid_location(xarr,grid_location=None):
    """Return True when the xarr grid_location attribute is grid_location

    Parameters
    ----------
    xarr : xarray.DataArray
       xarray dataarray which attributes should be tested.
    grid_location : str
        string describing the grid location : eg 'u','v','t','f'...

    Returns
    -------
    test : bool
        boolean value of the test.

    """
    test = True
    if xarr.attrs.has_key('grid_location'):
        test *= (xarr.attrs['grid_location']==grid_location)
    return test

def check_input_array(xarr,shape=None,chunks=None,\
                      grid_location=None,ndims=None):
    """Return true if arr is a dataarray with expected shape, chunks at
    grid_location attribute. Raise an error if one of the tests fails.

    Parameters
    ----------
    xarr : xarray.DataArray
       xarray dataarray which attributes should be tested.
    shape : tuple
       expected shape of the xarray dataarray xarr
    chunks : list-like of list-like object
        expected chunks of the xarray dataarray xarr
    grid_location : str
        string describing the expected grid location : eg 'u','v','t','f'...
    ndims : int
        number of dimensions over which chunks should be compared.

    Returns
    -------
    test : bool
        boolean value of the test.

    """
    if hasattr(xarr,'name'):
       arrayname = xarr.name
    else:
       arrayname = 'array'
    if not(isinstance(xarr,xr.DataArray)):
        raise TypeError(arrayname + 'is expected to be a xarray.DataArray')
        return False
    if not(assert_chunks_are_compatible(xarr.chunks,chunks,ndims=ndims)):
        raise ChunkError()
        return False
    if not(assert_grid_location(xarr,grid_location)):
        raise GridLocationError()
        return False
    return True

#
#========================== Data structures ====================================
#

#
# data structures for vector fields and tensors.
#

def assert_and_set_grid_location_attribute(xarr,grid_location=None):
    """Assert whether xarr holds an extra attribute 'grid_location' that
    equals grid_location. If xarr does not have such extra-attribute, create
    one with value grid_location.

    Parameters
    ----------
    xarr : xarray.DataArray
        xarray dataarray that should be associated with a grid location
    grid_location : str
        string describing the grid location : eg 'u','v','t','f'...
    """
    if xarr.attrs.has_key('grid_location'):
        assert ( xarr.attrs['grid_location'] == grid_location )
    else:
        xarr.attrs['grid_location'] = grid_location

def VectorField2d(vx,vy,\
                  x_component_grid_location=None,\
                  y_component_grid_location=None):
    """Minimal data structure for manupulating 2d vector fields on a grid.

    Parameters
    ----------
    vx : xarray.DataArray
        x-component of the vector fields
    vy : xarray.DataArray
        y-component of the vector fields
    x_component_grid_location : str
        string describing the grid location of the x-component
    y_component_grid_location : str
        string describing the grid location of the y-component

    Returns
    -------
    o : namedtuple
       namedtuple containing the vector field.
    """
    v = namedtuple('VectorField2d',['x_component','y_component',\
                                    'x_component_grid_location',\
                                    'y_component_grid_location'])
    #
    assert_and_set_grid_location_attribute(vx,x_component_grid_location)
    assert_and_set_grid_location_attribute(vy,y_component_grid_location)
    #
    o = v(vx,vy,x_component_grid_location,y_component_grid_location)
    return o

def VectorField3d(vx,vy,vz,\
                  x_component_grid_location=None,\
                  y_component_grid_location=None,\
                  z_component_grid_location=None):
    """Minimal data structure for manupulating 3d vector fields on a grid.

    Parameters
    ----------
    vx : xarray.DataArray
        x-component of the vector fields
    vy : xarray.DataArray
        y-component of the vector fields
    vz : xarray.DataArray
        z-component of the vector fields
    x_component_grid_location : str
        string describing the grid location of the x-component
    y_component_grid_location : str
        string describing the grid location of the y-component
    z_component_grid_location : str
        string describing the grid location of the z-component

    Returns
    -------
    o : namedtuple
           namedtuple containing the vector field.
    """
    v = namedtuple('VectorField3d',['x_component','y_component','z_component',\
                                    'x_component_grid_location',\
                                    'y_component_grid_location',\
                                    'z_component_grid_location'])
    #
    assert_and_set_grid_location_attribute(vx,x_component_grid_location)
    assert_and_set_grid_location_attribute(vy,y_component_grid_location)
    assert_and_set_grid_location_attribute(vz,z_component_grid_location)
    #
    o = v(vx,vy,vz,\
          x_component_grid_location,\
          y_component_grid_location,\
          z_component_grid_location)
    return o

def Tensor2d(axx,axy,ayx,ayy,\
             xx_component_grid_location=None,\
             xy_component_grid_location=None,\
             yx_component_grid_location=None,\
             yy_component_grid_location=None):
    """Minimal data structure for manupulating 2d tensors on a grid.

    use the following notations :
                | axx   axy|
            T = |          |
                | ayx   ayy|
    Parameters
    ----------
    axx : xarray.DataArray
        xx-component of the tensor
    axy : xarray.DataArray
        xy-component of the tensor
    ayx : xarray.DataArray
        yx-component of the tensor
    ayy : xarray.DataArray
        yy-component of the tensor
    xx_component_grid_location : str
        string describing the grid location of the xx-component
    xy_component_grid_location : str
        string describing the grid location of the xy-component
    yx_component_grid_location : str
        string describing the grid location of the yx-component
    yy_component_grid_location : str
        string describing the grid location of the yy-component

    Returns
    -------
    o : namedtuple
           namedtuple containing the tensor.
    """
    t = namedtuple('Tensor2d',['xx_component','xy_component',\
                               'yx_component','yy_component',\
                               'xx_component_grid_location',\
                               'xy_component_grid_location',\
                               'yx_component_grid_location',\
                               'yy_component_grid_location'])
    #
    if axx.attrs.has_key('grid_location'):
        assert (axx.attrs['grid_location'] == xx_component_grid_location )
    else:
        axx.attrs['grid_location'] = xx_component_grid_location
    #
    assert_and_set_grid_location_attribute(axx,xx_component_grid_location)
    assert_and_set_grid_location_attribute(axy,xy_component_grid_location)
    assert_and_set_grid_location_attribute(ayx,yx_component_grid_location)
    assert_and_set_grid_location_attribute(ayy,yy_component_grid_location)
    #
    o = t(axx,axy,ayx,ayy,\
          xx_component_grid_location,xy_component_grid_location,\
          yx_component_grid_location,yy_component_grid_location)
    return o

#
#========================== Minimal exceptions =================================
#
# TODO : should probably move to a dedicated file oocgcm.core.exceptions.py
#
class ChunkError(Exception):
    """Minimal exception for chunk incompatibility.
    """
    def __init__(self):
        Exception.__init__(self,"incompatible chunk size")

class GridLocationError(Exception):
    """Minimal exception for grid location incompatibility.
    """
    def __init__(self):
        Exception.__init__(self,"incompatible grid_location")

#
#======================== Generic 2D Grid Class ================================
#

class generic_2d_grid:
    """Model agnostic grid object, two dimensional version.

    This class holds the xarrays that describe the grid and
    implements grid related methods.

    This includes :
        - vector calculus (scalar product, norm, vector product,...)
        - interpolation between different grid locations (eg. 'u'-->'t')
        - differential operators (gradient, divergence, etc...)
        - spatial integration

    Most methods expect and return instance of xarray.DataArray.

    Assume that dimension names are 'x' and 'y'.
    """
    # the following arrays should be defined for the grid to be functional.
    # attempts to build the gride withouth these arrays will raise an Exception.
    _required_arrays = [\
        "latitude_at_t_location",\
        "longitude_at_t_location",\
        "sea_binary_mask_at_t_location",\
        "sea_binary_mask_at_u_location",\
        "sea_binary_mask_at_v_location",\
        "sea_binary_mask_at_f_location",\
        "cell_x_size_at_t_location",\
        "cell_x_size_at_u_location",\
        "cell_x_size_at_v_location",\
        "cell_y_size_at_t_location",\
        "cell_y_size_at_u_location",\
        "cell_y_size_at_v_location",\
        ]

    # the following arrays may or may not be available depending on how the
    # grid was created.
    _coordinate_arrays = [\
        "projection_x_coordinate_at_t_location",\
        "projection_y_coordinate_at_t_location",\
        "plane_x_coordinate_at_t_location",\
        "plane_y_coordinate_at_t_location",\
        ]

    def __init__(self,arrays=None,parameters=None):
        """Initialize a grid from a dictionary of xarrays and some parameters.

        Parameters
        ----------
        variables : dict-like object
            dictionnary of xarrays that describe the grid. Required variables
            for the method actually implemented are listed in _required_arrays.
            This naming convention follows a mixture of cf and comodo norms in
            order for this class to be model-agnostic.

        parameters : dict-object
            not used yet.
        """
        for arrayname in self._required_arrays:
            if not(arrays.has_key(arrayname)):
                raise Exception('Arrays are missing for building the grid.')
        self._arrays = arrays
        self._extra_parameters = parameters
        self._define_aliases_for_arrays()
        self._define_area_of_grid_cells()
        self._define_extra_latitude_longitude()
        self._define_coriolis_parameter()

#--------------------- Public attributes ---------------------------------------
    @property
    def dims(self):
        """Dimensions of the xarray dataarrays describing the grid.
        """
        dims = list(self._arrays["sea_binary_mask_at_t_location"].dims)
        if 't' in dims: dims.remove('t')
        return tuple(dims)

    @property
    def ndims(self):
        """Number of dimensions of the dataarrays describing the grid.
        """
        return len(self.dims)

    @property
    def shape(self):
        """Shape of the xarray dataarrays describing the grid.
        """
        return self._arrays["sea_binary_mask_at_t_location"].squeeze().shape

    @property
    def chunks(self):
        """Chunks of the xarray dataarrays describing the grid.
        """
        # TODO : not clear whether to store the priori description of chunks
        #        (dictionnary) or the posteriori value (tuple of tuples).

        return self._arrays["sea_binary_mask_at_t_location"].chunks

#--------------------- Extra xarrays for the grid ------------------------------

    def _define_area_of_grid_cells(self):
        """Define arrays of area at u,v,t,f grid_location.

        This is only a definition, the computation is performed only if used.
        """
        for gloc in ['t','u','v','f']:
            self._arrays["cell_area_at_" + gloc + "_location"] = \
                    self._arrays["cell_x_size_at_" + gloc + "_location"] \
                  * self._arrays["cell_y_size_at_" + gloc + "_location"]

    def _define_extra_latitude_longitude(self):
        """Define projection coordinates at u,v,f grid_location if needed.

        This is only a definition, the computation is performed only if used.
        """
        # latitude and longitude arrays at u location
        lonname = "longitude_at_u_location"
        latname = "latitude_at_u_location"
        if not(self._arrays.has_key(lonname)):
            self._arrays[lonname] = _mi(
                        self._arrays["longitude_at_t_location"])
        if not(self._arrays.has_key(latname)):
            self._arrays[latname] = _mi(
                        self._arrays["latitude_at_t_location"])

        # latitude and longitude arrays at v location
        lonname = "longitude_at_v_location"
        latname = "latitude_at_v_location"
        if not(self._arrays.has_key(lonname)):
            self._arrays[lonname] = _mj(
                        self._arrays["longitude_at_t_location"])
        if not(self._arrays.has_key(latname)):
            self._arrays[latname] = _mj(
                        self._arrays["latitude_at_t_location"])

        # latitude and longitude arrays at f location
        lonname = "longitude_at_f_location"
        latname = "latitude_at_f_location"
        if not(self._arrays.has_key(lonname)):
            self._arrays[lonname] = _mj(
                        self._arrays["longitude_at_u_location"])
        if not(self._arrays.has_key(latname)):
            self._arrays[latname] = _mj(
                        self._arrays["latitude_at_u_location"])

    def _define_coriolis_parameter(self):
        """Define arrays of coriolis parameter at t,u,v,f grid_location.

        This is only a definition, the computation is performed only if used.
        """
        for gloc in ['t','u','v','f']:
            corname = "coriolis_parameter_at_" + gloc + "_location"
            latname = "latitude_at_" + gloc + "_location"
            self._arrays[corname] = coriolis_parameter(self._arrays[latname])


#--------------------- Aliases for DataArrays ----------------------------------
    def _define_aliases_for_arrays(self):
        """Define alias for frequently used variables.

        The following shortcuts mostly follows nemo name conventions.
        This could evolve in the future and should therefore not be used in
        external libraries.
        """
        # coordinates
        self._array_navlon = self._arrays["longitude_at_t_location"]
        self._array_navlat = self._arrays["latitude_at_t_location"]
        # metrics
        self._array_e1t = self._arrays["cell_x_size_at_t_location"]
        self._array_e1u = self._arrays["cell_x_size_at_u_location"]
        self._array_e1v = self._arrays["cell_x_size_at_v_location"]
        self._array_e2t = self._arrays["cell_y_size_at_t_location"]
        self._array_e2u = self._arrays["cell_y_size_at_u_location"]
        self._array_e2v = self._arrays["cell_y_size_at_v_location"]

        # masks
        self._array_tmask = self._arrays["sea_binary_mask_at_t_location"]
        self._array_umask = self._arrays["sea_binary_mask_at_u_location"]
        self._array_vmask = self._arrays["sea_binary_mask_at_v_location"]
        self._array_fmask = self._arrays["sea_binary_mask_at_f_location"]

#--------------------- Chunking and Slicing ------------------------------------
    def chunk(self,chunks=None):
        """Rechunk all the variables defining the grid.

        Parameters
        ----------
        chunks : dict-like
            dictionnary of sizes of chunk along xarray dimensions.

        Example
        -------
        >>> xr_chunk = {'x':200,'y':200}
        >>> grd = generic_2d_grid(...)
        >>> grd.chunk(xr_chunk)

        """
        for dataname in self._arrays:
            data = self._arrays[dataname]
            if isinstance(data, xr.DataArray):
                self._arrays[dataname] = data.chunk(chunks)
        self.chunks = self._arrays["sea_binary_mask_at_t_location"].chunks

    def __getitem__(self,item):
        """The behavior of this fcuntion depends on the type of item.
            - if item is a string, return the array self._arrays[item]
            - if item is a slice, a grid object restricted to a subdomain.

        Use slicing with caution, this functionnality depends on the order of
        the dimensions in the netcdf files.

        Parameters
        ----------
        item : slice or str
            item can be a string identifying a key in self._arrays
            item can be a slice for restricting the grid to a subdomain.

        Returns
        -------
        out :  xarray.DataArray or generic_2d_grid
            either a dataarray corresponding to self._arrays[item]
            or a new grid object corresponding to the restructed region.

        Example
        -------
        >>> grd = generic_2d_grid(...)
        restricting the grid to a subdomain :
        >>> new_grd = grd[100:200,300:500]
        accessing a specific dataarray describing the grid
        >>> e1t = grd['cell_x_size_at_t_location']

        """
        if isinstance(item,str):
            returned = self._arrays[item]

        else:
            sliced_arrays = {}
            for dataname in self._arrays:
                sliced_arrays[dataname] = self._arrays[dataname][item]
            returned =  generic_2d_grid(arrays=sliced_arrays,\
                                        parameters=self._extra_parameters)

        return returned

    def __contains__(self, key):
        """The 'in' operator will return true or false depending on whether
        'key' is an array in the dataset or not.
        """
        return key in self._arrays

    def __iter__(self):
        return iter(self._arrays)

#---------------------------- Misc utilities ------------------------------------
#-
    def get_projection_coordinates(self,grid_location='t'):
        """Return (x,y) the coordinate arrays (in m) at grid location.

        Caution : This function may change in future versions of oocgcm.

	    """
        lat = self._arrays['latitude_at_' + grid_location + '_location']
        lon = self._arrays['longitude_at_' + grid_location + '_location']
        x = earthrad  * deg2rad * lon * xu.cos(lat * deg2rad)
        y = earthrad  * deg2rad * lat
        return x,y

#---------------------------- Grid Swapping ------------------------------------
#- Core swapping utilities
    def _to_eastern_grid_location(self,scalararray,
                                  weights_in=None,weights_out=None):
        """Return an average of scalararray at point i + 1/2
        """
        average = lambda xarr:( xarr.shift(x=-1) + xarr ) / 2.
        if weights_in is None:
            out = average(scalararray)
        else:
            if weights_out is None:
                weights_out = average(weights_in)
            out = average(scalararray * weights_in) / weights_out
        return out

    def _to_western_grid_location(self,scalararray,
                                  weights_in=None,weights_out=None):
        """Return an average of scalararray at point i - 1/2
        """
        average = lambda xarr:( xarr.shift(x=1) + xarr ) / 2.
        if weights_in is None:
            out = average(scalararray)
        else:
            if weights_out is None:
                weights_out = average(weights_in)
            out = average(scalararray * weights_in) / weights_out
        return out

    def _to_northern_grid_location(self,scalararray,
                                  weights_in=None,weights_out=None):
        """Return an average of scalararray at point j + 1/2
        """
        average = lambda xarr:( xarr.shift(y=-1) + xarr ) / 2.
        if weights_in is None:
            out = average(scalararray)
        else:
            if weights_out is None:
                weights_out = average(weights_in)
            out = average(scalararray * weights_in) / weights_out
        return out

    def _to_southern_grid_location(self,scalararray,
                                  weights_in=None,weights_out=None):
        """Return an average of scalararray at point j - 1/2
        """
        average = lambda xarr:( xarr.shift(y=1) + xarr ) / 2.
        if weights_in is None:
            out = average(scalararray)
        else:
            if weights_out is None:
                weights_out = average(weights_in)
            out = average(scalararray * weights_in) / weights_out
        return out

#- User swapping utilities
    def _weights_for_change_grid_location(self,input=None,output=None,
                                          conserving=None):
        """Return the weights for changing grid location.

        This function is used internally for change_grid_location_*_to_*
        """
        if conserving is 'area':
            weights_in  = self._arrays["cell_area_at_" + input  + "_location"]
            weights_out = self._arrays["cell_area_at_" + output + "_location"]
        elif conserving is 'x_flux':
            weights_in  = self._arrays["cell_y_size_at_" + input  + "_location"]
            weights_out = self._arrays["cell_y_size_at_" + output + "_location"]
        elif conserving is 'y_flux':
            weights_in  = self._arrays["cell_x_size_at_" + input  + "_location"]
            weights_out = self._arrays["cell_x_size_at_" + output + "_location"]

        return weights_in, weights_out

    def change_grid_location_t_to_u(self,scalararray,conserving='area'):
        """Return a xarray corresponding to scalararray averaged at a new
        grid location.

        Parameters
        ----------
        scalararray : xarray.DataArray
            original array to be relocated
        conserving : str
            any of 'area', 'x_flux' or 'y_flux'.
             - 'area' : conserves the area
             - 'x_flux' : conserves the flux in x-direction (eastward)
             - 'y_flux' : conserves the flux in y-direction (northward)
        """
        check_input_array(scalararray,\
                          chunks=self.chunks,grid_location='t',ndims=self.ndims)
        wi, wo = self._weights_for_change_grid_location(input='t',output='u',
                                                        conserving=conserving)
        out = self._to_eastern_grid_location(scalararray,weights_in=wi,
                                                         weights_out=wo)
        return add_extra_attrs_to_dataarray(out,grid_location='u')

    def change_grid_location_u_to_t(self,scalararray,conserving='x_flux'):
        """Return a xarray corresponding to scalararray averaged at a new
        grid location.

        Parameters
        ----------
        scalararray : xarray.DataArray
            original array to be relocated
        conserving : str
            any of 'area', 'x_flux' or 'y_flux'.
             - 'area' : conserves the area
             - 'x_flux' : conserves the flux in x-direction (eastward)
             - 'y_flux' : conserves the flux in y-direction (northward)
        """
        check_input_array(scalararray,\
                          chunks=self.chunks,grid_location='u',ndims=self.ndims)
        wi, wo = self._weights_for_change_grid_location(input='u',output='t',
                                                        conserving=conserving)
        out = self._to_western_grid_location(scalararray,weights_in=wi,
                                                         weights_out=wo)
        return add_extra_attrs_to_dataarray(out,grid_location='t')

    def change_grid_location_t_to_v(self,scalararray,conserving='area'):
        """Return a xarray corresponding to scalararray averaged at a new
        grid location.

        Parameters
        ----------
        scalararray : xarray.DataArray
            original array to be relocated
        conserving : str
            any of 'area', 'x_flux' or 'y_flux'.
             - 'area' : conserves the area
             - 'x_flux' : conserves the flux in x-direction (eastward)
             - 'y_flux' : conserves the flux in y-direction (northward)
        """
        check_input_array(scalararray,\
                          chunks=self.chunks,grid_location='t',ndims=self.ndims)
        wi, wo = self._weights_for_change_grid_location(input='t',output='v',
                                                        conserving=conserving)
        out = self._to_northern_grid_location(scalararray,weights_in=wi,
                                                         weights_out=wo)
        return add_extra_attrs_to_dataarray(out,grid_location='v')

    def change_grid_location_v_to_t(self,scalararray,conserving='y_flux'):
        """Return a xarray corresponding to scalararray averaged at a new
        grid location.

        Parameters
        ----------
        scalararray : xarray.DataArray
            original array to be relocated
        conserving : str
            any of 'area', 'x_flux' or 'y_flux'.
             - 'area' : conserves the area
             - 'x_flux' : conserves the flux in x-direction (eastward)
             - 'y_flux' : conserves the flux in y-direction (northward)
        """
        check_input_array(scalararray,\
                          chunks=self.chunks,grid_location='v',ndims=self.ndims)
        wi, wo = self._weights_for_change_grid_location(input='v',output='t',
                                                        conserving=conserving)
        out = self._to_southern_grid_location(scalararray,weights_in=wi,
                                                         weights_out=wo)
        return add_extra_attrs_to_dataarray(out,grid_location='t')

    def change_grid_location_f_to_u(self,scalararray,conserving='x_flux'):
        """Return a xarray corresponding to scalararray averaged at a new
        grid location.

        Parameters
        ----------
        scalararray : xarray.DataArray
            original array to be relocated
        conserving : str
            any of 'area', 'x_flux' or 'y_flux'.
             - 'area' : conserves the area
             - 'x_flux' : conserves the flux in x-direction (eastward)
             - 'y_flux' : conserves the flux in y-direction (northward)
        """
        check_input_array(scalararray,\
                          chunks=self.chunks,grid_location='f',ndims=self.ndims)
        wi, wo = self._weights_for_change_grid_location(input='f',output='u',
                                                        conserving=conserving)
        out = self._to_southern_grid_location(scalararray,weights_in=wi,
                                                         weights_out=wo)
        return add_extra_attrs_to_dataarray(out,grid_location='u')

    def change_grid_location_f_to_v(self,scalararray,conserving='y_flux'):
        """Return a xarray corresponding to scalararray averaged at a new
        grid location.

        Parameters
        ----------
        scalararray : xarray.DataArray
            original array to be relocated
        conserving : str
            any of 'area', 'x_flux' or 'y_flux'.
             - 'area' : conserves the area
             - 'x_flux' : conserves the flux in x-direction (eastward)
             - 'y_flux' : conserves the flux in y-direction (northward)
        """
        check_input_array(scalararray,\
                          chunks=self.chunks,grid_location='f',ndims=self.ndims)
        wi, wo = self._weights_for_change_grid_location(input='f',output='v',
                                                        conserving=conserving)
        out = self._to_southern_grid_location(scalararray,weights_in=wi,
                                                         weights_out=wo)
        return add_extra_attrs_to_dataarray(out,grid_location='v')

    def change_grid_location_v_to_u(self,scalararray,conserving='area'):
        """Return a xarray corresponding to scalararray averaged at a new
        grid location.

        Parameters
        ----------
        scalararray : xarray.DataArray
            original array to be relocated
        conserving : str
            any of 'area', 'x_flux' or 'y_flux'.
             - 'area' : conserves the area
             - 'x_flux' : conserves the flux in x-direction (eastward)
             - 'y_flux' : conserves the flux in y-direction (northward)
        """
        # first move to t-point
        newarr = self.change_grid_location_v_to_t(scalararray,
                                                  conserving=conserving)
        # then move t v-point
        out = self.change_grid_location_t_to_u(newarr,
                                                  conserving=conserving)
        return out

    def change_grid_location_u_to_v(self,scalararray,conserving='area'):
        """Return a xarray corresponding to scalararray averaged at a new
        grid location.

        Parameters
        ----------
        scalararray : xarray.DataArray
            original array to be relocated
        conserving : str
            any of 'area', 'x_flux' or 'y_flux'.
             - 'area' : conserves the area
             - 'x_flux' : conserves the flux in x-direction (eastward)
             - 'y_flux' : conserves the flux in y-direction (northward)
        """
        # first move to t-point
        newarr = self.change_grid_location_u_to_t(scalararray,
                                                  conserving=conserving)
        # then move t v-point
        out = self.change_grid_location_t_to_v(newarr,
                                                  conserving=conserving)
        return out

#---------------------------- Vector Operators ---------------------------------
    def norm_of_vectorfield(self,vectorfield):
        """Return the norm of a vector field, at t-point.

        So far, only available for vector fields at u,v grid_location

        Parameters
        ----------
        vectorfield : VectorField2d namedtuple
            so far only valid for vectorfield at u,v-points

        Return
        ------
        scalararray : xarray.DataArray
            xarray with a specified grid_location, so far t-point only
        """
        return xu.sqrt(self.scalar_product(vectorfield,vectorfield))

    def scalar_product(self,vectorfield1,vectorfield2):
        """Return the scalar product of two vector fields, at t-point.

        So far, only available for vector fields at u,v grid_location.

        Parameters
        ----------
        vectorfield1 : VectorField2d namedtuple
        vectorfield2 : VectorField2d namedtuple

        Return
        ------
        scalararray : xarray.DataArray

        Methods
        -------
        Multiplies each component independently, relocates each component at
        t grid_location then add the two products.
        """
        check_input_array(vectorfield1.x_component,\
                          chunks=self.chunks,grid_location='u',ndims=self.ndims)
        check_input_array(vectorfield1.y_component,\
                          chunks=self.chunks,grid_location='v',ndims=self.ndims)
        check_input_array(vectorfield2.x_component,\
                          chunks=self.chunks,grid_location='u',ndims=self.ndims)
        check_input_array(vectorfield2.y_component,\
                          chunks=self.chunks,grid_location='v',ndims=self.ndims)

        x_component_u = vectorfield1.x_component * vectorfield2.x_component
        y_component_v = vectorfield1.y_component * vectorfield2.y_component

        x_component_t = self.change_grid_location_u_to_t(x_component_u,
                                                    conserving='area')
        y_component_t = self.change_grid_location_v_to_t(y_component_v,
                                                    conserving='area')
        out = x_component_t + y_component_t
        return out

    def scalar_outer_product(self,scalararray,vectorfield):
        """Return the outer product of a scalar (t location)
        with a two-dimensional vector field (u,v location)

        So far, only available for vector fields at u,v grid_location.

        Parameters
        ----------
        scalararray : xarray.DataArray
        vectorfield : VectorField2d namedtuple
            two-dimensional vector field at u,v grid_location

        Return
        ------
        vectorfield : VectorField2d namedtuple

        Methods
        -------
        Relocates scalararray at u,v grid_location then multiply each component
        of vectorfield by the relocated scalararray.
        """

        #- check input arrays
        check_input_array(scalararray,\
                          chunks=self.chunks,grid_location='t',ndims=self.ndims)
        check_input_array(vectorfield.x_component,\
                          chunks=self.chunks,grid_location='u',ndims=self.ndims)
        check_input_array(vectorfield.y_component,\
                          chunks=self.chunks,grid_location='v',ndims=self.ndims)

        #- relocate scalararray at u,v grid_location
        scalararray_u_location = self.change_grid_location_t_to_u(scalararray,
                                                    conserving='area')
        scalararray_v_location = self.change_grid_location_t_to_v(scalararray,
                                                    conserving='area')

        #- multiplication and creation of VectorFiel2d
        x_component = scalararray_u_location * vectorfield.x_component
        y_component = scalararray_v_location * vectorfield.y_component
        return VectorField2d(x_component,y_component,\
                             x_component_grid_location = 'u',\
                             y_component_grid_location = 'v')

    def vertical_component_of_the_cross_product(self,vectorfield1,vectorfield2):
        """Return the cross product of two vector fields.

        So far, only available for vector fields at u,v grid_location.

        Parameters
        ----------
        vectorfield1 : VectorField2d namedtuple
            two-dimensional vector field at u,v grid_location
        vectorfield2 : VectorField2d namedtuple
            two-dimensional vector field at u,v grid_location

        Return
        ------
        scalararray : xarray.DataArray
            vertical component at t grid location

        Methods
        -------
        Relocates all the components of the VectorFields at t grid_location
        then compute c = v1_x * v2_y - v1_y * v2_x
        """

        #- check input arrays
        check_input_array(vectorfield1.x_component,\
                          chunks=self.chunks,grid_location='u',ndims=self.ndims)
        check_input_array(vectorfield1.y_component,\
                          chunks=self.chunks,grid_location='v',ndims=self.ndims)
        check_input_array(vectorfield2.x_component,\
                          chunks=self.chunks,grid_location='u',ndims=self.ndims)
        check_input_array(vectorfield2.y_component,\
                          chunks=self.chunks,grid_location='v',ndims=self.ndims)

        #- relocate all the arrays at t grid_location
        v1x = vectorfield1.x_component # short cuts
        v1y = vectorfield1.y_component
        v2x = vectorfield2.x_component
        v2y = vectorfield2.y_component

        v1x_at_t_location = self.change_grid_location_u_to_t(v1x,
                                                    conserving='x_flux')
        v1y_at_t_location = self.change_grid_location_v_to_t(v1y,
                                                    conserving='y_flux')
        v2x_at_t_location = self.change_grid_location_u_to_t(v2x,
                                                    conserving='x_flux')
        v2y_at_t_location = self.change_grid_location_v_to_t(v2y,
                                                    conserving='y_flux')

        # compose and output
        cross_product =  v1_x * v2_y - v1_y * v2_x
        return cross_product # the xarray is not finalized (name,grid_location)

#-------------------- Differential Operators------------------------------------

    def horizontal_gradient(self,datastructure):
        """
        Return the horizontal gradient of the input datastructure.

        Parameters
        ----------
        datastructure : xarray.DataArray or VectorField2d

        Return
        ------
        result : VectorField2d or Tensor2d

        Methods
        -------
        calls horizontal_gradient_vector or horizontal_gradient_tensor depending
        on th type of the input datastructure.

        See also:
        --------
        self.horizontal_gradient_vector,  self.horizontal_gradient_tensor

        """
        if is_xarray(datastructure):
            return self.horizontal_gradient_vector(datastructure)
        elif isinstance(datastructure,tuple):
            return self.horizontal_gradient_tensor(datastructure)
        else:
            raise Exception('unrecognized type of datastructure')

    def horizontal_gradient_vector(self,scalararray):
        """
        Return the horizontal gradient of a scalar field defined at t-points.

        Parameters
        ----------
        scalararray : xarray.DataArray
            xarray of a scalar variable at grid_location='t'

        Return
        ------
        vectorfield : VectorField2d namedtuple
            x and y component of the horizontal gradient at u,v-points
        """
        #- check input arrays
        check_input_array(scalararray,\
                          chunks=self.chunks,grid_location='t',ndims=self.ndims)

        #- define each component of the gradient
        gx = _di(scalararray) / self._arrays["cell_x_size_at_u_location"] # e1u
        gy = _dj(scalararray) / self._arrays["cell_y_size_at_v_location"] # e2v

        #- finalize attributes
        gxatts = convert_dataarray_attributes_xderivative(scalararray.attrs,\
                                                          grid_location='u')
        gyatts = convert_dataarray_attributes_yderivative(scalararray.attrs,\
                                                          grid_location='v')

        gx = _finalize_dataarray_attributes(gx,**gxatts)
        gy = _finalize_dataarray_attributes(gy,**gyatts)

        return VectorField2d(gx,gy,\
                             x_component_grid_location = 'u',\
                             y_component_grid_location = 'v')

    def horizontal_gradient_tensor(self,vectorfield):
        """
        Return the horizontal gradient tensor of a two-dimensional vector
        field at u,v locations.

        Namely, with $\mathbf{u} = (u,v)$
        $$
        \nabla \mathbf{u} =
        \begin{pmatrix}
        \partial_x u & \partial_y u \\
        \partial_y v &  \partial_y v
        \end{pmatrix}
        $$

        $ \partial_x u $ is defined at t grid location.
        $ \partial_y u $ is defined at f grid location.
        $ \partial_x v $ is defined at f grid location.
        $ \partial_y v $ is defined at t grid location.


        Parameters
        ----------
        vectorfield : VectorField2d
            namedtuple of a vector field defined at u,v points

        Return
        ------
        gradtensor : Tensor2d
            namedtuple holding the component of the horizontal gradient of
            the vector field at at t and f points.
        """
        #- check input arrays
        check_input_array(vectorfield.x_component,\
                          chunks=self.chunks,grid_location='u',ndims=self.ndims)
        check_input_array(vectorfield.y_component,\
                          chunks=self.chunks,grid_location='v',ndims=self.ndims)

        #- define each component of the gradient tensor
        axx = _di(vectorfield.x_component).shift(x=1) \
              / self._arrays["cell_x_size_at_t_location"]
        axy = _dj(vectorfield.x_component) \
              / self._arrays["cell_y_size_at_f_location"]
        ayx = _di(vectorfield.y_component) \
              / self._arrays["cell_x_size_at_f_location"]
        ayy = _dj(vectorfield.y_component).shift(y=1) \
              / self._arrays["cell_y_size_at_t_location"]

        #- finalize arrays' attributes
        arr_x = vectorfield.x_component
        arr_y = vectorfield.y_component
        axxatts = convert_dataarray_attributes_xderivative(arr_x.attrs,
                                                          grid_location='t')
        axyatts = convert_dataarray_attributes_xderivative(arr_x.attrs,
                                                          grid_location='f')
        ayxatts = convert_dataarray_attributes_xderivative(arr_y.attrs,
                                                          grid_location='f')
        ayyatts = convert_dataarray_attributes_yderivative(arr_y.attrs,
                                                          grid_location='t')

        axx = _finalize_dataarray_attributes(axx,**axxatts)
        axy = _finalize_dataarray_attributes(axy,**axyatts)
        ayx = _finalize_dataarray_attributes(ayx,**ayxatts)
        ayy = _finalize_dataarray_attributes(ayy,**ayyatts)

        return Tensor2d(axx,axy,ayx,ayy,\
                             xx_component_grid_location = 't',\
                             xy_component_grid_location = 'f',\
                             yx_component_grid_location = 'f',\
                             yy_component_grid_location = 't')

    def horizontal_laplacian(self,scalararray):
        """
        Return the horizontal laplacian of a scalar field at t-points.

        Compute the laplacian as the divergence of the gradient.

        Parameters
        ----------
        scalararray : xarray.DataArray
            xarray of a scalar variable defined at grid_location='t'

        Returns
        -------
        scalararray : xarray.DataArray
            xarray of laplacian defined at grid_location='t'
        """
        # check
        check_input_array(scalararray,\
                          chunks=self.chunks,grid_location='t',ndims=self.ndims)
        # define
        lap = self.horizontal_divergence(self.horizontal_gradient(scalararray))
        # finalize
        lapatts = convert_dataarray_attributes_laplacian(scalararray.attrs,
                                                         grid_location='t')
        lap = _finalize_dataarray_attributes(lap,**lapatts)
        return lap

    def vertical_component_of_curl(self,vectorfield):
        """Return the vertical component of the curl of a vector field.

        Parameters
        ----------
        vectorfield : VectorField2d namedtuple

        Returns
        -------
        scalararray: xarray.DataArray
        """

        #- check inpit vector field.
        check_input_array(vectorfield.x_component,\
                          chunks=self.chunks,grid_location='u',ndims=self.ndims)
        check_input_array(vectorfield.y_component,\
                          chunks=self.chunks,grid_location='v',ndims=self.ndims)

        #- define dataarray
        vx = _di(vectorfield.y_component) \
              / self._arrays["cell_x_size_at_f_location"]
        uy = _dj(vectorfield.x_component) \
              / self._arrays["cell_y_size_at_f_location"]
        curl = vx - uy

        curl.name = 'vertical component of the curl'
        return add_extra_attrs_to_dataarray(curl,grid_location='f')

    def horizontal_divergence(self,vectorfield):
        """
        Return the horizontal divergence of a vector field at u,v-points.

        Parameters
        ----------
        vectorfield : VectorField2d namedtuple
           Two-dimensional vector field at u,v-points.

        Returns
        -------
        scalararray: xarray.DataArray
           xarray of divergence at grid_location='t'
        """
        # check
        check_input_array(vectorfield.x_component,\
                          chunks=self.chunks,grid_location='u',ndims=self.ndims)
        check_input_array(vectorfield.y_component,\
                          chunks=self.chunks,grid_location='v',ndims=self.ndims)
        # define
        div  = _di( vectorfield.x_component * self._array_e2u).shift(x=1)
        div += _dj( vectorfield.y_component * self._array_e1v).shift(y=1)
        div /= self._array_e1t * self._array_e2t
        # finalize
        divatts = convert_dataarray_attributes_divergence(\
                    vectorfield.x_component.attrs,vectorfield.y_component.attrs)
        div = _finalize_dataarray_attributes(div,**divatts)
        return div

#
#-------------------- Spatial Integration and averages --------------------------
#

    def integrate_dxdy(self,array,where=None,grid_location=None,normalize=False):
        """Return the horizontal integral of array in regions where
        where is True.

        Parameters
        ----------
        array : xarray.DataArray
            a dataarray with an additonal attribute specifying the grid_location.
            The dimension of array should include 'x' and 'y'.
            The shape of array should match the shape of the grid.
        where: boolean xarray.DataArray
            dataarray with value = True where the integration should be applied.
            The dimension of where should be a subset of the dimension of array.
            For each dimension, the size should be equal to the corresponding
            size of array dataarray.
            if where is None, the function return the integral in all the domain
            defined by the grid object.
        grid_location : str
            string describing the grid location : eg 'u','v','t','f'...
             - if grid_location is not None
                    check compatibility with array.attrs.grid_location
             - if grid_location is None
                    use array.attrs.grid_location by default
        normalize

        Returns
        -------
        integral: xarray.DataArray
            a dataarray with reduced dimension defining the integral of array in
            the region of interest.
        """
        # check grid location
        if grid_location is None:
            if not(isinstance(array,xr.DataArray)):
                raise TypeError('input array should be a xarray.DataArray')
            elif array.attrs.has_key("grid_location"):
                grid_location = array.attrs["grid_location"]
            else:
                raise Exception('grid_location is not known.')
            #except:
            #    raise TypeError('input array should be a xarray.DataArray')

        # check arrays
        check_input_array(array,\
                            chunks=self.chunks,grid_location=grid_location,
                            ndims=self.ndims)
        if where is not None:
            check_input_array(where,\
                              chunks=self.chunks,grid_location=grid_location,
                              ndims=self.ndims)
        else:
            maskname = "sea_binary_mask_at_" + grid_location + "_location"
            where = self._arrays[maskname]

        # actual definition
        idims = ('x','y')
        dxdy = self._arrays['cell_area_at_' + grid_location + '_location']
        array_dxdy = array.where(where.squeeze()) * dxdy
        integral = array_dxdy.sum(dim=idims)

        # normalize if required
        if normalize:
            integral /= dxdy.where(where.squeeze()).sum(dim=idims)

        return integral

    def spatial_average_xy(self,array,where=None,grid_location=None):
        """Return the horizontal average of array in regions where where is True.

        Parameters
        ----------
        array : xarray.DataArray
            a dataarray with an additonal attribute specifying the grid_location.
            The dimension of array should include 'x' and 'y'.
            The shape of array should match the shape of the grid.
        where: boolean xarray.DataArray
            dataarray with value = True where the integration should be applied
            The dimension of where should be a subset of the dimension of array.
            For each dimension, the size should be equal to the corresponding
            size of array dataarray.
            if where is None, the function return the integral in all the domain
            defined by the grid object.
        grid_location : str
            string describing the grid location : eg 'u','v','t','f'...
             - if grid_location is not None
                    check compatibility with array.attrs.grid_location
             - if grid_location is None
                    use array.attrs.grid_location by default

        Returns
        -------
        average: xarray.DataArray
            a dataarray with reduced dimension defining the average of array in
            the region of interest.
        """

        average = self.integrate_dxdy(array,where=where,
                                 grid_location=grid_location,normalize=True)
        return average

#
#--------------- Operators specific to oceanic applications --------------------
#
#   TODO : could move to another module if oocgcm is used for atmospheric models

    def geostrophic_current_from_sea_surface_height(self,sea_surface_height):
        """Return the geostrophic current on u,v-grids.

        Parameters
        ----------
        scalararray : xarray.DataArray
            xarray of sea surface height at grid_location='t'

        Returns
        -------
        vectorfield : VectorField2d namedtuple
           Two-dimensional vector field of geostrophic currents at u,v-points.
        """
        check_input_array(sea_surface_height,\
                          chunks=self.chunks,grid_location='t',ndims=self.ndims)
        gssh = self.horizontal_gradient(sea_surface_height)

        vg = grav \
           * self.change_grid_location_u_to_v(gssh.x_component) \
           / self._arrays["coriolis_parameter_at_v_location"]
        ug = - grav \
           * self.change_grid_location_v_to_u(gssh.y_component) \
           / self._arrays["coriolis_parameter_at_u_location"]

        return VectorField2d(ug,vg,\
                             x_component_grid_location = 'u',\
                             y_component_grid_location = 'v')

    def q_vector_due_to_kinematic_deformation(self,velocity,buoyancy):
        """Return the component of the generalized Q-vector associated
        with kinematic deformation of a two-dimensional velocity field.

        Namely,
        $$
        \mathbf{Q}^g_{kd} =
        -
        \begin{pmatrix}
        \partial_x u \,\partial_x b  + \partial_x v_g \,\partial_y b \\
        \partial_y u \,\partial_x b  + \partial_y v_g \,\partial_y b
        \end{pmatrix}
        $$

        see Hoskins and Bretherthon 1972.

        Parameters
        ----------
        velocity : VectorField2d
            namedtuple of horisontal velocity at u,v grid locations
        buoyancy : xarray.DataArray
            xarray of buoyancy  at grid_location='t'

        Returns
        -------
        vectorfield : VectorField2d
           Two-dimensional vector field of $Q_{kd}$ at u,v-points.
        """
        gradhb = self.horizontal_gradient(buoyancy)  # at u,v location
        gradvel = self.horizontal_gradient(velocity) # at t,f locations

        # use readable notations
        dxb = gradhb.x_component
        dyb = gradhb.y_component
        dxu = gradvel.xx_component
        dyu = gradvel.xy_component
        dxv = gradvel.yx_component
        dyv = gradvel.yy_component

        qxcomp = dxu * dxb + self.change_grid_location_f_to_u(dxv * dyb)
        qycomp = self.change_grid_location_f_to_v(dyu * dxb) + dyv * dyb

        return VectorField2d(qxcomp,qycomp,\
                             x_component_grid_location = 'u',\
                             y_component_grid_location = 'v')

    def frontogenesis_function(self,velocity,buoyancy):
        """Return the component of the generalized Q-vector associated
        with kinematic deformation of a two-dimensional velocity field.

        Namely,
        $$F_s = \mathbf{Q}_{kd} \cdot \nabla_h b$$

        see Hoskins and Bretherthon 1972.

        Parameters
        ----------
        velocityfield : VectorField2d
            namedtuple of horisontal velocity at u,v grid locations
        buoyancy : xarray.DataArray
            xarray of buoyancy  at grid_location='t'

        Returns
        -------
        result : xarray.DataArray
           Frontogenesis function at at t location.
        """
        # TODO : avoid duplication of computation of gradhb
        Qkd = self.q_vector_due_to_kinematic_deformation(velocity,buoyancy)
        gradhb = self.horizontal_gradient(buoyancy)  # at u,v location
        Fs = self.scalar_product(Qkd,gradhb)
        return Fs
