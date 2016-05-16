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


from .utils import is_numpy,has_chunks,add_extra_attrs_to_dataarray

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
        gy,gx = np.gradient(data)
    else:
        x_derivative = lambda arr:np.gradient(arr,axis=-1)
        y_derivative = lambda arr:np.gradient(arr,axis=-2)
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
    _required_arrays = [\
        "sea_binary_mask_at_t_location",\
        "sea_binary_mask_at_u_location",\
        "sea_binary_mask_at_v_location",\
        "sea_binary_mask_at_f_location",\
        "projection_x_coordinate_at_t_location",\
        "projection_y_coordinate_at_t_location",\
        "cell_x_size_at_t_location",\
        "cell_x_size_at_u_location",\
        "cell_x_size_at_v_location",\
        "cell_y_size_at_t_location",\
        "cell_y_size_at_u_location",\
        "cell_y_size_at_v_location",\
        ]

    def __init__(self,variables=None,parameters=None):
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
        self.arrays = variables
        self.parameters = parameters
        self._define_aliases_for_arrays()
        self.chunks = self.arrays["sea_binary_mask_at_t_location"].chunks
        # TODO : not clear whether to store the priori description of chunks
        #        (dictionnary) or the posteriori value (tuple of tuples).
        self.shape  = self.arrays["sea_binary_mask_at_t_location"].shape
        self.dims   = self.arrays["sea_binary_mask_at_t_location"].dims
        self.ndims = len(self.dims)
        self._define_area_of_grid_cells()

#--------------------- Extra xarrays for the grid ------------------------------

    def _define_area_of_grid_cells(self):
        """Define arrays of area at u,v,t,f grid_location.

        This is only a definition, the computation is performed only if used.
        """
        for gloc in ['t','u','v','f']
        self.arrays["cell_area_at_" + gloc + "_location"] = \
                  self.arrays["cell_x_size_at_" + gloc + "_location"] \
                * self.arrays["cell_y_size_at_" + gloc + "_location"]


#--------------------- Aliases for DataArrays ----------------------------------
    def _define_aliases_for_arrays(self):
        """Define alias for frequently used variables.

        The following shortcuts mostly follows nemo name conventions.
        This could evolve in the future and should therefore not be used in
        external libraries.
        """
        # coordinates
        self._array_navlon = self.arrays["projection_x_coordinate_at_t_location"]
        self._array_navlat = self.arrays["projection_y_coordinate_at_t_location"]
        # metrics
        self._array_e1t = self.arrays["cell_x_size_at_t_location"]
        self._array_e1u = self.arrays["cell_x_size_at_u_location"]
        self._array_e1v = self.arrays["cell_x_size_at_v_location"]
        self._array_e2t = self.arrays["cell_y_size_at_t_location"]
        self._array_e2u = self.arrays["cell_y_size_at_u_location"]
        self._array_e2v = self.arrays["cell_y_size_at_v_location"]

        # masks
        self._array_tmask = self.arrays["sea_binary_mask_at_t_location"]
        self._array_umask = self.arrays["sea_binary_mask_at_u_location"]
        self._array_vmask = self.arrays["sea_binary_mask_at_v_location"]
        self._array_fmask = self.arrays["sea_binary_mask_at_f_location"]

#--------------------- Chunking and Slicing ------------------------------------
    def rechunk(self,chunks=None):
        """Rechunk all the variables defining the grid.

        Parameters
        ----------
        chunks : dict-like
            dictionnary of sizes of chunk along xarray dimensions.

        Example
        -------
        >>> xr_chunk = {'x':200,'y':200}
        >>> grd = generic_2d_grid(...)
        >>> grd.rechunk(xr_chunk)

        """
        for dataname in self.arrays:
            data = self.arrays[dataname]
            if isinstance(data, xr.DataArray):
                self.arrays[dataname] = data.chunk(chunks)
        self.chunks = self._array_tmask.chunks

    def __getitem__(self,item):
        """Return a grid object restricted to a subdomain.

        Use with caution, this functionnality depends on the order of the
        dimensions in the netcdf files.

        Parameters
        ----------
        item : slice
            slice that restrict the grid to a subdomain.

        Returns
        -------
        newgrd : generic_2d_grid
            a new grid object corresponding to the restructed region.

        Example
        -------
        >>> grd = generic_2d_grid(...)
        >>> new_grd = grd[100:200,300:500]

        """
        sliced_arrays = {}
        for dataname in self.arrays:
            sliced_arrays[dataname] = self.arrays[dataname][item]
        return generic_2d_grid(variables=sliced_arrays,\
                             parameters=self.parameters)

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

    def _to_western_grid_location(self,scalararray,
                                  weights_in=None,weights_out=None):
        """Return an average of scalararray at point i - 1/2
        """
        average = lambda xarr:( xarr.shift(x=-1) + xarr ) / 2.
        if weights_in is None:
            out = average(scalararray)
        else:
            if weights_out is None:
                weights_out = average(weights_in)
            out = average(scalararray * weights_in) / weights_out

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

    def _to_southern_grid_location(self,scalararray,
                                  weights_in=None,weights_out=None):
        """Return an average of scalararray at point j - 1/2
        """
        average = lambda xarr:( xarr.shift(y=-1) + xarr ) / 2.
        if weights_in is None:
            out = average(scalararray)
        else:
            if weights_out is None:
                weights_out = average(weights_in)
            out = average(scalararray * weights_in) / weights_out

#- User swapping utilities
    def _weights_for_change_grid_location(from=None,to=None,conserving=None):
        """Return the weights for changing grid location.

        This function is used internally for change_grid_location_*_to_*
        """
        if conserving is 'area':
            weights_in  = self.arrays["cell_area_at_" + from + "_location"]
            weights_out = self.arrays["cell_area_at_" + to   + "_location"]
        elif conserving is 'x_flux':
            weights_in  = self.arrays["cell_y_size_at_" + from + "_location"]
            weights_out = self.arrays["cell_y_size_at_" + to   + "_location"]
        elif conserving is 'y_flux':
            weights_in  = self.arrays["cell_x_size_at_" + from + "_location"]
            weights_out = self.arrays["cell_x_size_at_" + to   + "_location"]

        return weights_in, weights_out

    def change_grid_location_t_to_u(scalararray,conserving='area'):
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
        wi, wo = self._weights_for_change_grid_location(from='t',to='u'
                                                        conserving=conserving)
        out = self._to_eastern_grid_location(scalararray,weights_in=wi
                                                         weights_out=wo)
        return add_extra_attrs_to_dataarray(out,grid_location='u')

    def change_grid_location_u_to_t(scalararray,conserving='x_flux'):
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
        wi, wo = self._weights_for_change_grid_location(from='u',to='t'
                                                        conserving=conserving)
        out = self._to_western_grid_location(scalararray,weights_in=wi
                                                         weights_out=wo)
        return add_extra_attrs_to_dataarray(out,grid_location='t')

    def change_grid_location_t_to_v(scalararray,conserving='area'):
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
        wi, wo = self._weights_for_change_grid_location(from='t',to='v'
                                                        conserving=conserving)
        out = self._to_nortern_grid_location(scalararray,weights_in=wi
                                                         weights_out=wo)
        return add_extra_attrs_to_dataarray(out,grid_location='v')

    def change_grid_location_v_to_t(scalararray,conserving='y_flux'):
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
        wi, wo = self._weights_for_change_grid_location(from='v',to='t'
                                                        conserving=conserving)
        out = self._to_southern_grid_location(scalararray,weights_in=wi
                                                         weights_out=wo)
        return add_extra_attrs_to_dataarray(out,grid_location='t')

#---------------------------- Vector Operators ---------------------------------
    def norm_of_vectorfield(self,vectorfield):
        """Return the norm of a vector field, at t-point.

        Not implemented yet.

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

        Not implemented yet.

        Parameters
        ----------
        vectorfield1 : VectorField2d namedtuple
        vectorfield2 : VectorField2d namedtuple

        Return
        ------
        scalararray : xarray.DataArray
        """
        pass

    def scalar_outer_product(self,scalararray,vectorfield):
        """Return the outer product of a scalar with a vector field, at t-point.

        Not implemented yet.

        Parameters
        ----------
        scalararray : xarray.DataArray
        vectorfield : VectorField2d namedtuple

        Return
        ------
        vectorfield : VectorField2d namedtuple
        """
        pass

    def vertical_component_of_the_cross_product(self,vectorfield1,vectorfield2):
        """Return the cross product of two vector fields.

        Not implemented yet.

        Parameters
        ----------
        vectorfield1 : VectorField2d namedtuple
        vectorfield2 : VectorField2d namedtuple

        Return
        ------
        vectorfield : VectorField2d namedtuple
        """
        pass


#-------------------- Differential Operators------------------------------------
    def horizontal_gradient(self,scalararray):
        """
        Return the horizontal gradient of a scalar field defined at t-points.

        Parameters
        ----------
        scalararray : xarray.DataArray
            xarray of a variable at grid_location='t'

        Return
        ------
        vectorfield : VectorField2d namedtuple
            x and y component of the horizontal gradient at u,u-points
        """
        # check
        check_input_array(scalararray,\
                          chunks=self.chunks,grid_location='t',ndims=self.ndims)
        # define
        gx = _di(scalararray) / self._array_e1u
        gy = _dj(scalararray) / self._array_e2v
        # finalize attributes
        gxatts = convert_dataarray_attributes_xderivative(scalararray.attrs,\
                                                          grid_location='u')
        gyatts = convert_dataarray_attributes_yderivative(scalararray.attrs,\
                                                          grid_location='v')
        #
        gx = _finalize_dataarray_attributes(gx,**gxatts)
        gy = _finalize_dataarray_attributes(gy,**gyatts)
        #
        return VectorField2d(gx,gy,\
                             x_component_grid_location = 'u',\
                             y_component_grid_location = 'v')

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

        Not implemented yet.

        Parameters
        ----------
        vectorfield : VectorField2d namedtuple

        Returns
        -------
        scalararray: xarray.DataArray
        """
        pass

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
        div  = _di( vectorfield.x_component / self._array_e2u)
        div += _dj( vectorfield.y_component / self._array_e1v)
        div /= self._array_e1t * self._array_e2t
        # finalize
        divatts = convert_dataarray_attributes_divergence(\
                    vectorfield.x_component.attrs,vectorfield.y_component.attrs)
        div = _finalize_dataarray_attributes(div,**divatts)
        return div

#
#--------------- Operators specific to oceanic applications --------------------
#
#   TODO : could move to a subclass if oocgcm is used for atmospheric models

    def geostrophic_current_from_sea_surface_height(self,ssh):
        """Return the geostrophic current on u,v-grids.

        Not implemented yet.

        Parameters
        ----------
        scalararray : xarray.DataArray
            xarray of sea surface height at grid_location='t'

        Returns
        -------
        vectorfield : VectorField2d namedtuple
           Two-dimensional vector field of geostrophic currents at u,v-points.
        """
        pass
