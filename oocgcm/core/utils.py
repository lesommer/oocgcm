#!/usr/bin/env python
#
"""oocgcm.core.utils
Define various generic utilities tools to be used in several submodules.

"""
import numpy as np
import xarray as xr
import dask.array as da

#
#=========================== General purpose ==================================
#

class _SliceGetter(object):
    """Class that returns the slice that is passed to __getitem__.

    Example
    -------
    >>getslice = SliceGetter()
    >>getslice[100:150,300:340]
    """
    def __init__(self):
        pass

    def __getitem__(self, index):
        return index

returnslice = _SliceGetter()

#
#================= Applying numpy functions to dataarray =======================
#

def map_apply(func,scalararray):
    """Return a xarray dataarray with value func(scalararray.data)

    Parameters
    ----------
    func : function
        Any function that works on numpy arrays such that input and output
        arrays have the same shape.
    scalararray : xarray.DataArray

    Returns
    -------
    out : xarray.DataArray

    Methods
    -------
    uses dask map_block without ghost cells (map_overlap)
    """
    data = scalararray.data
    coords = scalararray.coords
    dims = scalararray.dims
    if is_daskarray(data):
        _out = data.map_block(func)
    else:
        _out = func(data)
    out = xr.DataArray(_out,coords,dims)
    return out



#
#============================ Testing types  ==================================
#


def is_numpy(array):
    """Return True if array is a numpy array

    Parameters
    ----------
    array : array-like
       array is either a numpy array, a masked array, a dask array or a xarray.

    Returns
    -------
    test : bool
    """
    test =  bool( isinstance(array,np.ndarray)
                  + isinstance(array,np.ma.masked_array) )
    return test

def is_xarray(array):
    """Return True if array is a xarray.DataArray

    Parameters
    ----------
    array : array-like

    Returns
    -------
    test : bool
    """
    return isinstance(array,xr.DataArray)

def is_daskarray(array):
    """Return True if array is a dask array

    Parameters
    ----------
    array : array-like

    Returns
    -------
    test : bool
    """
    return isinstance(array,da.core.Array)


def has_chunks(array):
    """Return True if array is a xarray or a daskarray with chunks.

    Parameters
    ----------
    array : array-like
       array is either a numpy array, a masked array, a dask array or a xarray.

    Returns
    -------
    test : bool
    """
    if is_xarray(array) or is_daskarray(array):
        return not(array.chunks is None)
    else:
        return False


#
#============================ Testing dataarrays  ==================================
#



def _append_dataarray_extra_attrs(xarr,**extra_kwargs):
    """Update the dictionnary of attributes a xarray dataarray (xarr.attrs).

    Parameters
    ----------
    xarr : xarray.DataArray
        The function will add extra arguments to xarr.attrs
    **extra_kwargs
        not used

    Returns
    -------
    da : xarray.DataArray
    """
    if not(is_xarray(xarr)):
        raise TypeError('except a xarray.DataArray')
    for kwargs in extra_kwargs:
        xarr.attrs[kwargs] = extra_kwargs[kwargs]
    return xarr



def _grid_location_equals(xarr,grid_location=None):
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
    if 'grid_location' in xarr.attrs:
        test *= (xarr.attrs['grid_location']==grid_location)
    return test


def _chunks_are_compatible(chunks1=None,chunks2=None,ndims=None):
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

def _assert_are_compatible_dataarrays(array1,array2):
    """Assert whether two arrays are dataarray with similar dimensions, shapes
    and dask chunk (if relevant).

    Parameters
    ----------
    array1 : array-like
    array2 : array-like

    Returns
    -------
    test : bool
        True if the two arrays are compatible
    """
    assert(is_xarray(array1) and is_xarray(array2))
    assert(array1.dims == array2.dims)
    assert(array1.shape == array2.shape)
    assert(has_chunks(array1) == has_chunks(array2))
    if (has_chunks(array1) and has_chunks(array2)):
        assert_chunks_are_compatible(array1.chunks, array2.chunks)
    return True


def _assert_and_set_grid_location_attribute(xarr,grid_location=None):
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
    if 'grid_location' in xarr.attrs:
        assert ( xarr.attrs['grid_location'] == grid_location )
    else:
        xarr.attrs['grid_location'] = grid_location


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
    if not(is_xarray(xarr)):
        raise TypeError(arrayname + 'is expected to be a xarray.DataArray')
    if not(_chunks_are_compatible(xarr.chunks,chunks,ndims=ndims)):
        raise ChunkError()
    if not(_grid_location_equals(xarr,grid_location)):
        raise GridLocationError()
    return True

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
