#!/usr/bin/env python
#
"""oocgcm.core.utils
Define various generic utilities tools to be used in several submodules.

"""
import numpy as np
import xarray as xr
import dask.array as da


def add_extra_attrs_to_dataarray(xarr,**extra_kwargs):
    """Update the dictionnary of attributes a xarray dataarray (xarr.attrs).

    Parameters
    ----------
    xarr : xarray.DataArray
        The function will add extra arguments to xarr.attrs
    **extra_kwargs
        not used

    Returns
    ------
    da : xarray.DataArray
    """
    if not(is_xarray(array)):
        raise TypeError('except a xarray.DataArray')
    for kwargs in extra_kwargs:
        xarr.attrs[kwargs] = extra_kwargs[kwargs]
    return xarr


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

def map_function(func,scalararray):
    """Return a xarray dataarray with value func(scalararray.data)

    Parameters
    ----------
    scalararray : xarray.DataArray

    Return
    ------
    out : xarray.DataArray

    Method
    ------
    uses dask map_block
    """
    data = scalararray.data
    coords = scalararray.coords
    dims = scalararray.dims
    if is_daskarray(data):
        _out = data.map_block(func)
    else:
        _out = func(data)
    out = xr.DataArray(_out,coords,dims)
    return da_dj,da_di
