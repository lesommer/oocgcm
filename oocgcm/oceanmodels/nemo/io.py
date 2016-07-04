#!/usr/bin/env python
#
"""oocgcm.nemo.io
Define tools that help dealing with the creation of xarray objects from
NEMO netcdf files and creating NEMO-like netcdf files from xarray objects.

"""

from ...core.io import return_xarray_dataset as _return_xarray_dataset
from ...core.io import return_xarray_mfdataset as _return_xarray_mfdataset
from ...core.io import return_xarray_dataarray as _return_xarray_dataarray


def return_xarray_dataset(*args,**kwargs):
    """Wrapper for core.io.return_xarray_dataset.

    Parameters
    ----------
    *args : arguments
    **kwargs : keyword arguments

    Returns
    ------
    ds : xarray.Dataset

    Methods
    ------
    change the name of dimension 'time_counter' in 't'
    change the name of depth dimension to 'depth' and add an attribute
    about the actual depth location
    """
    
    # filters chunks out of the kwargs in order to perform chunking after
    # coordinate renaming
    if 'chunks' in kwargs:
        _chunks = kwargs['chunks']
        kwargs.pop('chunks')
    
    if 'time_counter' in ds.keys():
        ds = ds.rename({'time_counter':'t'})

    if 'deptht' in ds.keys():
        ds = ds.rename({'deptht':'depth'})
        ds['depth_location'] = 't'
    if 'depthu' in ds.keys():
        ds = ds.rename({'depthu':'depth'})        
        ds['depth_location'] = 'u'
    if 'depthv' in ds.keys():
        ds = ds.rename({'depthv':'depth'})
        ds['depth_location'] = 'v'
               
    return ds

def return_xarray_mfdataset(*args,**kwargs):
    """Wrapper for core.io.return_xarray_mfdataset.

    Parameters
    ----------
    *args : arguments
    **kwargs : keyword arguments

    Returns
    ------
    ds : xarray.Dataset

    Methods
    ------
    change the name of dimension 'time_counter' in 't'
    change the name of depth dimension to 'depth' and add an attribute
    about the actual depth location
    """
    
    # filters chunks out of the kwargs in order to perform chunking after
    # coordinate renaming
    if 'chunks' in kwargs:
        _chunks = kwargs['chunks']
        kwargs.pop('chunks')
    
    ds = _return_xarray_mfdataset(*args,**kwargs)
    
    if 'time_counter' in ds.keys():
        ds = ds.rename({'time_counter':'t'})

    if 'deptht' in ds.keys():
        ds = ds.rename({'deptht':'depth'})
        ds['depth_location'] = 't'
    if 'depthu' in ds.keys():
        ds = ds.rename({'depthu':'depth'})        
        ds['depth_location'] = 'u'
    if 'depthv' in ds.keys():
        ds = ds.rename({'depthv':'depth'})
        ds['depth_location'] = 'v'

    ds = ds.chunk(_chunks)
    
    return ds

def return_xarray_dataarray(*args,**kwargs):
    """Wrapper for core.io.return_xarray_dataarray.

    Parameters
    ----------
    *args : arguments
    **kwargs : keyword arguments

    Returns
    -------
    da : xarray.DataArray

    Methods
    -------
    change the name of dimension 'time_counter' in 't'
    """
    _da = _return_xarray_dataarray(*args,**kwargs)
    if 'time_counter' in _da.coords.keys():
       da = _da.rename({'time_counter':'t'})
    else:
       da = _da
    return da
