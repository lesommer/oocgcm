#!/usr/bin/env python
#
"""oocgcm.core.io
Define tools that help dealing with the creation of xarray objects from
netcdf files and creating netcdf files from xarray objects.

"""
import xarray as xr

# TODO : some of these methods might move to oocgcm.oceanmodels.nemo.io

#
#--------------------- Creation of xarray objects ------------------------------
#

def return_xarray_dataset(filename,chunks=None,**kwargs):
    """Return an xarray dataset corresponding to filename.

    Parameters
    ----------
    filename : str
        path to the netcdf file from which to create a xarray dataset
    chunks : dict-like
        dictionnary of sizes of chunk for creating xarray.Dataset.

    Returns
    -------
    ds : xarray.Dataset
    """
    return xr.open_dataset(filename,chunks=chunks,**kwargs)

def return_xarray_mfdataset(filename,chunks=None,**kwargs):
    """Return an xarray dataset corresponding to filename which may include
    wildcards (e.g. file_*.nc).

    Parameters
    ----------
    filename : str
        path to a netcdf file or several netcdf files from which to create a
        xarray dataset
    chunks : dict-like
        dictionnary of sizes of chunk for creating xarray.Dataset.

    Returns
    ------
    ds : xarray.Dataset
    """
    if 'autoclose' not in kwargs:
        # fixes OS error arising from too many files open
        # see xarray pull request #1198
        kwargs['autoclose']=True
    if 'compat' not in kwargs:
        # files are usually aligned
        kwargs['compat']='equals'
    return xr.open_mfdataset(filename,chunks=chunks,**kwargs)

def return_xarray_dataarray(filename,varname,chunks=None,**extra_kwargs):
    """Return a xarray dataarray corresponding to varname in filename.

    Parameters
    ----------
    filename : str
        path to the netcdf file from which to create a xarray.DataArray
    varname : str
        name of the variable from which to create a xarray.DataArray
    chunks : dict-like
        dictionnary of sizes of chunk for creating a xarray.DataArray.
    **extra_kwargs
        not used

    Returns
    -------
    da : xarray.DataArray
    """
    ds = return_xarray_dataset(filename,chunks=chunks)
    dataarray = ds[varname]
    for kwargs in extra_kwargs:
        dataarray.attrs[kwargs] = extra_kwargs[kwargs]
    return dataarray


#
#--------------------- Creation of netcdf files --------------------------------
#
