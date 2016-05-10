#!/usr/bin/env python
#

import xarray as xr


def return_xarray_dataset(filename,chunks=None):
    """Return an xarray dataset corresponding to filename.

    Parameters
    ----------
    filename : str
        path to the netcdf file from which to create a xarray dataset
    chunks : dict-like
        dictionnary of sizes of chunk for creating xarray.Dataset.

    Returns
    ------
    ds : xarray.Dataset
    """
    return xr.open_dataset(filename,chunks=chunks,lock=False)

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
    ------
    da : xarray.DataArray
    """
    ds = return_xarray_dataset(filename,chunks=chunks)
    dataarray = ds[varname]
    for kwargs in extra_kwargs:
        dataarray.attrs[kwargs] = extra_kwargs[kwargs]
    return dataarray
