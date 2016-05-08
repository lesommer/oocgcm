#!/usr/bin/env python
#

import xarray as xr


def return_xarray_dataset(filename,chunks=None):
    """Return an xarray dataset corresponding to filename.
    """
    return xr.open_dataset(filename,chunks=chunks,lock=False)

def return_xarray_dataarray(filename,varname,chunks=None,**extra_kwargs):
    """Return a xarray dataarray corresponding to varname in filename.
    """
    ds = return_xarray_dataset(filename,chunks=chunks)
    dataarray = ds[varname]
    for kwargs in extra_kwargs:
        dataarray.attrs[kwargs] = extra_kwargs[kwargs]
    return dataarray
