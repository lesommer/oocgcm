#!/usr/bin/env python
#
"""oocgcm.core.utils
Define various generic utilities tools to be used in several submodules.

"""
import xarray as xr

#
#--------------------- Tools for xarray objects ------------------------------
#


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
    for kwargs in extra_kwargs:
        xarr.attrs[kwargs] = extra_kwargs[kwargs]
    return xarr
