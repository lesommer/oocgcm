#!/usr/bin/env python

import math
import xarray.ufuncs as xu

from ..core.utils import is_numpy,is_xarray
from .mathematicalparameters import deg2rad

# Physical parameters
grav = 9.81                  # acceleration due to gravity (m.s-2)
omega = 7.292115083046061e-5 # earth rotation rate (s-1)
earthrad = 6371229            # mean earth radius (m)


# Functions
def coriolis_parameter(latitudes):
    """Return Coriolis parameter for a given latitude.

    Parameters
    ----------
    latitudes : float or array-like
        latitudes can be a float, a numpy array or a xarray.

    Returns
    -------
    corio : same type as input
    """
    if isinstance(latitudes,float):
        sin = math.sin
    elif is_numpy(latitudes):
        sin = np.sin
    elif is_xarray(latitudes):
        sin = xu.sin
    corio = 2. * omega * math.sin(latitudes * deg2rad)
    return corio

def beta_parameter(latitudes):
    """Return planetary beta parameter.

    Parameters
    ----------
    latitudes : float or array-like
        latitudes can be a float, a numpy array or a xarray.

    Returns
    -------
    corio : same type as input
    """
    if isinstance(latitudes,float):
        cos = math.cos
    elif is_numpy(latitudes):
        cos = np.cos
    elif is_xarray(latitudes):
        cos = xu.cos
    beta = 2. * omega * cos(lat * deg2rad) / earthrad
    return beta
