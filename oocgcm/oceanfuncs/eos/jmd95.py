#!/usr/bin/env python
#
"""oocgcm.oceanfuncs.eos.jmd95
Equation of state of sea water and related quantities.

This module uses the formulas from Jackett and McDougall (1995) ([1]_).

Potential temperature is the 0-bar reference.


Currently, functions are split into two families: one dealing with
numpy arrays only (if useful for optimisation), the other dealing
with  xarray dataarrays as input (if no optimisation needed).

The list is as follows:

- Numpy only:
_density: density from theta0,s,p                        

- Xarray datarray input
density_meters: density from theta0,s,z with pressure-depth approximation
                Numpy calculation and output
_density_xr: density from theta0,s,p
             Xarray only
density_meters_xr: density from theta0,s,z with pressure-depth approximation
                   Xarray only


Ckeck:
>>> density(40,40,10000)
    1060.93299  


References
----------

  .. [1] Jackett, D.R., Mcdougall, T.J., 1995. Minimal Adjustment of 
     Hydrographic Profiles to Achieve Static Stability. J. Atmos. 
     Oceanic Technol. 12, 381-389.

"""

import numpy as np
from numba import jit

#from ...core.utils import map_apply, _assert_are_compatible_dataarrays
from ...core.utils import *
from matplotlib.pyplot import xkcd
#from build.lib.oocgcm.parameters.physicalparameters import grav


#------------------ Numpy only ---------------------------



def _density(theta0,S,P):
    """In-situ density (volumic mass)

    In situ density is computed directly as a function of
    potential temperature relative to the surface.

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    S : numpy.array
        salinity
    P : numpy.array
        pressure (dB)

    Notes 
    -----
    We use  Jackett and McDougall (1995)'s [1]_ equation of state.
    the in situ density is computed directly as a function of
    potential temperature relative to the surface (theta0), salt 
    and pressure

    with:

    - pressure                      p        decibars

    - potential temperature         theta0        deg celsius

    - salinity                      s        psu
    

    """
    
    zsr=np.sqrt(np.abs(S))
    zt=theta0
    zs=S
    zh=P
     
    # compute volumic mass pure water at atm pressure
    zr1= ( ( ( ( 6.536332e-9*zt-1.120083e-6 )*zt+1.001685e-4)*zt
             -9.095290e-3 )*zt+6.793952e-2 )*zt+999.842594
    # seawater volumic mass atm pressure
    zr2= ( ( ( 5.3875e-9*zt-8.2467e-7 ) *zt+7.6438e-5 ) *zt
           -4.0899e-3 ) *zt+0.824493
    zr3= ( -1.6546e-6*zt+1.0227e-4 ) *zt-5.72466e-3
    zr4= 4.8314e-4

    # potential volumic mass (reference to the surface)
    zrhop= ( zr4*zs + zr3*zsr + zr2 ) *zs + zr1

    # add the compression terms
    ze = ( -3.508914e-8*zt-1.248266e-8 ) *zt-2.595994e-6
    zbw= (  1.296821e-6*zt-5.782165e-9 ) *zt+1.045941e-4
    zb = zbw + ze * zs

    zd = -2.042967e-2
    zc =   (-7.267926e-5*zt+2.598241e-3 ) *zt+0.1571896
    zaw= ( ( 5.939910e-6*zt+2.512549e-3 ) *zt-0.1028859 ) *zt - 4.721788
    za = ( zd*zsr + zc ) *zs + zaw

    zb1=   (-0.1909078*zt+7.390729 ) *zt-55.87545
    za1= ( ( 2.326469e-3*zt+1.553190)*zt-65.00517 ) *zt+1044.077
    zkw= ( ( (-1.361629e-4*zt-1.852732e-2 ) *zt-30.41638 )
           *zt + 2098.925 ) *zt+190925.6
    zk0= ( zb1*zsr + za1 )*zs + zkw

    prd=  zrhop / (  1.0 - zh / ( zk0 - zh * ( za - zh * zb )))

    return prd





#------------------ Wrapped functions for dataarrays ---------------------------



def density_meters(theta0,s,z):
    """In-situ density (volumic mass) using depth value
    in meters. Numpy transposition.

    In situ density is computed directly as a function of
    potential temperature relative to the surface.

    Parameters
    ----------
    theta0 : xarray datarray
        potential temperature (degC)
    S : xarray datarray
        salinity (psu)
    z : xarray datarray
        depth (m)

    Returns
    -------
    rho : numpy.array
       density (volumic mass)

    Notes 
    -----
   We use  Jackett and McDougall (1995)'s [1]_ equation of state.
   the in situ density is computed directly as a function of
   potential temperature relative to the surface (theta0), salt 
   and pressure (assuming no pressure variation
   along geopotential surfaces, i.e. the pressure p in decibars
   is approximated by the depth in meters.

    with:

    - depth                         z        meters

    - temperature                   theta0        deg celsius

    - salinity                      s        psu


    """

    theta0=theta0.to_masked_array()
    s=s.to_masked_array()    
    z=z.to_masked_array()    

    # caution: depth (m) replaces pressure (db)        
    rho=_density(theta0,s,z)
                    
    return rho

    
def _density_xr(theta0,S,Z):
    """In-situ density (volumic mass)

    In situ density is computed directly as a function of
    potential temperature relative to the surface.

    Parameters
    ----------
    theta0 : xarray dataarray
        potential temperature
    S : xarray dataarray
        salinity
    Z : xarray dataarray
        pressure (dB)

    Notes 
    -----
    We use  Jackett and McDougall (1995)'s [1]_ equation of state.
    the in situ density is computed directly as a function of
    potential temperature relative to the surface (theta0), salt 
    and pressure (assuming no pressure variation
    along geopotential surfaces, i.e. the pressure p in decibars
    is approximated by the depth in meters.

    with:

    - pressure                      p        decibars

    - potential temperature         theta0        deg celsius

    - salinity                      s        psu
    

    """
    
    zsr=np.sqrt(np.abs(S))
    zt=theta0
    zs=S
    zh=Z
     
    # compute volumic mass pure water at atm pressure
    zr1= ( ( ( ( 6.536332e-9*zt-1.120083e-6 )*zt+1.001685e-4)*zt
             -9.095290e-3 )*zt+6.793952e-2 )*zt+999.842594
    # seawater volumic mass atm pressure
    zr2= ( ( ( 5.3875e-9*zt-8.2467e-7 ) *zt+7.6438e-5 ) *zt
           -4.0899e-3 ) *zt+0.824493
    zr3= ( -1.6546e-6*zt+1.0227e-4 ) *zt-5.72466e-3
    zr4= 4.8314e-4

    # potential volumic mass (reference to the surface)
    zrhop= ( zr4*zs + zr3*zsr + zr2 ) *zs + zr1

    # add the compression terms
    ze = ( -3.508914e-8*zt-1.248266e-8 ) *zt-2.595994e-6
    zbw= (  1.296821e-6*zt-5.782165e-9 ) *zt+1.045941e-4
    zb = zbw + ze * zs

    zd = -2.042967e-2
    zc =   (-7.267926e-5*zt+2.598241e-3 ) *zt+0.1571896
    zaw= ( ( 5.939910e-6*zt+2.512549e-3 ) *zt-0.1028859 ) *zt - 4.721788
    za = ( zd*zsr + zc ) *zs + zaw

    zb1=   (-0.1909078*zt+7.390729 ) *zt-55.87545
    za1= ( ( 2.326469e-3*zt+1.553190)*zt-65.00517 ) *zt+1044.077
    zkw= ( ( (-1.361629e-4*zt-1.852732e-2 ) *zt-30.41638 )
           *zt + 2098.925 ) *zt+190925.6
    zk0= ( zb1*zsr + za1 )*zs + zkw

    prd=  zrhop / (  1.0 - zh / ( zk0 - zh * ( za - zh * zb )))

    return prd


def density_meters_xr(theta0,s,z):
    """In-situ density (volumic mass) using depth value
    in meters.

    Parameters
    ----------
    theta0 : xarray dataarray
        potential temperature
    s : xarray dataarray
        salinity
    z : xarray dataarray
        depth (m)

    Notes 
    -----
    We use  Jackett and McDougall (1995)'s [1]_ equation of state.
    the in situ density is computed directly as a function of
    potential temperature relative to the surface (theta0), salt 
    and pressure (assuming no pressure variation
    along geopotential surfaces, i.e. the pressure p in decibars
    is approximated by the depth in meters.

    with:

    - depth                         z        meters

    - potential temperature                   theta0        deg celsius

    - salinity                      s        psu
    

    """

    
#    theta=_theta_xr(t,s,z)

    # caution: depth (m) replaces pressure (db)        
    rho=_density_xr(theta0,s,z)
                    
    return rho
