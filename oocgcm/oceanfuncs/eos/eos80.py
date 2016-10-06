#!/usr/bin/env python
#
"""oocgcm.oceanfuncs.eos.eos80
Equation of state of sea water and related quantities.

This module uses the formulas from Unesco’s joint panel on oceanographic 
tables and standards, UNESCO 1981 and UNESCO 1983 (EOS-80).

"""



#--------------------------------------------------------------------------


def _theta(t,s,p):
    """Return potential temperature (degC) at 0-bar level using pressure in
    db.

    Parameters
    ----------
    t : numpy.array
        temperature (degC)
    s : numpy.array
        salinity (PSU)
    p : numpy.array
        pressure (db)

    Returns
    -------
    theta : numpy.array
        potential temperature

    Notes
    -----
    The calculation follows Bryden (1973) for the adiabatic lapse rate and
    Fofonoff (1977) for the integration to a reference level (i.e. a 
    1-iterate Runge-Kutta 4 scheme)


    References
    ----------

    """

    r2=np.sqrt(2.)

    # Adiabatic lapse rate (degC/db)
    g=lambda t,s,p: _gamma_b73_2(t,s,p)
    
    # Fofonoff and Millard (1983)

    pr=0.
    h=pr-p
    xk=h*g(t,s,p)
    t=t+0.5*xk
    q=xk
    p=p+0.5*h
    xk=h*g(t,s,p)
    t=t+0.29289322*(xk-q)
    q=0.58578644*xk+0.121320344*q
    xk=h*g(t,s,p)
    t=t+1.707106781*(xk-q)
    q=3.424213562*xk-4.121320344*q
    p=p+0.5*h
    xk=h*g(t,s,p)
    theta=t+(xk-2.0*q)/6.0

    
    return theta


def _gamma_b73_2_xr(t,s,p):
    """Adiabatic lapse rate (deg C/dBar).
 
    Adiabatic lapse rate (deg C/dBar) from salinity (psu), 
    temperature (deg C) and pressure (dbar)

    Parameters
    ----------
    t : xarray dataarray
        temperature (deg C)
    s : xarray dataarray
        salinity (psu)
    p : xarray dataarray
        pressure (dbar)


    Notes 
    -----
    
    The script is drawn from Fofonoff and Millard (1983).
    The calculation follows Bryden (1973) polynomial.

 
    References
    ----------


    """ 
    #
    ds = s - 35.0
    atg = ((-2.1687e-16 * t + 1.8676e-14) * t - 4.6206e-13) * p * p
    atg+= (2.7759e-12 * t - 1.1351e-10 ) * ds * p
    atg+= (((-5.4481e-14 * t + 8.7330e-12) * t - 6.7795e-10) * t + 1.8741e-8) * p
    atg+= (-4.2393e-8 * t + 1.8932e-6 ) * ds
    atg+= ((6.6228e-10 * t - 6.8360e-8) * t + 8.5258e-6) * t + 3.5803e-5
    
    
    return atg


#------------------ Wrapped functions for dataarrays ---------------------------


def _temperature_xr(theta,s,p):
    """Return temperature (degC) from potential temperature
    at 0-bar level using pressure in db.

    Parameters
    ----------
    theta : xarray dataarray
        potential temperature (degC)
    s : xarray dataarray
        salinity (PSU)
    p : xarray dataarray
        pressure (db)

    Returns
    -------
    temp : xarray dataarray
        temperature

    Notes
    -----
    The calculation follows Bryden (1973) for the adiabatic lapse rate and
    Fofonoff (1977) for the integration to a reference level (i.e. a 
    1-iterate Runge-Kutta 4 scheme).


    References
    ----------

    """

    # Adiabatic lapse rate (degC/db)
    g=lambda t,s,p:  _gamma_b73_2_xr(t,s,p)

    # inversion of Fofonoff and Millard (1983)

    pr=0.
    h=-(pr-p)
    xk=h*g(theta,s,p)
    theta=theta+0.5*xk
    q=xk
    p=p+0.5*h
    xk=h*g(theta,s,p)
    theta=theta+0.29289322*(xk-q)
    q=0.58578644*xk+0.121320344*q
    xk=h*g(theta,s,p)
    theta=theta+1.707106781*(xk-q)
    q=3.424213562*xk-4.121320344*q
    p=p+0.5*h
    xk=h*g(theta,s,p)
    temp=theta+(xk-2.0*q)/6.0
    
    return temp


def _theta_xr(t,s,p):
    """Return potential temperature (degC) at 0-bar level using pressure in
    db.

    Parameters
    ----------
    t : xarray dataarray
        temperature (degC)
    s : xarray dataarray
        salinity (PSU)
    p : xarray dataarray
        pressure (db)

    Returns
    -------
    theta : xarray dataarray
        potential temperature

    Notes
    -----
    The calculation follows Bryden (1973) for the adiabatic lapse rate and
    Fofonoff (1977) for the integration to a reference level (i.e. a 
    1-iterate Runge-Kutta 4 scheme).


    References
    ----------

    """

    r2=uf.sqrt(2.)

    # Adiabatic lapse rate (degC/db)
    g=lambda t,s,p:  _gamma_b73_2_xr(t,s,p)

    # Fofonoff and Millard (1983)

    pr=0.
    h=pr-p
    xk=h*g(t,s,p)
    t=t+0.5*xk
    q=xk
    p=p+0.5*h
    xk=h*g(t,s,p)
    t=t+0.29289322*(xk-q)
    q=0.58578644*xk+0.121320344*q
    xk=h*g(t,s,p)
    t=t+1.707106781*(xk-q)
    q=3.424213562*xk-4.121320344*q
    p=p+0.5*h
    xk=h*g(t,s,p)
    theta=t+(xk-2.0*q)/6.0
    
    
    return theta


def _gamma_b73_2_xr(t,s,p):
    """Adiabatic lapse rate (deg C/dBar).
 
    Adiabatic lapse rate (deg C/dBar) from salinity (psu), 
    temperature (deg C) and pressure (dbar)

    Parameters
    ----------
    t : xarray dataarray
        temperature (deg C)
    s : xarray dataarray
        salinity (psu)
    p : xarray dataarray
        pressure (dbar)


    Notes 
    -----
    
    The script is drawn from Fofonoff and Millard (1983).
    The calculation follows Bryden (1973) polynomial.

 
    References
    ----------


    """ 
    #
    ds = s - 35.0
    atg = ((-2.1687e-16 * t + 1.8676e-14) * t - 4.6206e-13) * p * p
    atg+= (2.7759e-12 * t - 1.1351e-10 ) * ds * p
    atg+= (((-5.4481e-14 * t + 8.7330e-12) * t - 6.7795e-10) * t + 1.8741e-8) * p
    atg+= (-4.2393e-8 * t + 1.8932e-6 ) * ds
    atg+= ((6.6228e-10 * t - 6.8360e-8) * t + 8.5258e-6) * t + 3.5803e-5
    
    
    return atg
