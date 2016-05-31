#!/usr/bin/env python
#
"""oocgcm.oceanfuncs.eos.misc
Equation of state related functions that are not attached to a particular
equation of state of sea water.

"""

import numpy as np
from numba import jit

from ...core.utils import map_apply, _assert_are_compatible_dataarrays

#---------------------- Functions for numpy arrays ----------------------------


_spice_coefs = \
    np.array([
        [ 0,            7.7442e-001,   -5.85e-003,   -9.84e-004,  -2.06e-004 ],
        [5.1655e-002,    2.034e-003,  -2.742e-004,    -8.5e-006,    1.36e-005],
        [6.64783e-003, -2.4681e-004,  -1.428e-005,   3.337e-005,  7.894e-006 ],
        [-5.4023e-005,   7.326e-006,  7.0036e-006, -3.0412e-006, -1.0853e-006],
        [3.949e-007,    -3.029e-008, -3.8209e-007,  1.0012e-007,  4.7133e-008],
        [-6.36e-010,    -1.309e-009,   6.048e-009, -1.1409e-009,  -6.676e-010]
    ])

@jit
def _spice(t,s):
    """Return spiciness.

    A state variable for characterizing water masses and their
    diffusive stability

    Parameters
    ----------
    t : numpy.array
        potential temperature (degC)
    s : numpy.array
        salinity (PSU)

    Returns
    -------
    spice : numpy.array
        spiciness  [kg/m^3]

    Notes
    -----
    This variable is defined in Flament (2002) [1]_. Several related state
    variables have been proposed (see e.g. [2]_). Note that spiciness is only
    physically meaningful close to the surface.

    References
    ----------
    .. [1] Flament (2002) A state variable for characterizing water
           masses and their diffusive stability: spiciness. Progress
           in Oceanography Volume 54, 2002, Pages 493-501.
           http://www.satlab.hawaii.edu/spice/spice.html
    .. [2] Jackett and McDougall, Deep Sea Research, 32A, 1195-1208, 1985.

    Examples
    --------
    >>> _spice(15,33)
    0.54458641375

    """
    dspi    = 0.
    dsalref = s - 35.
    dtmp  = 1.
    for ji in range(6):
        dsal = 1.
        for jj in range(5):
            dspi = dspi +   _spice_coefs[ji,jj] * dtmp * dsal
            dsal *= dsalref
        dtmp *= t
    return dspi


#------------------ Wrapped functions for dataarrays ---------------------------
def spice(t,s):
    """Return spiciness.

    A state variable for characterizing water masses and their
    diffusive stability

    Parameters
    ----------
    t : xarray dataarray
        potential temperature (degC)
    s : xarray dataarray
        salinity (PSU)

    Returns
    -------
    spice : numpy.array
        spiciness

    Notes
    -----
    This variable is defined in Flament (2002) [1]_. Several related state
    variables have been proposed (see e.g. [2]_). Note that spiciness is only
    physically meaningful close to the surface.

    References
    ----------
    .. [1] Flament (2002) A state variable for characterizing water
           masses and their diffusive stability: spiciness. Progress
           in Oceanography Volume 54, 2002, Pages 493-501.
           http://www.satlab.hawaii.edu/spice/spice.html
    .. [2] Jackett and McDougall, Deep Sea Research, 32A, 1195-1208, 1985.
    """
    pass
