#!/usr/bin/env python
#
"""oocgcm.oceanfuncs.eos.misc
Equation of state related functions that are not attached to a particular
equation of state of sea water.

"""

import numpy as np

from ...core.utils import map_apply, _assert_are_compatible_dataarrays

#---------------------- Functions for numpy arrays ----------------------------

def _spice(t,s):
    """Return spiciness.

    A state variable for characterizing water masses and their
    diffusive stability

    Parameters
    ----------
    t : numpy.array
        potential temperature (°C)
    s : numpy.array
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

    Examples
    --------
    >>> spice(15,33)
    0.54458641375

    """
    B = numpy.zeros((7,6))
    B[1,1] = 0
    B[1,2] = 7.7442e-001
    B[1,3] = -5.85e-003
    B[1,4] = -9.84e-004
    B[1,5] = -2.06e-004

    B[2,1] = 5.1655e-002
    B[2,2] = 2.034e-003
    B[2,3] = -2.742e-004
    B[2,4] = -8.5e-006
    B[2,5] = 1.36e-005

    B[3,1] = 6.64783e-003
    B[3,2] = -2.4681e-004
    B[3,3] = -1.428e-005
    B[3,4] = 3.337e-005
    B[3,5] = 7.894e-006

    B[4,1] = -5.4023e-005
    B[4,2] = 7.326e-006
    B[4,3] = 7.0036e-006
    B[4,4] = -3.0412e-006
    B[4,5] = -1.0853e-006

    B[5,1] = 3.949e-007
    B[5,2] = -3.029e-008
    B[5,3] = -3.8209e-007
    B[5,4] = 1.0012e-007
    B[5,5] = 4.7133e-008

    B[6,1] = -6.36e-010
    B[6,2] = -1.309e-009
    B[6,3] = 6.048e-009
    B[6,4] = -1.1409e-009
    B[6,5] = -6.676e-010
    #
    coefs = B[1:7,1:6]
    spice = numpy.zeros(t.shape)
    ss = s - 35.
    bigT = numpy.ones(t.shape)
    for i in range(6):
        bigS = numpy.ones(t.shape)
        for j in range(5):
            spice+= coefs[i,j]*bigT*bigS
            bigS*= ss
        bigT*=t
    return spice


#------------------ Wrapped functions for dataarrays ---------------------------
def _spice(t,s):
    """Return spiciness.

    A state variable for characterizing water masses and their
    diffusive stability

    Parameters
    ----------
    t : xarray dataarray
        potential temperature (°C)
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
