import os
import numpy as np

import xarray as xr

from . import TestCase, assert_equal,assert_allclose,requires_numba

from oocgcm.oceanfuncs.eos import misc

@requires_numba
def test_numpy_spice():
    assert_allclose(misc._spice(15,33),0.54458641375)
