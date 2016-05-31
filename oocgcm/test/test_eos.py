import os
import numpy as np

import xarray as xr

from . import TestCase, assert_equal,assert_allclose

from oocgcm.oceanfuncs.eos import misc

def test_numpy_spice():
    assert_allclose(misc._spice(15,33),0.54458641375)
