#!/usr/bin/env python

"""
oocgcm.filtering.linearfilters
Define functions for linear filtering that works on multi-dimensional
xarray.DataArray and xarray.Dataset objects.
"""

# Unittest
import unittest
# Xarray
import xarray as xr
# Oocgcm
import oocgcm.filtering.linearfilters as tf
from oocgcm.test.signals import signaltest_xyt1
# Matplotlib
import pylab as plt

class TestExecutable(unittest.TestCase):

	def window_plot2d(self):
		signal_xyt = signaltest_xyt1()
		win2d = signal_xyt.win
		win2d.set(window_name='tukey', dims=['y', 'x'], n=[24, 36])
		win2d.plot()
		plt.savefig('lanczos2d_plot.png')

	def compute_boundary_weights(self):
		signal_xyt = signaltest_xyt1(coastlines=True)
		win2d = signal_xyt.win
		win2d.set(window_name='tukey', dims=['y', 'x'], n=[24, 36])
		win2d.boundary_weights(drop_dims=['time'])

if __name__ == '__main__':
	test_list = ['compute_boundary_weights']
	suite = unittest.TestSuite(map(TestExecutable, test_list))
	unittest.TextTestRunner(verbosity=2).run(suite)