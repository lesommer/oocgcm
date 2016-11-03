#!/usr/bin/env python

"""
oocgcm.test.signal
Define a bunch of useful signals to test the different analysis functions of
oocgcm
"""

import numpy as np
import xarray as xr
import numba
import unittest

savedir = '../../examples/signals/'

#-------------------------------------------------------------------------------
# Timeseries
#-------------------------------------------------------------------------------
@numba.jit
def rednoise(alpha, n, c=0., coords=None):
	"""
	Generate a timeseries similar to a red noise signal of a particular size

	Parameters
	----------
	alpha: int
		Coefficient of autocorrelation
	n: int
		Size of the vector
	c: float
		Mean of the vector

	Returns
	-------
	res: 1darray
		A vector of size n describing a red noise

	"""
	res = np.zeros(n)
	res[0] = c + np.random.normal(0, 1)
	for i in range(1, n):
		res[i] = c + alpha * res[i - 1] + np.random.normal(0, 1)
	return res


@numba.jit
def ar2(alpha, beta, n, c=0.):
	"""
	Generate a timeseries using an autoregressive process of order 2

	Parameters
	----------
	alpha: int
		Coefficient of autocorrelation
	beta: int
		Second coefficient
	n: int
		Size of the vector
	c: float
		Mean of the vector

	Returns
	-------
	res: 1darray
		A vector of size n describing a red noise

	"""
	res = np.zeros(n)
	res[0] = c + np.random.normal(0, 1)
	res[1] = c + np.random.normal(0, 1)
	for i in range(2, n):
		res[i] = c + alpha * res[i - 1] +  beta * res[i - 2] + \
		         np.random.normal(0, 1)
	return res


#@numba.guvectorize
def trend(time, slope, offset):
	"""
	Generate a trend of size n
	"""
	return slope * time + offset

def signaltest_xyt1(coastlines=False):
	"""
	Generate
	"""
	nt = 100
	nx = 128
	ny = 128
	t1d = np.linspace(0, 20 * np.pi, nt)
	x1d = np.linspace(0, 2 * np.pi, nx)
	y1d = np.linspace(0, 2 * np.pi, ny)
	t, y, x = np.meshgrid(t1d, y1d, x1d, indexing='ij')
	# Create four times modulation with
	m1 = np.cos(1.5 * t)
	m2 = np.cos(2 * t)
	m3 = np.cos(0.5 * t)
	m4 = np.cos(t)
	# Create a spatio-temporal gaussian noise
	noise = 0.8 * np.random.normal(0, 0.2, (nt, ny, nx))
	# Create four spatial patterns
	z1 = np.sin(x) * np.sin(y) * m1
	z2 = np.sin(2.5 * x) * np.sin(y) * m2
	z3 = np.sin(x) * np.sin(2.5 * y) * m3
	z4 = np.sin(2.5 * x) * np.sin(2.5 * y) * m4
	z = z1 + z2 + z3 + z4 + noise
	if coastlines:
		z[:, 0:ny/4, 0:nx/4] = np.nan
	return xr.DataArray(z, coords=[t1d, y1d, x1d], dims=['time', 'y', 'x'], name='signal')


class TestExecutable(unittest.TestCase):

	def generate_signaltest_xyt1(self):
		ds = signaltest_xyt1().to_dataset()
		ds.to_netcdf(path=(savedir + 'signaltest_xyt1.nc'),
		             format='NETCDF3_64BIT')

	def generate_signaltest_xyt1_with_coastlines(self):
		ds = signaltest_xyt1(coastlines=True).to_dataset()
		ds.to_netcdf(path=(savedir + 'signaltest_xyt1_with_coastlines.nc'),
		             format='NETCDF3_64BIT')

if __name__ == '__main__':
	test_list = ['generate_signaltest_xyt1_with_coastlines']
	suite = unittest.TestSuite(map(TestExecutable, test_list))
	unittest.TextTestRunner(verbosity=2).run(suite)