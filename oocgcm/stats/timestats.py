#!/usr/bin/env python

"""
oocgcm.stats.timestats
Define functions to perform timeseries analysis
"""
import xarray as xr
import numba
import numpy as np


def detrend(x, t=None, typ='linear'):
	"""
	Fit a linear or quadratic trend and remove it.

	Parameters
	----------
	x : ndarray
		Input timeseries
	typ : {'linear', 'quadratic'}
		Type of interpolation

	Returns
	-------
	x_dtr : ndarray

	trend :  ndarray
	"""
	dim = np.shape(x)
	T = np.reshape(np.resize(t, np.size(x)), dim, order='F')
	if typ == 'linear':
		a, b, _, _, _ = linreg(x, t)
		trend = a * T + b
		x_dtr =  x - trend
	elif typ == 'quadratic':
		a, b, c, _ = quadreg(x, t)
		trend = a * T ** 2 + b * T + c
		x_dtr =  x - trend
	return x_dtr, trend

@numba.jit
def loess(x, t=None):
	"""
	Fit a nonlinear trend using the LOcal regrEssion method
	"""
	#TODO: Write the LOESS function using Numba