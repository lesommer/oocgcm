#!/usr/bin/env python

"""
oocgcm.filtering.timefilters
Define functions for linear and non-linear temporal filtering
"""

import dask.array as da
import xarray as xr
import numpy as np
import scipy.signal.windows as win
import pylab as plt
import scipy.ndimage as im
import importlib

# ------------------------------------------------------------------------------
# First part : Definition of several window functions
# ------------------------------------------------------------------------------


def lanczos(n, fc=0.02):
	"""
	Compute the coefficients for a Lanczos window

	Parameters
	----------
	n : int
		Number of points in the output window, must be an odd integer.
	fc : float
		Cutoff frequency

	Returns
	-------
	w : ndarray
		The weights associated to the boxcar window
	"""
	if not isinstance(n, int):
		try:
			n = int(n)
		except:
		  TypeError, "n must be an integer"
	if not n % 2 == 1:
		raise ValueError, "n must an odd integer"
	k = np.arange(- n / 2 + 1 , n / 2 + 1)
	#k = np.arange(0, n + 1)
	#w = (np.sin(2 * pi * fc * k) / (pi * k) *
	#     np.sin(pi * k / n) / (pi * k / n))
	w = (np.sin(2. * np.pi * fc * k) / (np.pi * k) *
	     np.sin(np.pi * k / (n / 2.)) / (np.pi * k / (n / 2.)))
	# Particular case where k=0
	w[n / 2] = 2. * fc
	return w


# Define a list of available windows
_scipy_window_dict = win._win_equiv
_local_window_dict = {'lanczos': lanczos}
# ------------------------------------------------------------------------------
# First part : Definition of window classes for filtering
# ------------------------------------------------------------------------------


class Window(xr.DataArray):
	"""
	Class for all different type of windows
	"""
	def __init__(self, window_name, n=None, dims=None, fastpath=False, **kargs):
		# TODO: Test if the function generalizes to several dimensions
		"""
		Parameters
		----------
		"""
		if fastpath:
			self._variable = window_name
			self._initialized = True
		else:
			if window_name in _scipy_window_dict:
				window_function = _scipy_window_dict[window_name]
			elif window_name in _local_window_dict:
				window_function = _local_window_dict[window_name]
			else:
				raise ValueError, "This type of window is not supported, " \
				                  "please check the name"

			coords = []
			coefficients = 1.
			self.order = dict()
			for weight_numbers, dimension_name in zip(n, dims) :
				self.order[dimension_name] = weight_numbers
				# Compute the coefficients associated to the window using the right
				# function
				coefficients1d = window_function(2 * weight_numbers + 1, **kargs)
				# Normalize the coefficients
				coefficients1d /= np.sum(coefficients1d)
				coefficients = np.outer(coefficients, coefficients1d)
				coords.append((dimension_name,
					           np.arange(- weight_numbers, weight_numbers + 1)))
			super(self.__class__, self).__init__(coefficients.squeeze(),
			                                     coords=coords,
			                                     name=window_name)

	def convolve(self, data, mode='reflect'):
		"""
		Convolve the current window with
		"""
		# Check if the data has more dimensions than the window and add
		# extra-dimensions to the window if it is the case
		new_dims = []
		for dim in data.dims:
			if not dim in self.dims:
				new_dims.append(dim)
		if new_dims:
			coefficients = self.variable.expand_dims(new_dims)
		else:
			coefficients = self.variable
		depth = dict()
		for coord in self.coords:
			print self.coords
			depth[self.get_axis_num(dim)] = self.order[dim]
		print depth

		def convolve(x):
			xf = im.convolve(x, coefficients, mode=mode)
			return xf
		res = data.data.map_overlap(convolve, depth=depth,
		                            boundary=mode, trim=True)
		return res

	def plot(self):

		# Note: I have overloaded the plot for function because I have an
		# issue with the inheritance of DataArray
		# TODO: Check if it possible to use the plot function from DataArray
		"""
		Plot the weights distribution of the window
		"""
		plt.figure()
		plt.plot(self.data)
		plt.title(self.name + " window")
		plt.ylabel("Amplitude")
		plt.xlabel("Sample")
		plt.figure()
		A = np.fft.fft(self.data, 2048) / (len(self.data) / 2.0)
		freq = np.linspace(-0.5, 0.5, len(A))
		response = 20 * np.log10(np.abs(np.fft.fftshift(A / abs(A).max())))
		plt.plot(freq, response)
		plt.axis([-0.5, 0.5, -120, 0])
		plt.title("Frequency response of the " + self.name + " window")
		plt.ylabel("Normalized magnitude [dB]")
		plt.xlabel("Normalized frequency [cycles per sample]")


def tapper(data, name, axis=0, **kargs):
	"""
	Do a tappering of the data using any window among the available windows (
	see the notes below)

	Parameters
	----------
	data : dask array
		The data on which apply the window function
	name : string
		The name of the window to use

	keywords arguments:
		Any arguments relative to the additional parameters of the window
		function (see the notes below)

	Returns
	-------
	data_tappered : dask array
		The data tappered y the window

	Notes
	-----
	"""
	#TODO: Write the function using the Window class
	pass


def rednoise(alpha, n, c=0.):
	"""
	Generate a vector similar to a red noise signal

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
	#TODO: Should it be moved to a library dedicated to timeseries ?
	res = np.zeros(n)
	res[0] = c + np.random.normal(0, 1)
	for i in range(1, n):
		res[i] = c + alpha * res[i-1] +  np.random.normal(0, 1)
	return res