#!/usr/bin/env python

"""
oocgcm.filtering.linearfilters
Define functions for linear filtering that works on multi-dimensional
xarray.DataArray and xarray.Dataset objects.
"""

import xarray as xr
import numpy as np
import pylab as plt
import scipy.signal.windows as win
import scipy.ndimage as im
from dask.diagnostics import ProgressBar

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
	k = np.arange(- n / 2 + 1, n / 2 + 1)
	# k = np.arange(0, n + 1)
	# w = (np.sin(2 * pi * fc * k) / (pi * k) *
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

# @xr.register_dataarray_accessor('win')
# @xr.register_dataset_accessor('win')
class Window(object):
	"""
	Class for all different type of windows
	"""

	_attributes = ['name', 'dims', 'order']

	def __init__(self, xarray_obj):
		self._obj = xarray_obj
		self.obj = xarray_obj
		self.name = 'boxcar'
		self.dims = None
		self.order = 2
		self.coefficients = 1.
		self.coords = []
		self._depth = dict()

	def __repr__(self):
		"""
		Provide a nice string representation of the window object
		"""
		# Function copy from xarray.core.rolling
		attrs = ["{k}->{v}".format(k=k, v=getattr(self, k))
		         for k in self._attributes if
		         getattr(self, k, None) is not None]
		return "{klass} [{attrs}]".format(klass=self.__class__.__name__,
		                                  attrs=','.join(attrs))

	def set(self, window_name=None, n=None, dims=None, chunks=None, **kargs):
		"""
		Set the different properties of the current window

        If the variable associated to the window objetc is a non-dask array,
        it will be converted to dask array. If it's a dask array, it will be
        rechunked to the given chunksizes.

        If neither chunks is not provided for one or more dimensions, chunk
        sizes along that dimension will not be updated; non-dask arrays will be
        converted into dask arrays with a single block.

		Parameters
		----------
		window_name : str
			Name of the window among the available windows (see Notes below)
		n : int or tuple of int
			Half-size of the window
		dims : str or tuple of str
			Name of the dimension along which to
        chunks : int, tuple or dict, optional
            Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
            ``{'x': 5, 'y': 5}``.
		keywords
			Additional parameters specific to some windows functions (see the
			Notes below)

		"""
		self.obj = self._obj.chunk(chunks=chunks)
		if window_name is not None:
			self.name = window_name
		if dims is None and self.dims is None:
			dims = self._obj.dims[0]
		if isinstance(dims, str):
			self.dims = [dims,]
		else:
			self.dims = dims
		for dim in self.dims:
			if not dim in self._obj.dims:
				raise ValueError, ("Dimension " + dim + "does not exist")
		if window_name in _scipy_window_dict:
			window_function = _scipy_window_dict[window_name]
		elif window_name in _local_window_dict:
			window_function = _local_window_dict[window_name]
		else:
			raise ValueError, "This type of window is not supported, " \
			                  "please check the name"
		self.order = dict()
		self.coefficients = 1.
		self.coords = []
		for weight_numbers, dimension_name in zip(n, dims):
			self.order[dimension_name] = weight_numbers
			# Compute the coefficients associated to the window using the right
			# function
			coefficients1d = window_function(2 * weight_numbers + 1,
			                                 **kargs)
			# Normalize the coefficients
			coefficients1d /= np.sum(coefficients1d)
			self.coefficients = np.outer(self.coefficients, coefficients1d)

		self.coefficients = self.coefficients.squeeze()
		for dim in self.obj.dims:
			axis_dim = self.obj.get_axis_num(dim)
			if dim not in self.dims:
				self.coefficients = np.expand_dims(self.coefficients,
				                                   axis=axis_dim)
			else:
				self._depth[self.obj.get_axis_num(dim)] = self.order[dim]
			self.coords.append(np.asarray(self.obj.coords[dim]))

	def apply(self, mode='reflect', weights=None, compute=True):
		"""
		Convolve the current window with the data
		"""
		# Check if the data has more dimensions than the window and add
		# extra-dimensions to the window if it is the case
		if weights is None:
			mask = self.obj.notnull()
			weights = im.convolve(mask.astype(float), self.coefficients,
			                      mode=mode)
		filled_data = self.obj.fillna(0.).data / weights

		def convolve(x):
			xf = im.convolve(x, self.coefficients, mode=mode)
			return xf

		data = filled_data.map_overlap(convolve, depth=self._depth,
	                                   boundary=mode, trim=True)
		if compute:
			with ProgressBar():
				out = data.compute()
		else:
			out = data
 		res = xr.DataArray(out, dims=self.obj.dims, coords=self.coords,
		                   name=self.obj.name)

		return res.where(mask == 1)

	def plot(self):
		"""
		Plot the weights distribution of the window and the associated
		spectrum (work only for 1D and 2D windows).
		"""
		nod = len(self.dims)
		if nod == 1:
			fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
			# First plot: weight distribution
			n = self._depth[0]
			ax1.plot(np.arange(-n, n + 1), self.coefficients.squeeze())
			ax1.set_xlim((-n, n))
			ax1.set_title("Window: " + self.name)
			ax1.set_ylabel("Amplitude")
			ax1.set_xlabel("Sample")
			# Second plot: frequency response
			A = np.fft.fft(self.coefficients.squeeze(), 2048) / (len(
				self.coefficients.squeeze()) / 2.0)
			freq = np.linspace(-0.5, 0.5, len(A))
			response = 20 * np.log10(np.abs(np.fft.fftshift(A / abs(A).max())))
			ax2.plot(freq, response)
			ax2.axis([-0.5, 0.5, -120, 0])
			ax2.set_title("Frequency response of the " + self.name + " window")
			ax2.set_ylabel("Normalized magnitude [dB]")
			ax2.set_xlabel("Normalized frequency [cycles per sample]")
			ax2.grid(True)
			plt.tight_layout()
		elif nod == 2:
			fig = plt.figure(1, figsize=(8, 8))
			# Definitions for the axes
			left, width = 0.13, 0.5
			bottom, height = 0.13, 0.5
			bottom_h = left + width + 0.05
			left_h = bottom_h + 0.05
			rect_2D = [left, bottom, width, height]
			rect_x = [left, bottom_h, width, 0.25]
			rect_y = [left_h, bottom, 0.25, height]
			ax_2D = plt.axes(rect_2D)
			ax_x = plt.axes(rect_x)
			ax_y = plt.axes(rect_y)
			ax_x.plot(f_x[Ix].squeeze(), R[Ix].squeeze())
			ax_x.set_xlim(xlim)
			ax_x.grid(True)
			ax_x.set_ylabel(r'$R(f_x,0)$', fontsize=24)
			ax_y.plot(R[Iy].squeeze(),f_y[Iy].squeeze())
			ax_y.set_ylim(ylim)
			ax_y.grid(True)
			ax_y.set_xlabel(r'$R(0,f_y)$', fontsize=24)
			ax_2D.pcolormesh(f_x, f_y, R)
			ax_2D.set_xlim(xlim)
			ax_2D.set_ylim(ylim)
			ax_2D.grid(True)
			ax_2D.set_ylabel(r'$f_y$', fontsize=24)
			ax_2D.set_xlabel(r'$f_x$', fontsize=24)
			box = dict(facecolor='white', pad=10.0)
			ax_2D.text(0.97, 0.97, r'$R(f_x,f_y)$', fontsize='x-large',
					   bbox=box, transform=ax_2D.transAxes,
					   horizontalalignment='right',
					   verticalalignment='top')
			pass
		else:
			raise ValueError, "This number of dimension is not supported by the" \
			                  "plot function"

def tapper(data, window_name, dims, **kargs):
	"""
	Do a tappering of the data using any window among the available windows (
	see the notes below)

	Parameters
	----------
	data : Datarray
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
	# TODO: Write the function using the Window class
	pass

