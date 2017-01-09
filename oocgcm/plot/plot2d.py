#!/usr/bin/env python

"""
oocgcm.plot.plot1d
Define nice plotting function for bidimensional data series using matplotlib
"""

import numpy as np
import pylab as plt
import matplotlib.mlab as mlab
from matplotlib.ticker import MultipleLocator
import matplotlib

def spectrum2d_plot(ax, x, y, z, xlog=False, ylog=False, zlog=False, **kwargs):
	"""
	Define a nice spectrum with twin x-axis and twin y-axis, one with 
	frequencies, the other one with periods, on a predefined axis 
	object.

	Parameters
	----------
	x,y : array_like
		1D array defining the coordinates
	z : array_like
		2D array
	xlog, ylog, zlog : bool, optional
		Define if the x-axis, y-axis and z-axis are plotted with a
		log	scale
	** kwargs : optional keyword arguments
		See matplotlib.axes.Axes.contourf method in matplotlib
		documentation
	"""
	if not 'xlim' in kwargs:
		xlim = None
	else:
		xlim = kwargs['xlim']
		del kwargs['xlim']
	if not 'ylim' in kwargs:
		ylim = None
	else:
		ylim = kwargs['ylim']
		del kwargs['ylim']
	if not 'zlim' in kwargs:
		zlim = None
	else:
		zlim = kwargs['zlim']
		del kwargs['zlim']

	n_lev = 40
	# if symmetric:	
	# lim = max(np.max(z), abs(np.min(z)))
	# lev = np.hstack((np.linspace(- lim, 0, n_lev / 2 + 1),
	# np.linspace(0, lim, n_lev / 2)[1:]))
	#
	# else:		   
	# lev = np.linspace(np.min(z), np.max(z), n_lev / 2 + 1)
	if zlog:
		plot = ax.pcolormesh(np.log10(z), **kwargs)
	else:
		plot = ax.pcolormesh(z, **kwargs)
	# X limits			
	if xlog:
		ax.set_xscale('symlog', nonposx='clip')
		xmin = np.ceil(np.log10(x[1,])) - 1
		xmax = np.ceil(np.log10(x[-1,]))
		ax.set_xlim((10 ** xmin, 10 ** xmax))
	else:
		try:
			ax.set_xlim(xlim)
		except:
			ax.set_xlim(np.min(x), np.max(x))
	# Y limits		
	if ylog:
		ax.set_yscale('symlog', nonposx='clip')
		ymin = np.ceil(np.log10(x[1,])) - 1
		ymax = np.ceil(np.log10(x[-1,]))
		ax.set_ylim((-10 ** ymin, 10 ** ymax))
	else:
		try:
			ax.set_ylim(ylim)
		except:
			ax.set_ylim(np.min(y), np.max(y))
	axtwiny = ax.twiny()
	if xlog:
		axtwiny.set_xscale('symlog', nonposx='clip')
		axtwiny.set_xlim((-10 ** xmin, 10 ** xmax))
		A = np.arange(2, 10, 2)[np.newaxis]
		B = 10 ** (np.arange(-xmax, -xmin, 1)[np.newaxis])
		C = np.dot(B.transpose(), A)
		new_major_ticks = 10 ** np.arange(xmin + 1, xmax, 1)
		new_minor_ticklabels = C.flatten()
		new_minor_ticklabels = new_minor_ticklabels.astype(int)
		new_minor_ticks = 1. / new_minor_ticklabels
		axtwiny.set_xticks(new_minor_ticks, minor=True)
		axtwiny.set_xticklabels(new_minor_ticklabels, minor=True,
		                        rotation=30)
		new_major_ticklabels = 1. / new_major_ticks
		new_major_ticklabels = new_major_ticklabels.astype(int)
		axtwiny.set_xticks(new_major_ticks)
		axtwiny.set_xticklabels(new_major_ticklabels, rotation=30)
	axtwinx = ax.twinx()
	if ylog:
		axtwinx.set_yscale('symlog', nonposx='clip')
		axtwinx.set_ylim(y[1], y[-1])
		axtwinx.set_ylim((10 ** ymin, 10 ** ymax))
		new_major_ticks = 10 ** np.arange(ymin + 1, ymax, 1)
		new_major_ticklabels = 1. / new_major_ticks
		new_major_ticklabels = new_major_ticklabels.astype(int)
		axtwinx.set_yticks(new_major_ticks)
		axtwinx.set_yticklabels(new_major_ticklabels)
		A = np.arange(2, 10, 2)[np.newaxis]
		B = 10 ** (np.arange(-ymax, -ymin, 1)[np.newaxis])
		C = np.dot(B.transpose(), A)
		new_minor_ticklabels = C.flatten()
		new_minor_ticklabels = new_minor_ticklabels.astype(int)
		new_minor_ticks = 1. / new_minor_ticklabels
		axtwinx.set_yticks(new_minor_ticks, minor=True)
		axtwinx.set_yticklabels(new_minor_ticklabels, minor=True)
	ax.grid(True, which='both')		
