#!/usr/bin/env python
#
"""modelgrids module.
Define the classes that give acces to model grid metrics and operators (e.g. gradients) 
"""

"""
## to do list 
  - dtype
  - get_shape
  - dask from arrays
  - pbm with xarray concatenation
"""

import numpy as np
import numpy.ma as ma
import dask.array as da
from netCDF4 import Dataset
import xarray as xr

from collections import Iterable

axis2dim = {-1:'x',-2:'y'}

class generic_netcdf_loader_for_grids:
    def __init__(self,array_type=None,chunks=None):
	"""
	"""
        self.array_type = array_type
        self.chunks = chunks

    def __call__(self,filename=None,varname=None):
	if self.array_type == 'numpy':
            out = Dataset(filename).variables[varname][:].squeeze()
        elif self.array_type == 'xarray':
	    ds = xr.open_dataset(filename)
            out = ds[varname][:]
	elif self.array_type == 'dask':
	    d = Dataset(filename).variables[varname][:].squeeze()
	    out = da.from_array(np.array(d), chunks=self.chunks)
        return out

class generic_grid:
    """Base class implementing differential operators
    """
    def __init__(self):
	pass

    def load_horizontal_metrics(self):
        self.e1t = self._load(self.coordfile,varname='e1t')
        self.e2t = self._load(self.coordfile,varname='e2t')
        self.e1u = self._load(self.coordfile,varname='e1u')
        self.e2u = self._load(self.coordfile,varname='e2u')
        self.e1v = self._load(self.coordfile,varname='e1v')
        self.e2v = self._load(self.coordfile,varname='e2v')
	self.shape = self.e1t.shape # with dask ? 

    def d_i(self,q):
        """Return the difference q(i+1) - q(i)"""
        di= q[...,1:] - q[...,0:-1]
        return di

    def d_j(self,q):
        """Return the difference q(j+1) - q(j)"""
        dj= q[...,1:,:] - q[...,0:-1,:]
        return dj

    def _extend_array(self,array,axis=None,where=None):
        """Extends an array to fit the initial grid.
             input : axis, where are integer   
	"""
	shape = list(self.shape)
	shape[axis] = 1
	shape = tuple(shape)
	boundary_value = self._zeros(shape)
	if where==1:
	   list_to_concatenate = [boundary_value,array]
	elif where==-1:
	   list_to_concatenate = [array,boundary_value]
        out = self._concatenate(list_to_concatenate,axis=axis)
        return out 

    def extend_array(self,q,axis=None,where=None):
        """Extends an array to fit the initial grid.
	     input : axis, where are integer or iterable
	"""
        if isinstance( axis, int ) and isinstance( where, int ):
            out = self._extend_array(q,axis=axis,where=where)
        #
        elif isinstance( axis, Iterable) and isinstance( where, Iterable):
            for iax,(ax,wh) in enumerate(zip(axis,wh)):
                if iax==0:
                   out = self._extend_array(q,axis=ax,where=wh)
                else:
                   out = self._extend_array(out,axis=ax,where=wh)
        return out


    def gradh(self,q):
        """Return the 2D gradient of a scalar field.
            input :  on T-grid
            output : on U-grid and V-grid
         """
        gx = self.d_i(q) / self.e1u[:,0:-1]
        gy = self.d_j(q) / self.e2v[0:-1,:]
        #
        egx = self.extend_array(gx,axis=-1,where=-1)
        egy = self.extend_array(gy,axis=-2,where=-1)
        #
        return egx,egy



class nemo_grid_with_numpy(generic_grid):

    def __init__(self, coordfile=None):
        generic_grid.__init__(self)
        self.coordfile = coordfile
        self.define_array_type_specific_functions()
        self.load_horizontal_metrics()

    def define_array_type_specific_functions(self):
	self._load = generic_netcdf_loader_for_grids(array_type='numpy')
        self._concatenate = np.concatenate
	self._zeros = np.zeros

 

class nemo_grid_with_dask(generic_grid):

    def __init__(self, coordfile=None,chunks=(1000,1000)):
	self.chunks = chunks
        generic_grid.__init__(self)
        self.coordfile = coordfile
        self.define_array_type_specific_functions()
        self.load_horizontal_metrics()

    def define_array_type_specific_functions(self):
        self._load = generic_netcdf_loader_for_grids\
			(array_type='dask',chunks=self.chunks)
	self._zeros = lambda n:da.zeros(n,chunks=self.chunks)
 
    def _concatenate(self,list_of_bits,axis=None):
	    return da.concatenate(list_of_bits,axis= axis)


class nemo_grid_with_xarray(generic_grid):

    def __init__(self, coordfile=None,chunks=(1000,1000)):
	self.chunks = chunks
        generic_grid.__init__(self)
        self.coordfile = coordfile
        self.define_array_type_specific_functions()
        self.load_horizontal_metrics()

    def define_array_type_specific_functions(self):
        self._load = generic_netcdf_loader_for_grids\
			(array_type='xarray',chunks=self.chunks)
        self._zeros = lambda n : xr.DataArray(np.zeros(n))
        self._concatenate = lambda listbits: xr.concat(listbits,dim=axis2dim[axis])
