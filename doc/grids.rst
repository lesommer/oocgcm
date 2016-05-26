.. currentmodule:: oocgcm

Grid descriptor objects
=======================

One of the key concepts of oocgcm is the notion of grid that is implemented
in py:module:`oocgcm.core.grids`. The current version implements a generic
two-dimensional lat/lon grid in py:class:`oocgcm.core.grids.generic_2d_grid`.
Future versions will have a similar object for three-dimensional data.

Grid objects can be created in various ways depending on the source of gridded
data. A present two-dimensional grid objects can be created from arrays of
latitude and longitude, from arrays of coordinate or from nemo model output.

Grid descriptors provide access to all the information that may be needed for
defining operations on the grid. They implement methods for vector calculus,
differential calculus and spatial integration. Grid descriptors can also be
sliced for defining descriptors of smaller portions of the physical domain.

In practice, a grid object can be created from arrays of latitude and longitude 
(1d or 2d arrays).

>>> from oocgcm.griddeddata import grids
>>> lats = ... 
>>> lons = ...
>>> grd = grids.latlon_2d_grid(latitudes=lats,longitudes=lons)

grids can also be constructed from netcdf files describing the model grid 
(only available for NEMO so far). This method should ne preferred as the 
metric factor defining the grid are more accurate in this case. 

>>> from oocgcm.oceanmodels import grids
>>> grids.nemo_2d_grid(nemo_coordinate_file=...,nemo_byte_mask_file=...,chunks=...)

grids can also be constructed from x,y coordinate (in m). This can be useful for 
analysing idealized model experiments (eg. on the f-plane or beta-plane).

>>> from oocgcm.griddeddata import grids 
>>> x = np.arange(start=0, stop=1.e7, step=1.e6,dtype=float)
>>> y = np.arange(start=0, stop=1.2e7, step=1.e6,dtype=float)
>>> grd = grids.plane_2d_grid(ycoord=y,xcoord=x)

Grid descriptor objects allow to perform **operations on vector fields** (as defined in oocgcm.core). 
For instance, 

>>> from oocgcm.core.grids import VectorField2d
>>> vectorfield1 = VectorField2d(...,...)
>>> vectorfield2 = VectorField2d(...,...)
>>> sprod = grd.scalar_product(vectorfield1,vectorfield2)
>>> cprod = grd.vertical_component_of_the_cross_product(vectorfield1,vectorfield2)


Grid descriptor object implement **differential operators** to be applied to model output : 

>>> sst = ...
>>> u = ...
>>> v = ...
>>> current = VectorField2d(u,v)
>>> gsst = grd.horizontal_gradient(sst)
>>> curl = grd.vertical_component_of_curl(current)

Note that by default, oocgcm uses low order discretization methods. This can be easily overridden 
by the user provided the reference API is followed. 



Grid descriptors also implement methods for performing spatial integration : 

>>> ssh = ...
>>> sst = ...
>>> index = grd.spatial_average_xy(ssh,where=sst<2.) # average ssh where sst <2.
>>> index.plot() # plots the time series 


Two important points 

 - as most methods in oocgcm, grid methods define xarray.DataArray, and therefore only define
   how the computation will be done. The actual computation only occur when xarray.DataArray 
   values are explicitely requested. 

 - grid methods define operations based on ('y','x') dimension. They can therefore be applied 
   to xarray.DataArray with extra dimensions (for instance, time, index in an ensemble simulation). 

