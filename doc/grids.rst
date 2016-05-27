.. currentmodule:: oocgcm

Grid descriptor objects
=======================

One of the key concepts of oocgcm is the notion of grid that is implemented
in py:module:`oocgcm.core.grids`. The current version implements a generic
two-dimensional lat/lon grid in py:class:`oocgcm.core.grids.generic_2d_grid`.
Future versions will have a similar object for three-dimensional data.

Grid objects can be created in various ways depending on the source of gridded
data. A present, two-dimensional grid objects can be created : 
 - from arrays of latitude and longitude, 
 - from arrays of projection coordinate x and y,
 - from NEMO ocean model netCDF output files.

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

Grids descriptors can also be constructed from netCDF files describing the model grid 
(only available for NEMO so far). This method should be preferred as the 
metric factors that described the grid (e.g. ``grd["cell_x_size_at_t_location"]``)
are more accurate in this case. 

>>> from oocgcm.oceanmodels.nemo import grids
>>> grd = grids.nemo_2d_grid(nemo_coordinate_file=...,nemo_byte_mask_file=...,chunks=...)

.. note::

   Construction from netCDF files should be preferred as the metric factors that 
   described the grid (e.g. ``grd["cell_x_size_at_t_location"]``)
   are more accurate in this case.


Grids descriptors can also be constructed from x,y coordinate (in m). This can be useful for 
analysing idealized model experiments (eg. on the f-plane or beta-plane).

>>> from oocgcm.griddeddata import grids 
>>> x = np.arange(start=0, stop=1.e7, step=1.e6,dtype=float)
>>> y = np.arange(start=0, stop=1.2e7, step=1.e6,dtype=float)
>>> grd = grids.plane_2d_grid(ycoord=y,xcoord=x)

.. note::

   It should be noted that the creation of grid descriptors does not load any data nor creates
   any additional numpy arrays. A grid descriptors only defines xarray.DataArray instances.
   Grid object just contain the information on how to perform a particular calculation. 
 

Once created, a grid descriptor ``grd`` can be restricted to a subdomain as follows

>>> zoom = grd[500:800,2000:2500]

Operation on vector fields
--------------------------

Grid descriptor objects allow to perform **operations on vector fields** (as defined in oocgcm.core). 
For instance, 

>>> from oocgcm.core.grids import VectorField2d
>>> vectorfield1 = VectorField2d(...,...)
>>> vectorfield2 = VectorField2d(...,...)
>>> sprod = grd.scalar_product(vectorfield1,vectorfield2)
>>> cprod = grd.vertical_component_of_the_cross_product(vectorfield1,vectorfield2)

.. note::

    Methods associated to grid object only define xarray.DataArray instances. The actual 
    computation only occurs when xarray.DataArray values are explicitely requested.

Differential operators 
----------------------

Grid descriptor objects implement **differential operators** that can be applied to gridded data : 

>>> sst = ...
>>> u = ...
>>> v = ...
>>> current = VectorField2d(u,v)
>>> gsst = grd.horizontal_gradient(sst)
>>> curl = grd.vertical_component_of_curl(current)

Note that by default, oocgcm uses low order discretization methods. This can be easily overridden 
by the user provided the reference API is followed. 

Spatial integration
-------------------

Grid descriptors also implement methods for performing spatial integration : 

>>> ssh = ...
>>> sst = ...
>>> index = grd.spatial_average_xy(ssh,where=sst<2.) # average ssh where sst <2.
>>> index.plot() # plots the time series 


Broadcasting additional dimensions 
---------------------------------

Methods associated with two-dimensional grid descriptors define operations based on ('y','x') dimension. 
Because of the flexibility of xarray, these methods can therefore be applied to dataarrays with 
additional extra dimensions (for instance, time, index in an ensemble simulation). 

For instance, if ``ssh`` is a xarray dataarray of sea surface height built from a collection of netCDF 
files corresponding to different times, 

>>> import xarray as xr
>>> xr.open_mfdataset(path_to_my_files)['sossheig']

the norm of sea surface heigh gradient is defined as follows

>>> gssh = grd.norm_of_vectorfield(grd.horizontal_gradient(ssh))

but ``gssh`` does not hold any data yet. The calculation is only performed when a partical slice of 
data is requested, as for instance if the first two-dimensional field is written in a netCDF file. 

>>> gssh[0].to_necdf(path_to_my_output_file)

This absstrat representation of operations that allows xarray is key for efficiently implementing 
out-of-core procedures.

