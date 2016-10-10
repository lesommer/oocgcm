.. currentmodule:: oocgcm

#############
API reference
#############

This page provides an auto-generated summary of oocgcm's API.



General purpose utilities and data-structures
=============================================

Parameters
----------

.. autosummary::
   :toctree: generated/

   parameters.physicalparameters.coriolis_parameter
   parameters.physicalparameters.beta_parameter

Data-structures
---------------

.. autosummary::
   :toctree: generated/

   core.grids.VectorField2d
   core.grids.VectorField3d
   core.grids.Tensor2d

I/O tools
--------
.. autosummary::
   :toctree: generated/

   core.io.return_xarray_dataset
   core.io.return_xarray_dataarray
   oceanmodels.nemo.io.return_xarray_dataset
   oceanmodels.nemo.io.return_xarray_dataarray

Miscellaneous
-------------
.. autosummary::
   :toctree: generated/

   core.utils.is_numpy
   core.utils.is_xarray
   core.utils.is_daskarray
   core.utils.has_chunks
   core.utils.map_apply


Two-dimensional grid descriptor objects
=======================================

Two-dimensional grid descriptors hold all the information required for
defining operations on xarray dataarray that require information on the
underlying grid of the model.

Generic two-dimensional model-agnostic grid descriptor
------------------------------------------------------

.. autosummary::
   :toctree: generated/

   core.grids.generic_2d_grid


Creating grids descriptor from arrays
-------------------------------------

.. autosummary::
   :toctree: generated/

   griddeddata.grids.variables_holder_for_2d_grid_from_latlon_arrays
   griddeddata.grids.latlon_2d_grid
   griddeddata.grids.variables_holder_for_2d_grid_from_plane_coordinate_arrays
   griddeddata.grids.plane_2d_grid

Tools for filtering timeseries and spatial fields
=================================================

.. autosummary::
   :toctree: generated/

   oocgcm.filtering.timefilters


Testing tools
=============

.. autosummary::
   :toctree: generated/

   oocgcm.test.signals


Tools for NEMO ocean model
==========================

Creating grids descriptors from model output
--------------------------------------------

.. autosummary::
   :toctree: generated/

   oceanmodels.nemo.grids.variables_holder_for_2d_grid_from_nemo_ogcm
   oceanmodels.nemo.grids.nemo_2d_grid

