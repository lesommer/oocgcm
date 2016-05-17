.. currentmodule:: oocgcm

#############
API reference
#############

This page provides an auto-generated summary of oocgcm's API.


Parameters
==========

.. autosummary::
   :toctree: generated/

   parameters.physicalparameters.coriolis_parameter
   parameters.physicalparameters.beta_parameter

Core utilities
==============

Grid-related data structures
----------------------------

.. autosummary::
   :toctree: generated/

   core.grids.generic_2d_grid
   core.grids.VectorField2d
   core.grids.VectorField3d
   core.grids.Tensor2d

I/O tools
--------
.. autosummary::
   :toctree: generated/

   core.io.return_xarray_dataset
   core.io.return_xarray_dataarray

Other utilities
---------------
.. autosummary::
   :toctree: generated/

   core.utils.add_extra_attrs_to_dataarray
   core.utils.is_numpy
   core.utils.is_xarray
   core.utils.is_daskarray
   core.utils.has_chunks
   core.utils.map_apply

Tools for gridded data
======================

Grid-related data structures
----------------------------

.. autosummary::
   :toctree: generated/

   griddeddata.grids.variables_holder_for_2d_grid_from_latlon_arrays
   griddeddata.grids.latlon_2d_grid

Tools for NEMO ocean model
==========================

Grid-related data structures
----------------------------

.. autosummary::
   :toctree: generated/

   oceanmodels.nemo.grids.variables_holder_for_2d_grid_from_nemo_ogcm
   oceanmodels.nemo.grids.nemo_2d_grid
