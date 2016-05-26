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
   oceanmodels.nemo.return_xarray_dataset
   oceanmodels.nemo.return_xarray_dataarray

Miscellaneous
-------------
.. autosummary::
   :toctree: generated/

   core.utils.add_extra_attrs_to_dataarray
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


Creating grids descriptors from model output
--------------------------------------------

.. autosummary::
   :toctree: generated/

   oceanmodels.nemo.grids.variables_holder_for_2d_grid_from_nemo_ogcm
   oceanmodels.nemo.grids.nemo_2d_grid


Attributes
----------

.. autosummary::
   :toctree: generated/

   core.grids.generic_2d_grid.dims
   core.grids.generic_2d_grid.ndims
   core.grids.generic_2d_grid.shape
   core.grids.generic_2d_grid.chunks

Manipulating grids
------------------

generic_2d_grid implement the mapping interface with keys given by variable names
and values given by ``DataArray`` objects.

.. autosummary::
   :toctree: generated/

   core.grids.generic_2d_grid.__getitem__
   core.grids.generic_2d_grid.__contains__
   core.grids.generic_2d_grid.__iter__
   core.grids.generic_2d_grid.chunk
   core.grids.generic_2d_grid.get_projection_coordinates


Changing the grid location of arrays
------------------------------------

.. autosummary::
   :toctree: generated/

   core.grids.generic_2d_grid.change_grid_location_t_to_u
   core.grids.generic_2d_grid.change_grid_location_u_to_t
   core.grids.generic_2d_grid.change_grid_location_t_to_v
   core.grids.generic_2d_grid.change_grid_location_v_to_t
   core.grids.generic_2d_grid.change_grid_location_f_to_u
   core.grids.generic_2d_grid.change_grid_location_f_to_v
   core.grids.generic_2d_grid.change_grid_location_v_to_u
   core.grids.generic_2d_grid.change_grid_location_u_to_v



Vector calculus
--------------

 .. autosummary::
    :toctree: generated/

    core.grids.generic_2d_grid.norm_of_vectorfield
    core.grids.generic_2d_grid.scalar_product
    core.grids.generic_2d_grid.scalar_outer_product
    core.grids.generic_2d_grid.vertical_component_of_the_cross_product


Differential operators
---------------------

 .. autosummary::
    :toctree: generated/

    core.grids.generic_2d_grid.horizontal_gradient
    core.grids.generic_2d_grid.horizontal_gradient_vector
    core.grids.generic_2d_grid.horizontal_gradient_tensor
    core.grids.generic_2d_grid.horizontal_laplacian
    core.grids.generic_2d_grid.vertical_component_of_curl
    core.grids.generic_2d_grid.horizontal_divergence


Spatial integration
------------------

 .. autosummary::
    :toctree: generated/

    core.grids.generic_2d_grid.integrate_dxdy
    core.grids.generic_2d_grid.spatial_average_xy

Operators specific to oceanic applications
------------------------------------------

 .. autosummary::
    :toctree: generated/

    core.grids.generic_2d_grid.geostrophic_current_from_sea_surface_height
    core.grids.generic_2d_grid.q_vector_due_to_kinematic_deformation
    core.grids.generic_2d_grid.frontogenesis_function
