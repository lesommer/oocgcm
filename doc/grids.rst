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
