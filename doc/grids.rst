.. currentmodule:: oocgcm

Grid descriptor objects
=======================

One of the key concepts of oocgcm is the notion of grid that is implemented
in py:module:`oocgcm.core.grids`. The current version implements a
two-dimensional lat/lon grid in py:class:`oocgcm.core.grids.generic_2d_grid`.
Future versions will have a similar object for three-dimensional data.


Grid objects can be created in various ways depending on the source of gridded
data.


Grid objects can be sliced
