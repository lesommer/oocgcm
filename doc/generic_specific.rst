.. currentmodule:: oocgcm

Generic versus data-specific tools
===================================

oocgcm is designed so that most methods can be used with several different types
of gridded data.

There are two classes of methods in oocgcm. Methods that are specific to a
particular model or source of data and methods that are data-agnostic or
generic.

For generic tools, we try to use naming conventions for internal variables
based on CF conventions or COMODO conventions.

Data-specific methods are generally created as particular instances of
generic methods. This is in particular the case for grid descriptor objects.

The distinction between data-agnostic and data-specific methods is reflected
in the general structure of the code.

All the methods related to a particular sources of data are put together in a
specific folder (eg. oocgcm.oceanmodels.nemo) so that it is easier for third
party to maintain data-specific tools associated to their particular source
of data / model. 
