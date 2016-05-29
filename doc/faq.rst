.. currentmodule:: oocgcm

Frequently Asked Questions
==========================

Why is the code architecture so complex ?
-----------------------------------------
The architecture and layout of oocgcm reflects the constraint that are inherent
to build a tool that can be easily adapted to different sources of data.

We have decided to separate as much as possible the generic tools from the
data-specific tools. Data-specific tools are generally created from generic
method and try to avoid unnecessary duplication. But some method are inherently
data-specific. 

For instance, the 'time' dimension of NEMO ocean model output is
called 'time_counter', but oocgcm assumes that the 'time' dimension is 't'.
It is therefore necessary to change the name of this dimension when reading
NEMO model output. This is done in oocgcm.oceanmodels.nemo.io methods.

We have chosen to gather all the methods that are specific to a certain source
of data within a unique folder for facilitating the maintenance by third-parties.

In practice, switching from one model to the other should be as simple as
changing

.. code-block:: python

   from oocgcm.oceanmodels.nemo import io, grids

into

.. code-block:: python

   from oocgcm.oceanmodels.mitgcm import io, grids
