.. oocgcm documentation master file, created by
   sphinx-quickstart on Tue May 10 23:41:48 2016.


oocgcm : out of core analysis tools for C-grid general circulation models in python.
====================================================================================

This project provides tools for processing and analysing output of ocean general
circulation models and gridded satellite data.

Our aim is to simplify the analysis of **very large datasets of model output**
(~1-100Tb) like those produced by basin-to-global scale submesoscale permitting
ocean models and ensemble simulations of edddying ocean models by leveraging the
potential of xarray_ and dask_  python packages.

The main ambition of this project is to provide simple tools for performing
**out-of-core** computations with ocean model output, namely processing data
that is too large to fit into a computer's main memory at one time.

The project is so far mostly targetting NEMO_
ocean model and gridded ocean satellite data (AVISO, SST, ocean color...)
but we try to build a framework that can be used for other ocean models as well.
The framework could also probably be used for C-grid atmospheric general
circulation models.

We are trying to developp a framework **flexible** enough in order not to impose
too stricly a specific workflow to the end user.

We try to keep the **list of dependencies** as small as possible to simplify the
deployment on a number of platforms.

.. _xarray: https://github.com/pydata/xarray
.. _dask: http://dask.pydata.org
.. _NEMO: http://www.nemo-ocean.eu/


.. note::

   oocgcm is at the pre-alpha stage. The API is therefore unstable.  

Documentation
-------------

.. toctree::
   :maxdepth: 1

   installation
   generic_specific
   grids
   io
   api
   faq
   whats-new


Get in touch
------------

- Report bugs, suggest feature ideas or view the source code `on GitHub`_.

.. _on GitHub: http://github.com/lesommer/oocgcm
