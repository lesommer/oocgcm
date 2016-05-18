.. oocgcm documentation master file, created by
   sphinx-quickstart on Tue May 10 23:41:48 2016.


oocgcm : out of core analysis tools for general circulation models in python.
=============================================================================

This project provides tools for processing and analysing output of general
circulation models and gridded satellite data in the field of Earth system
science.

Our aim is to simplify the analysis of **very large datasets of model output**
(~1-100Tb) like those produced by basin-to-global scale sub-mesoscale permitting
ocean models and ensemble simulations of eddying ocean models by **leveraging
the potential of xarray_ and dask_**  python packages.

The main ambition of this project is to provide simple tools for performing
**out-of-core** computations with model output and gridded data, namely
processing data that is too large to fit into memory at one time.

The project is so far mostly targeting NEMO_
ocean model and gridded ocean satellite data (AVISO, SST, ocean color...)
but our aim is to build **a framework that can be used for a variety of models
based on the Arakawa C-grid**. The framework can in principle also be used for
atmospheric general circulation models.

We are trying to develop a framework **flexible** enough in order not to impose
too strictly a specific workflow to the end user.

oocgcm is a pure Python package and we try to keep the **list of dependencies**
as small as possible in order to simplify the deployment on a number of platforms.

oocgcm is **not intended to provide advanced visualization** functionalities for
gridded geographical data as several powerful tools already exist in the
python ecosystem
(see in particular cartopy_ and basemap_).

We rather focus on building a framework that simplifies the design and production
of **advanced statistical and dynamical analyses of large datasets** of model
output and gridded data.


.. _xarray: https://github.com/pydata/xarray
.. _dask: http://dask.pydata.org
.. _NEMO: http://www.nemo-ocean.eu/
.. _cartopy : http://scitools.org.uk/cartopy/
.. _basemap : http://matplotlib.org/basemap/


.. note::

   oocgcm is at the pre-alpha stage. The API is therefore unstable and likely
   to change without notice.

Documentation
-------------

.. toctree::
   :maxdepth: 1

   out-of-core
   why-xarray-and-dask
   installation
   generic_specific
   user-workflow
   grids
   io
   api
   faq
   whats-new


Get in touch
------------

- Report bugs, suggest feature ideas or view the source code `on GitHub`_.

.. _on GitHub: http://github.com/lesommer/oocgcm
