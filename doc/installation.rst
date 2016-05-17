.. currentmodule:: oocgcm

Installation
============


Dependancies
------------

oocgcm is a pure Python package, but some of its dependencies are not. The
easiest way to get them installed is to use conda_. You can install all oocgcm
dependancies with the conda command line tool::

    $ conda install xarray dask netCDF4 bottleneck

.. _conda: http://conda.io/


Installing oocgcm developpers version
-------------------------------------

The project is still in pre-alpha phase, the only way to install oocgcm is
therefore to clone the repository on github:: 

    $ git clone https://github.com/lesommer/oocgcm.git
    $ cd oocgcm
    $ python setup.py install
