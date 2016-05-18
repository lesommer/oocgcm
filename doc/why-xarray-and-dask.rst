.. currentmodule:: oocgcm

Why using xarray and dask ?
===========================

oocgcm is a pure Python package built on the top of xarray_, which itself
integrates dask_ to support streaming computation on large datasets that
don’t fit into memory. Why have we choosen to use xarray_ and dask_ ?

.. _xarray: https://github.com/pydata/xarray
.. _dask: http://dask.pydata.org

xarray
------

xarray_ implements a N-dimensional variants of the core pandas_ data structures.
In addition, xarray_ adopts the Common Data Model for self-describing  data.
In practice, py:class:`xarray.Dataset` is an therefore in-memory representation
of a netCDF file or a collection of netCDF files.

Building upon xarray_ has several advantages :
 - metadata available in the netCDF files are associated with the python data-structure. This simplifies the exploration of the dataset, yields more robust code, and simplifies the export of the results to netCDF files.
 - xarray objects do not load data in memory by default. Loading the data is only done at the execution time if need. This means that the user has access to all his dataset without having to worry about loading the data, therefore simplifying the prototyping of a new analysis.
 - xarray is natively integrated with pandas_, meaning that xarray objects can straightforwardly be exported to pandas DataFrames. This allows to easilt access a range of time-series analysis tools.
 - xarray objects can be export to iris_ or cdms_ so that the user can merge several different analysis tools in his workflow.
 - Little work is needed for applying a numpy function to xarray objects.


.. _xarray: https://github.com/pydata/xarray
.. _pandas : http://pandas.pydata.org/
.. _iris : http://scitools.org.uk/iris/
.. _cdms : http://uvcdat.llnl.gov/documentation/cdms/cdms_1.html

dask
----
dask_ implement an abstract description of the split/apply/combine process
needed for performing out-of-core computation. dask_ also implement an efficient
scheduling procedure for optimizing the execution of the tree of task on a given
machine.

from a user standpoint the key concept of dask_ is the notion of chunk. A chunk
is the user-defined shape of the subdataset on which the unitary tasks will be
applied.

dask_ allows to easily leverage the resources of shared memory architectures
(multi-core laptop or work-station) but also the resources of distributed memory
architectures (clusters of cpu).

At present, xarray_ leveraged dask functionnalities for shared memory
architectures. xarray_ will also allow to leverage dask potential on
distributed memory architectures in the future.


Building upon dask_ has several advantages :
 - parallelization comes at no cost. The only modification of your code that is needed is your defining the chunks on which the computation should be performed.
 - dask_ back-end methods are generic, powerfull and well tested for a number of different applications.
 - dask_ comes with powerful and easy-to-use profiling tools for optimizing the execution time on a given machine.

Most importantly, xarray_ and dask_ are supported by active and friendly teams
of developpers, that we hereby gratefully acknowledge.

 .. _dask: http://dask.pydata.org
 .. _xarray: https://github.com/pydata/xarray
