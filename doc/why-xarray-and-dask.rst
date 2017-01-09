.. currentmodule:: oocgcm

Why using xarray and dask ?
===========================

oocgcm is a pure Python package built on the top of xarray_, which itself
integrates dask.array to support streaming computation on large datasets that
don’t fit into memory. Why have we choosen to use xarray_ and dask_ ?


xarray
------

xarray_ implements a N-dimensional variants of the core pandas_ data structures.
In addition, xarray_ adopts the Common Data Model for self-describing  data.
In practice, xarray.Dataset is an in-memory representation
of a netCDF file or of a collection of netCDF files.

Building upon xarray_ has several advantages :
 - metadata available in the netCDF files are associated with xarray objects in the form of a Python dictionary ``x.attrs``. This simplifies the exploration of the dataset, yields more robust code, and simplifies the export of the results to netCDF files.
 - because dimensions are associated with the variable in xarray objects, xarray allows flexible split-apply-combine operations with groupby ``x.groupby('time.dayofyear').mean()``
 - xarray objects do not load data in memory by default. Loading the data is only done at the execution time if needed. This means that the user has access to all his dataset without having to worry about loading the data, therefore simplifying the prototyping of a new analysis.
 - xarray is natively integrated with pandas_, meaning that xarray objects can straightforwardly be exported to pandas DataFrames. This allows to easily access a range of time-series analysis tools.
 - xarray objects can be exported to iris_ or cdms_ so that the user can merge several different analysis tools in his workflow.
 - Little work is needed for applying a numpy function to xarray objects. Several numpy ufunc are already applicable to xarray.DataArray data-structure.


dask
----

dask_ implement an abstract graph representation of the dynamic task scheduling
needed for performing out-of-core computation. dask_ also implement an efficient
scheduling procedure for optimizing the execution time of acyclic graphs (DAG)
of tasks on a given machine.

From a user standpoint the key concept of dask.array is the notion of chunk.
A chunk is the user-defined shape of the subdataset on which the unitary tasks
will be applied.

dask_ allows to easily leverage the resources of **shared memory architectures**
(multi-core laptop or work-station) but also the resources of **distributed memory
architectures** (clusters of cpu).

At present, xarray_ integrates dask_ functionalities for shared memory
architectures. xarray_ will also allow to leverage dask_ potential on
distributed memory architectures in the future.


Building upon dask_ has several advantages :
 - parallelization comes at no cost. The only modification of your code that is needed is your defining the chunks on which the computation should be performed.
 - dask_ back-end methods are generic, powerful and well tested for a number of different applications.
 - dask_ comes with powerful and easy-to-use profiling tools for optimizing the execution time on a given machine.

Most importantly, xarray_ and dask_ are supported by active and friendly teams
of developers, that we hereby gratefully acknowledge.


.. _xarray: http://xarray.pydata.org
.. _dask: http://dask.pydata.org
.. _pandas : http://pandas.pydata.org/
.. _iris : http://scitools.org.uk/iris/
.. _cdms : http://uvcdat.llnl.gov/documentation/cdms/cdms_1.html
