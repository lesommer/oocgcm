.. currentmodule:: oocgcm

Linear filtering
================

Description
-----------

Linear filters are implemented in py:module:`oocgcm.filtering.linearfilters`.
 They are based on the py:class:`xarray.DataArray.Window`/py:class:`xarray
 .DataSet.Window` object that works in a similar fashion as py:class:`xarray
 .DataArray.Rolling` and can be used with multi-dimensional arrays. If chunks
  are precised in the py:class:`xarray.DataSet` or py:class:`xarray
  .DataArray` objects, the computation is performed using py:module:`dask`.

The py:class:`xarray.DataArray.Window` can be applied on dataset with missing
 values such as land areas for oceanographic data. In this case, the filter
 weights are normalized to take into account only valid data. In general,
 such normalization is applied by computing the low-passed data :math:`Y_{LP}`:
.. math::
    Y_{LP} = \frac{W * Y}{W * M},
 where :math:`Y` is the raw data, :math:`W` the window used, and :math:`M a
 mask that is one for valid data and zero for missing values.


Example
-------

.. notebook:: linearfilters.ipynb