.. currentmodule:: oocgcm

For oocgcm developpers : 
=======================

If you are willing to contribute to developping oocgcm, we here provide information 
about the structure of oocgcm library. 

structure of oocgcm library
---------------------------
 - core : contains data-agnostic versions of methods that are inherently
   data-specific
 - oceanmodels : contains data-specific methods adapted to a range of c-grid
   ocean models
 - griddeddata : contains data-specific methods adapted to two-dimensional
   gridded data, including satellite data
 - oceanfuncs : contains functions that are only relevant for analysing ocean data
   but transvers to particular sources of data (numpy version and xarray wrappers)
 - airseafuncs : will contain methods for analysing air-sea exchanges. 
 - regrid : will contain methods for regridding gridded data.
 - spectra : will contain methode for computing wavenumber spectra and frequency 
   spectra out of xarray.DataArray
 - stats : will contain methods for applying descriptive statistics to gridded 
   data.
 - filtering : will contain methods for filtering gridded data in space or in
   time
