.. currentmodule:: oocgcm

Contributing to oocgcm
======================

If you are willing to contribute to developing oocgcm (reporting bugs,
suggesting improvments, ...), here is some information.

Suggesting improvments to oocgcm
--------------------------------

As for any piece of software stored on github_, contributing to oocgcm_ involves
your knowing the basics of git_ version control. Then just

 - create a github_ account
 - fork_ oocgcm repository
 - push the proposed changes on your forked repository
 - Submit a `pull request`_

Contributions are also welcome on oocgcm `issues tracker`_.

If you don't feel comfortable with the above tools, just contact_ me by email.

.. _github: https://github.com/
.. _fork: https://help.github.com/articles/fork-a-repo/
.. _oocgcm: https://github.com/lesommer/oocgcm
.. _pull request: https://help.github.com/articles/using-pull-requests/
.. _git: https://git-scm.com/doc
.. _issues tracker: https://github.com/lesommer/oocgcm/issues
.. _contact: http://lesommer.github.io/contact/

Overall layout of the library
-----------------------------

Here is some information about the overall layout of oocgcm library::

    oocgcm/
        setup.py
        oocgcm/
        docs/
        examples/
        ci/

``setup.py``
    installation script.

``oocgcm``
    contains the actual library

``docs``
    contains the documentation in rst format. Used for building the docs on
    readthedocs with sphinx and numpydoc

``examples``
    provides example of applications in notebooks and python scripts

``ci``
    contains the yaml files for continuous integration. Used for testing the
    build and testing oocgcm with several combinations of libraries.


Structure of ``oocgcm`` package
-------------------------------

The actual package itself contains the following submodules::


    oocgcm/
        oocgcm/
            core/
            parameters/
            griddeddata/
            oceanmodels/
            oceanfuncs/
            airseafuncs/
            spectra/
            stats/
            filtering/
            test/

``core``
    contains data-agnostic versions of methods that are inherently
    data-specific.

``parameters``
    defines physical and mathematical parameters that may be used in
    several submodules.

``griddeddata``
    contains data-specific methods adapted to two-dimensional
    gridded data, including satellite data.

``oceanmodels``
    contains data-specific methods adapted to a range of c-grid
    ocean models. Currently, contains only tools for NEMO ocean model.

``oceanfuncs``
    contains functions that are only relevant for analyzing ocean data
    but transverse to particular sources of data
    (a priori numpy version of the function and wrappers for xarray dataarrays).

``airseafuncs``
     will contain methods for analysing air-sea exchanges.

``regrid``
    will contain methods for regridding gridded data.

``spectra``
    will contain methods for computing wavenumber spectra and frequency
    spectra out of xarray.DataArray

``stats``
    will contain methods for applying descriptive statistics to gridded
    data.

``filtering``
    will contain methods for filtering gridded data in space or in
    time.

``test``
    contains the test series for oocgcm. unit tests are runs after each commit.


Structure of ``oocgcm.core``
----------------------------

``oocgcm/core`` contains::

    oocgcm/core/
        io.py
        grids.py
        utils.py


``io.py``
    contains functions that are used for creating xarray datasets and xarray
    dataarrays and functions used for writing output files.

``grids.py``
    contains tools that define grid descriptor objects.

``utils.py``
    contains useful functions for several submodules. This includes function
    for testing and asserting types.

Files with similar names and contents can be repeated for each specific source
of data when needed.
