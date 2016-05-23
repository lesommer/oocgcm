.. currentmodule:: oocgcm


The need for out-of-core analysis tools
=======================================

A practical need
----------------

oocgcm is a pure Python package built on the top of xarray_ and dask_.
But why have we choosen to implement a diagnostic package based on
xarray_ and dask_ instea of pursuing the development of packages based on numpy ?

Most of the tools that are implemented in oocgcm are indeed already available in
several python libraries based on numpy_ and one of the several netCDF interfaces
for python.

But numpy-based model diagnostic libraries are facing a challenge with the ongoing
evolution of geoscientific models and earth observing networks. With **the
most high-end models being runs on several tens of thousand cores**, even
a two-dimensional slice of model output cannot be loaded in memory at one time.
Model diagnostic tools and gridded data analysis **tools should therefore be
parallelized** and run out-of-core.

One option is to run FORTRAN/MPI codes steered from shell scripts but there you
loose the **flexibility and the multiple benefits of a python-based workflow**.
Another option is to use one of the several libraries available for parallel
computing in python. This usually implies a complete refactoring of your
analysis package at the risk of eventually using different analysis tools
depending on the size of your dataset...

A wider perspective
-------------------

We believe that the above question is not a "technical" problem but **a real
challenge for our fields of research**. We are here facing a "technical" translation
of one of the "big data challenges" in earth system science.

Transforming large gridded datasets into scientific results indeed requires
**innovative descriptive approaches that merge statistical descriptions and
physically-motivated analyses**. This usually involves performing rather "complex"
analysis tasks on the dataset.

In practice, this is usually a two stage process. A first stage involves
transforming an **idea** into a **prototype** code. Using a interactive scripting
language usually accelerates this phase because you can glue different bits of
code and see the results on the fly.

The second stage involves transforming the **prototype** code into a
**production** code. And this is usually a difficult task especially for students,
because it requires a more in-depth understanding of the hardware infrastructure
and of the software design.

So, depending on the language and on the libraries you use, what actually
changes is not your ability to perform your analysis but rather **the time it takes
for you to prepare your production code** and eventually reach the scientific result
you are after.

More generally, there is an objective risk for our fields of research if we don't
embrace the question of the development of analysis tools that **accelerate the
idea-prototype-production-results cycle**. If the time needed to transform
an idea into an efficient production code is too long, we will keep performing
only simple or routine analyses on our datasets, eventually missing
**the potential breakthrough of big-data in earth system sciences**.

.. _xarray: https://github.com/pydata/xarray
.. _dask: http://dask.pydata.org
.. _numpy: http://www.numpy.org/
