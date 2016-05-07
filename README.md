# oocgcm
#### Out of core diagnostics of C-grid general circulation models in python.


This project provides tools for processing and analysing output of ocean general 
circulation models and gridded satellite data.

Our aim is to simplify the analysis of **very large datasets of model output**
(~1-100Tb) like those produced by basin-to-global scale submesoscale permitting 
ocean models and ensemble simulations of edddying ocean models by leveraging the 
potential of [xarray](https://github.com/pydata/xarray) and 
[dask](https://github.com/dask/dask) python packages.

The main ambition of this project is to provide simple tools for performing
**out-of-core** computations with ocean model output, namely processing data
that is too large to fit into a computer's main memory at one time.

The project is so far mostly targetting [NEMO](http://www.nemo-ocean.eu/) 
ocean model and gridded ocean satellite data (AVISO, SST, ocean color...)
but we try to build a framework that can be used for other ocean models as well. 
The framework could also probably be used for C-grid atmospheric general circulation 
models. 

We are trying to developp a framework **flexible** enough that does not impose
too stricly a specific workflow to the end user.

We try to keep the **list of dependencies** as small as possible to simplify the
deployment on a number of platforms.

This project builds upon previous projects including
[pyclim](http://servforge.legi.grenoble-inp.fr/projects/soft-pyclim)
[PyDom](http://servforge.legi.grenoble-inp.fr/projects/PyDom) and 
[pycomodo](http://pycomodo.forge.imag.fr/).

#### Installation
```
git clone https://github.com/lesommer/oocgcm.git
python setup.py install
```
#### Status
The project is in pre-alpha phase . More information can be found on the
project [wiki](https://github.com/lesommer/oocgcm/wiki). Ideas, comments and
contributions are welcome !!

