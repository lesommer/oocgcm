# oocgcm
#### Out of core diagnostics of C-grid ocean models


This project provides tools for processing and analysing output of NEMO ocean model. 

Our aim is to simplify the analysis of **very large datasets of model output** (~1-100Tb) like those produced by submesoscale permitting models at basin scales and ensemble simulations of edddying ocean models. 

The main ambition of this project is to provide simple tools for performing **out-of-core** computations with ocean model output, namely processing  data that is too large to fit into a computer's main memory at one time. 


We are trying to developp a framework **flexible** enough that does not impose too stricly a specific workflow to the end user. 
 
We try to keep the **list of dependencies** as small as possible to simplify the deployment on a number of platforms. 
 
This project builds upon previous projects including [pyclim](http://servforge.legi.grenoble-inp.fr/projects/soft-pyclim) and [PyDom](http://servforge.legi.grenoble-inp.fr/projects/PyDom). 

#### Acknowledgment

oocgcm is made possible thanks to numpy/scipy ecosystem and in particular [dask](https://github.com/dask/dask) and [xarray](https://github.com/pydata/xarray) modules. 

