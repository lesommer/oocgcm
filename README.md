[![Build Status](https://travis-ci.org/lesommer/oocgcm.svg?branch=master)](https://travis-ci.org/lesommer/oocgcm)
[![codecov.io](https://codecov.io/github/lesommer/oocgcm/coverage.svg?branch=master)](https://codecov.io/github/lesommer/oocgcm?branch=master)
# oocgcm
#### Out of core analysis tools for C-grid general circulation models in python.

This project provides tools for post-processing and **analyzing** output of
**general circulation models** and **gridded satellite data**.

Our aim is to simplify the analysis of **very large datasets of model output**
(~1-100Tb) like those produced by basin-to-global scale sub-mesoscale permitting
ocean models and ensemble simulations of eddying ocean models by leveraging the
potential of [xarray](https://github.com/pydata/xarray) and
[dask](https://github.com/dask/dask) python packages.

The project is so far mostly targeting [NEMO](http://www.nemo-ocean.eu/)
ocean model and gridded ocean satellite data (AVISO, SST, ocean color...)
but we try to build a framework that can be used for other ocean models as well.
The framework can also be used for C-grid atmospheric general
circulation models.

#### Installation
```
git clone https://github.com/lesommer/oocgcm.git
cd oocgcm
python setup.py install
```

#### Status
The project is in pre-alpha phase. More information can be found on the
project [documentation]( http://oocgcm.rtfd.io) and on the project [wiki](https://github.com/lesommer/oocgcm/wiki).
Ideas, comments and contributions are welcome !!


#### Related projects

oocgcm project builds upon experience with previous projects including :
 - [PyDom](http://servforge.legi.grenoble-inp.fr/projects/PyDom) and
 - [pycomodo](http://pycomodo.forge.imag.fr/).
 - [pyclim](http://servforge.legi.grenoble-inp.fr/projects/soft-pyclim)

A number of related projects based on [xarray](https://github.com/pydata/xarray)
and [dask](https://github.com/dask/dask) python packages are available in the
atmopshere/ocean/climate community. This include in particular:
 - [xgcm](https://github.com/xgcm/xgcm)
 - [aospy](https://github.com/spencerahill/aospy)
 - [MPAS xarray wrapper](https://github.com/pwolfram/mpas_xarray_wrapper)
 - [infinite-diff](https://github.com/spencerahill/infinite-diff/)
 - [windspharm](https://github.com/ajdawson/windspharm)

More can be found on related projects on this discussion thread on
[xarray forum](https://groups.google.com/forum/#!topic/xarray/pv1d3txTLEw).
#### License

Copyright 2016, oocgcm Developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
