#!/usr/bin/env python 

from setuptools import setup, find_packages

MAJOR = 0
MINOR = 0
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''

DISTNAME = 'oocgcm'
LICENSE = 'not defined yet'
AUTHOR = 'oocgcm Developers'
AUTHOR_EMAIL = 'julien.lesommer@gmail.com'
URL = 'https://github.com/lesommer/oocgcm'

CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
]


INSTALL_REQUIRES = ['numpy >= 1.7', 'xarray >= 0.7.2']

DESCRIPTION = "Out of core diagnostics of C-grid GCMs in Python"

LONG_DESCRIPTION = """
**oocgcm** is an open source project and Python package that aims to 
simplify the analysis of large scale datasets (1-100Tb) produced by 
ocean general circulation model. The project builds upon the xarray
and dask python packages.
Our goal is to provide a set of tools that allow to perform out-of-core
computations within a robust and intuitive framework. We intend to  
reduce the time needed to developp a complex analysis from the prototyping
phase to the production phase.

Important links
---------------
- Issue tracker: http://github.com/lesommer/oocgcm/issues
- Source code: http://github.com/lesommer/oocgcm
"""

setup(name=DISTNAME,
      version=VERSION,
      license=LICENSE,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      classifiers=CLASSIFIERS,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      install_requires=INSTALL_REQUIRES,
      url=URL,
      packages=find_packages())
