#!/usr/bin/env python 

import os
import re
import sys
import warnings

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
ocean general circulation models. The project builds upon the xarray
and dask python packages.

Our goal is to provide a set of tools that allow to perform out-of-core
computations within a robust and intuitive framework. We intend to  
reduce the time needed to deploy a complex analysis of model output 
from the prototyping phase to the production phase.

Important links
---------------
- Issue tracker: http://github.com/lesommer/oocgcm/issues
- Source code: http://github.com/lesommer/oocgcm
"""


# code to extract and write the version copied from xarray
FULLVERSION = VERSION
write_version = True

if not ISRELEASED:
    import subprocess
    FULLVERSION += '.dev'

    pipe = None
    for cmd in ['git', 'git.cmd']:
        try:
            pipe = subprocess.Popen(
                [cmd, "describe", "--always", "--match", "v[0-9]*"],
                stdout=subprocess.PIPE)
            (so, serr) = pipe.communicate()
            if pipe.returncode == 0:
                break
        except:
            pass

    if pipe is None or pipe.returncode != 0:
        # no git, or not in git dir
        if os.path.exists('xarray/version.py'):
            warnings.warn("WARNING: Couldn't get git revision, using existing xarray/version.py")
            write_version = False
        else:
            warnings.warn("WARNING: Couldn't get git revision, using generic version string")
    else:
        # have git, in git dir, but may have used a shallow clone (travis does this)
        rev = so.strip()
        # makes distutils blow up on Python 2.7
        if sys.version_info[0] >= 3:
            rev = rev.decode('ascii')

        if not rev.startswith('v') and re.match("[a-zA-Z0-9]{7,9}", rev):
            # partial clone, manually construct version string
            # this is the format before we started using git-describe
            # to get an ordering on dev version strings.
            rev = "v%s.dev-%s" % (VERSION, rev)

        # Strip leading v from tags format "vx.y.z" to get th version string
        FULLVERSION = rev.lstrip('v')

else:
    FULLVERSION += QUALIFIER


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'oocgcm', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()

if write_version:
    write_version_py()



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
