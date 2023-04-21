

# lenspyx

[![PyPI version](https://badge.fury.io/py/lenspyx.svg)](https://badge.fury.io/py/lenspyx)[![alt text](https://readthedocs.org/projects/lenspyx/badge/?version=latest)](https://lenspyx.readthedocs.io/en/latest)[![Build Status](https://travis-ci.com/carronj/lenspyx.svg?branch=master)](https://travis-ci.com/carronj/lenspyx)

Curved-sky python lensed CMB maps simulation package by Julien Carron.

This allows one to build very easily lensed CMB simulations. 

The package explicitly provides two methods for most basic usage. Check the [doc](https://lenspyx.readthedocs.io/en/latest). 
There is also a notebook [demo_lenspyx](examples/demo_lenspyx.ipynb) for examples and sanity checks.

There are further tools for CMB lensing reconstruction (adjoint lensing etc.) which may be useful to CMB-lensing intensive applications.

**From v2 onwards (april 2023)**: 

Lenspyx now essentially only wraps extremely efficient routines from [DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc) by M.Reinecke,
with massive speed-ups and accuracy improvements (see [this paper](https://arxiv.org/abs/2304.10431)), in a way incompatible to v1 which is now abandoned.

Required is ducc0 version >= 0.30.0.
For best performance, please refer to the [DUCC page](https://gitlab.mpcdf.mpg.de/mtr/ducc) for installation instructions making profit of compiler-specfic optimizations.

### Installation

The code should hopefully be platform independent.

Editable installation from source: clone the repo and
    
    pip install --no-binary ducc0 -e ./ [--user]

From pypi

    pip install --no-binary ducc0 lenspyx [--user]

The –-user is required only if you don’t have write permission to your main python installation.
The ducc0 installation without binaries might take a while (several minutes).

In case you use this package and also wish to acknowledge it, you can provide a reference to [this paper](https://arxiv.org/abs/2304.10431).

![SNSF logo](./docs/SNF_logo_standard_web_color_neg_e.svg)