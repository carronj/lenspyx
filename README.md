# lenspyx

[![PyPI version](https://badge.fury.io/py/lenspyx.svg)](https://badge.fury.io/py/lenspyx)[![alt text](https://readthedocs.org/projects/lenspyx/badge/?version=latest)](https://lenspyx.readthedocs.io/en/latest)[![Build Status](https://travis-ci.com/carronj/lenspyx.svg?branch=master)](https://travis-ci.com/carronj/lenspyx)

Curved-sky python lensed CMB maps simulation package by Julien Carron.

This allows one to build very easily (if familiar with healpy) lensed CMB simulations. Parallelization is done with openmp.
The numerical cost is approximately that of an high-res harmonic transform.

The package basically provides two methods. Check the [doc](https://lenspix.readthedocs.io/en/latest).

(NB: This implementation is independent from the similar-sounding [lenspix](https://github.com/cmbant/lenspix) package by A.Lewis)

### Installation
    
    pip install lenspyx [--user]

The –-user is required only if you don’t have write permission to your main python installation. A fortran compiler is required for a successful installation.

![ERC logo](https://erc.europa.eu/sites/default/files/content/erc_banner-vertical.jpg)
