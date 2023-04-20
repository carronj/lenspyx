import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    return config


setup(
    name='lenspyx',
    version='2.0.0',
    packages=['lenspyx'],
    url='https://github.com/carronj/lenspyx',
    author='Julien Carron',
    data_files=[('lenspyx/data/cls', ['lenspyx/data/cls/FFP10_wdipole_lensedCls.dat',
                                      'lenspyx/data/cls/FFP10_wdipole_lenspotentialCls.dat',
                                      'lenspyxs/data/cls/FFP10_wdipole_params.ini'])],
    author_email='to.jcarron@gmail.com',
    description='lensed CMB sims pipe',
    long_description=long_description,
    configuration=configuration)

