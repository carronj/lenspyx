import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    config.add_extension('lenspyx.bicubic.bicubic', ['lenspyx/bicubic/bicubic.f90'])
    config.add_extension('lenspyx.shts.fsht', ['lenspyx/shts/shts.f90'],
                extra_link_args=['-lgomp'],libraries=['gomp'], extra_f90_compile_args=['-fopenmp', '-w'])
    return config

setup(
    name='lenspyx',
    version='0.0.1',
    packages=['lenspyx', 'tests', 'lenspyx.bicubic'],
    url='https://github.com/carronj/lenspyx',
    author='Julien Carron',
    author_email='to.jcarron@gmail.com',
    description='lensed CMB sims pipe',
    install_requires=['healpy', 'numpy', 'pyfftw'],
    long_description=long_description,
    configuration=configuration)

