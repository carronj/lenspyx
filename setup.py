import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    config.add_extension('lenspyx.bicubic', ['lenspyx/bicubic.f90'])
    config.add_extension('lenspyx.shts.fsht', ['lenspyx/shts/shts.f90'],
                         libraries=['gomp'], extra_compile_args=['-Xpreprocessor', '-fopenmp', '-w'])
#    config.add_extension('lenspix.bicubic', ['lenspix.bicubic.f90'],
#                         libraries=['gomp'],  extra_compile_args=['-Xpreprocessor', '-fopenmp', '-w'])
    return config

setup(
    name='lenspyx',
    version='0.0.1',
    packages=['lenspyx', 'tests'],
    url='https://github.com/carronj/lenspyx',
    author='Julien Carron',
    author_email='to.jcarron@gmail.com',
    description='lensed CMB sims pipe',
    install_requires=['healpy', 'numpy'],
    long_description=long_description,
    configuration=configuration)

