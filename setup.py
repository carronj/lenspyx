import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    return config

exec(open('lenspyx/_version.py').read())
setup(
    name='lenspyx',
    version=__version__,
    packages=['lenspyx', 'lenspyx.remapping', 'lenspyx.tests'],
    url='https://github.com/carronj/lenspyx',
    author='Julien Carron',
    data_files=[('lenspyx/data/cls', ['lenspyx/data/cls/FFP10_wdipole_lensedCls.dat',
                                      'lenspyx/data/cls/FFP10_wdipole_lenspotentialCls.dat'])],
    author_email='to.jcarron@gmail.com',
    description='lensed CMB sims pipe',
    install_requires=['ducc0'],
    long_description=long_description,
    configuration=configuration)

