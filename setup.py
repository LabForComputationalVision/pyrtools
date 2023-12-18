#! /usr/bin/env python

from wheel.bdist_wheel import bdist_wheel
from setuptools import setup, Extension
import importlib
import os

# copied from kymatio's setup.py: https://github.com/kymatio/kymatio/blob/master/setup.py
pyrtools_version_spec = importlib.util.spec_from_file_location('pyrtools_version',
                                                               'pyrtools/version.py')
pyrtools_version_module = importlib.util.module_from_spec(pyrtools_version_spec)
pyrtools_version_spec.loader.exec_module(pyrtools_version_module)
VERSION = pyrtools_version_module.version

# Adapted from the cibuildwheel example https://github.com/joerick/python-ctypes-package-sample
# it marks the wheel as not specific to the Python API version.
class WheelABINone(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = bdist_wheel.get_tag(self)
        return "py3", "none", plat


setup(
    name='pyrtools',
    version=VERSION,
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    description='Python tools for multi-scale image processing, including Laplacian pyramids, Wavelets, and Steerable Pyramids',
    license='MIT',
    url='https://github.com/LabForComputationalVision/pyrtools',
    author='Eero Simoncelli',
    author_email='eero.simoncelli@nyu.edu',
    keywords='multi-scale image-processing',
    packages=['pyrtools', 'pyrtools.pyramids', 'pyrtools.tools', 'pyrtools.pyramids.c'],
    package_data={'': ['*.h', 'LICENSE']},
    install_requires=['numpy>=1.1',
                      'scipy>=0.18',
                      'matplotlib>=1.5',
                      'tqdm>=4.29',
                      'requests>=2.21'],
    ext_modules=[Extension('pyrtools.pyramids.c.wrapConv',
                           sources=['pyrtools/pyramids/c/py.c',
                                    'pyrtools/pyramids/c/convolve.c',
                                    'pyrtools/pyramids/c/edges.c',
                                    'pyrtools/pyramids/c/wrap.c',
                                    'pyrtools/pyramids/c/internal_pointOp.c'],
                           depends=['pyrtools/pyramids/c/meta.h',
                                    'pyrtools/pyramids/c/convolve.h',
                                    'pyrtools/pyramids/c/internal_pointOp.h'],
                           extra_compile_args=['-fPIC', '-shared'])],
    cmdclass={"bdist_wheel": WheelABINone},
    tests='TESTS',
    )
