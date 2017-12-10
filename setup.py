#! /usr/bin/env python

from setuptools import setup, Extension

setup(
    name='pyPyrTools',
    version='0.2.1',
    description='Python tools for multi-scale image processing, including Laplacian pyramids, Wavelets, and Steerable Pyramids',
    license='MIT',
    url='https://github.com/LabForComputationalVision/pyPyrTools',
    author='Eero Simoncelli',
    author_email='eero.simoncelli@nyu.edu',
    keywords='multi-scale image-processing',
    packages=['pyPyrTools'],
    package_data={'': ['*.h', 'LICENSE']},
    install_requires=['numpy>=1.1',
                      'scipy>=0.18',
                      'matplotlib>=1.5',
                      'Pillow>=3.4'],
    ext_modules=[Extension('wrapConv',
                           sources=['pyPyrTools/convolve.c', 'pyPyrTools/edges.c',
                                    'pyPyrTools/wrap.c', 'pyPyrTools/internal_pointOp.c'],
                           depends=['pyPyrTools/convolve.h', 'pyPyrTools/internal_pointOp.h'],
                           extra_compile_args=['-fPIC', '-shared'])],
    tests='tests',
    )
