#! /usr/bin/env python

from wheel.bdist_wheel import bdist_wheel
from setuptools import setup, Extension

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
    ext_modules=[Extension('pyrtools.pyramids.c.wrapConv',
                           sources=['src/pyrtools/pyramids/c/py.c',
                                    'src/pyrtools/pyramids/c/convolve.c',
                                    'src/pyrtools/pyramids/c/edges.c',
                                    'src/pyrtools/pyramids/c/wrap.c',
                                    'src/pyrtools/pyramids/c/internal_pointOp.c'],
                           depends=['src/pyrtools/pyramids/c/meta.h',
                                    'src/pyrtools/pyramids/c/convolve.h',
                                    'src/pyrtools/pyramids/c/internal_pointOp.h'],
                           extra_compile_args=['-fPIC', '-shared'])],
    cmdclass={"bdist_wheel": WheelABINone},
    )
