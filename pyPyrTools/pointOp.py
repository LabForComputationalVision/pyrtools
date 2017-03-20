import numpy
import ctypes
import os

libpath = os.path.dirname(os.path.realpath(__file__))+'/../wrapConv.so'
# load the C library
lib = ctypes.cdll.LoadLibrary(libpath)

def pointOp(image, lut, origin, increment, warnings):
    """Apply a point operation, specified by lookup table `lut`, to image `im`.
    
    `lut` must be a row or column vector, and is assumed to contain (equi-spaced) samples of the
    function.  `origin` specifies the abscissa associated with the first sample, and `increment`
    specifies the spacing between samples.  Between-sample values are estimated via linear
    interpolation.  If `warnings` is non-zero, the function prints a warning whenever the lookup
    table is extrapolated.

    This function is very fast and allows extrapolation beyond the lookup table domain.  The
    drawbacks are that the lookup table must be equi-spaced, and the interpolation is linear.
    """    
    result = numpy.zeros((image.shape[0], image.shape[1]))

    lib.internal_pointop(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         image.shape[0] * image.shape[1], 
                         lut.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         lut.shape[0],
                         ctypes.c_double(origin),
                         ctypes.c_double(increment), warnings)

    return numpy.array(result)
