import numpy
import ctypes
import os

libpath = os.path.dirname(os.path.realpath(__file__))+'/../wrapConv.so'
# load the C library
lib = ctypes.cdll.LoadLibrary(libpath)


def corrDn(image, filt, edges='reflect1', step=(1, 1), start=(0, 0), stop=None, result=None):
    """Compute correlation of matrices image with `filt, followed by downsampling.  

    These arguments should be 1D or 2D matrices, and image must be larger (in both dimensions) than
    filt.  The origin of filt is assumed to be floor(size(filt)/2)+1.
 
    edges is a string determining boundary handling:
      'circular' - Circular convolution
      'reflect1' - Reflect about the edge pixels
      'reflect2' - Reflect, doubling the edge pixels
      'repeat'   - Repeat the edge pixels
      'zero'     - Assume values of zero outside image boundary
      'extend'   - Reflect and invert (continuous values and derivs)
      'dont-compute' - Zero output when filter overhangs input boundaries

    Downsampling factors are determined by step (optional, default=(1, 1)), which should be a
    2-tuple (y, x).
 
    The window over which the convolution occurs is specfied by start (optional, default=(0,0), and
    stop (optional, default=size(image)).
 
    NOTE: this operation corresponds to multiplication of a signal vector by a matrix whose rows
    contain copies of the filt shifted by multiples of step.  See `upConv` for the operation
    corresponding to the transpose of this matrix.

    WARNING: if both the image and filter are 1d, they must be 1d in the same dimension. E.g., if
    image.shape is (1, 36), then filt.shape must be (1, 5) and NOT (5, 1). If they're both 1d and
    1d in different dimensions, then this may encounter a segfault. I've not been able to find a
    way to avoid that within this function (simply reshaping it does not work).
    """
    image = image.copy()
    filt = filt.copy()

    if len(filt.shape) == 1:
        filt = numpy.reshape(filt, (1, len(filt)))

    if stop is None:
        stop = (image.shape[0], image.shape[1])

    if result is None:
        rxsz = len(range(start[0], stop[0], step[0]))
        rysz = len(range(start[1], stop[1], step[1]))
        result = numpy.zeros((rxsz, rysz))
    else:
        result = numpy.array(result.copy())
        
    if edges == 'circular':
        lib.internal_wrap_reduce(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                                 image.shape[1], image.shape[0], 
                                 filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                                 filt.shape[1], filt.shape[0], 
                                 start[1], step[1], stop[1], start[0], step[0], 
                                 stop[0], 
                                 result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    else:
        tmp = numpy.zeros((filt.shape[0], filt.shape[1]))
        lib.internal_reduce(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                            image.shape[1], image.shape[0], 
                            filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                            tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                            filt.shape[1], filt.shape[0], 
                            start[1], step[1], stop[1], start[0], step[0], 
                            stop[0], 
                            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                            edges)

    return result
