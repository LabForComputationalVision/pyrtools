import numpy
import ctypes
import os

libpath = os.path.dirname(os.path.realpath(__file__))+'/../wrapConv.so'
# load the C library
lib = ctypes.cdll.LoadLibrary(libpath)


def upConv(image, filt, edges='reflect1', step=(1, 1), start=(0, 0), stop=None, result=None):
    """Upsample matrix image, followed by convolution with matrix filt.

    These arguments should be 1D or 2D matrices, and image must be larger (in both dimensions) than
    filt.  The origin of filt is assumed to be floor(size(filt)/2)+1.

    edges is a string determining boundary handling:
       'circular' - Circular convolution
       'reflect1' - Reflect about the edge pixels
       'reflect2' - Reflect, doubling the edge pixels
       'repeat'   - Repeat the edge pixels
       'zero'     - Assume values of zero outside image boundary
       'extend'   - Reflect and invert
       'dont-compute' - Zero output when filter overhangs OUTPUT boundaries

    Upsampling factors are determined by step (optional, default=(1, 1)),
    a 2-tuple (y, x).
 
    The window over which the convolution occurs is specfied by start (optional, default=(0, 0),
    and stop (optional, default = step .* (size(IM) + floor((start-1)./step))).

    result is an optional result matrix.  The convolution result will be destructively added into
    this matrix.  If this argument is passed, the result matrix will not be returned. DO NOT USE
    THIS ARGUMENT IF YOU DO NOT UNDERSTAND WHAT THIS MEANS!!
 
    NOTE: this operation corresponds to multiplication of a signal vector by a matrix whose columns
    contain copies of the time-reversed (or space-reversed) FILT shifted by multiples of STEP.  See
    corrDn.m for the operation corresponding to the transpose of this matrix.
    """
    image = image.copy()
    filt = filt.copy()
        
    if len(filt.shape) == 1:
        filt = numpy.reshape(filt, (1, len(filt)))

    # there's a segfault that occasionally happens when image is 1d and the filt is not properly
    # shaped. In order to avoid this, I think we need to reshape the filter: the filters are 1d,
    # but held in 2d arrays and typically we want the first dimension to be the non-trivial one
    # (ie, the shape is (5,1) instead of (1,5)). But (based on Gpyr, which was not having this
    # issue), it appears that when the image is 1d the filter should have the trivial dimension
    # first (ie, shape is (1,5)), so that's what we do here.

    if image.shape[0] == 1:
        filt = filt.reshape(1, max(filt.shape)).copy()

    if ( (edges != "reflect1" or edges != "extend" or edges != "repeat") and
         (filt.shape[0] % 2 == 0 or filt.shape[1] % 2 == 0) ):
        if filt.shape[1] == 1:
            filt = numpy.append(filt,0.0);
            filt = numpy.reshape(filt, (len(filt), 1))
        elif filt.shape[0] == 1:
            filt = numpy.append(filt,0.0);
            filt = numpy.reshape(filt, (1, len(filt)))
        else:
            raise Exception('Even sized 2D filters not yet supported by upConv.')

    if stop is None and result is None:
        stop = (image.shape[0]*step[0], image.shape[1]*step[1])
        stop = (stop[0], stop[1])
    elif stop is None:
        stop = (result.shape[0], result.shape[1])

    if result is None:
        result = numpy.zeros((stop[1], stop[0]))
    else:
        result = numpy.array(result.copy())

    temp = numpy.zeros((filt.shape[1], filt.shape[0]))

    if edges == 'circular':
        lib.internal_wrap_expand(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 filt.shape[1], filt.shape[0], start[1], 
                                 step[1], stop[1], start[0], step[0], stop[0],
                                 result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 stop[1], stop[0])
        result = result.T
    else:
        lib.internal_expand(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            temp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            filt.shape[1], filt.shape[0], start[1], step[1],
                            stop[1], start[0], step[0], stop[0],
                            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            stop[1], stop[0], edges)
        result = numpy.reshape(result, stop)

    return result
