#!/usr/bin/python
"""functions that interact with the C code, for handling convolutions mostly.
"""
import ctypes
import os
import glob
import numpy as np

# the wrapConv.so file can have some system information after it from the compiler, so we just find
# whatever it is called
libpath = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'wrapConv*.so'))
# print(libpath)

# load the c library
lib = ctypes.cdll.LoadLibrary(libpath[0])


def corrDn(image, filt, edges='reflect1', step=(1, 1), start=(0, 0), stop=None):
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
    image = image.copy().astype(float)
    filt = filt.copy().astype(float)

    if edges not in ['circular', 'reflect1', 'reflect2', 'repeat', 'zero', 'extend',
                     'dont-compute']:
        raise Exception("Don't know how to do convolution with edges %s!" % edges)

    if filt.ndim == 1:
        filt = filt.reshape(1, -1)

    if stop is None:
        stop = (image.shape[0], image.shape[1])

    rxsz = len(range(start[0], stop[0], step[0]))
    rysz = len(range(start[1], stop[1], step[1]))
    result = np.zeros((rxsz, rysz))

    if edges == 'circular':
        lib.internal_wrap_reduce(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 image.shape[1], image.shape[0],
                                 filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 filt.shape[1], filt.shape[0],
                                 start[1], step[1], stop[1], start[0], step[0],
                                 stop[0],
                                 result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    else:
        tmp = np.zeros((filt.shape[0], filt.shape[1]))
        lib.internal_reduce(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            image.shape[1], image.shape[0],
                            filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            filt.shape[1], filt.shape[0],
                            start[1], step[1], stop[1], start[0], step[0],
                            stop[0],
                            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            edges.encode('ascii'))

    return result


def upConv(image, filt, edges='reflect1', step=(1, 1), start=(0, 0), stop=None):
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

    NOTE: this operation corresponds to multiplication of a signal vector by a matrix whose columns
    contain copies of the time-reversed (or space-reversed) FILT shifted by multiples of STEP.  See
    corrDn.m for the operation corresponding to the transpose of this matrix.

    WARNING: if both the image and filter are 1d, they must be 1d in the same dimension. E.g., if
    image.shape is (1, 36), then filt.shape must be (1, 5) and NOT (5, 1). If they're both 1d and
    1d in different dimensions, then this may encounter a segfault. I've not been able to find a
    way to avoid that within this function (simply reshaping it does not work).
    """
    image = image.copy().astype(float)
    filt = filt.copy().astype(float)

    if filt.ndim == 1:
        filt = filt.reshape(1, -1)

    if edges not in ['circular', 'reflect1', 'reflect2', 'repeat', 'zero', 'extend',
                     'dont-compute']:
        raise Exception("Don't know how to do convolution with edges %s!" % edges)

    # from upConv.c, the c code that gets compiled in the matlab version: upConv has a bug for
    # even-length kernels when using the reflect1, extend, or repeat edge-handlers
    if ((edges in ["reflect1", "extend", "repeat"]) and
            (filt.shape[0] % 2 == 0 or filt.shape[1] % 2 == 0)):
        if filt.shape[1] == 1:
            filt = np.append(filt, 0.0)
            filt = np.reshape(filt, (len(filt), 1))
        elif filt.shape[0] == 1:
            filt = np.append(filt, 0.0)
            filt = np.reshape(filt, (1, len(filt)))
        else:
            raise Exception('Even sized 2D filters not yet supported by upConv.')

    if stop is None:
        stop = [imshape_d * step_d for imshape_d, step_d in zip(image.shape, step)]

    result = np.zeros((stop[1], stop[0]))

    temp = np.zeros((filt.shape[1], filt.shape[0]))

    if edges == 'circular':
        lib.internal_wrap_expand(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 filt.shape[1], filt.shape[0], start[1],
                                 step[1], stop[1], start[0], step[0], stop[0],
                                 result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 stop[1], stop[0])
    else:
        lib.internal_expand(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            temp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            filt.shape[1], filt.shape[0], start[1], step[1],
                            stop[1], start[0], step[0], stop[0],
                            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            stop[1], stop[0], edges.encode('ascii'))
    result = np.reshape(result, stop)

    return result


def pointOp(image, lut, origin, increment, warnings):
    """Apply a point operation, specified by lookup table `lut`, to image `im`.

    `lut` must be a row or column vector, and is assumed to contain (equi-spaced) samples of the
    function.

    `origin` specifies the abscissa associated with the first sample, and `increment` specifies the
    spacing between samples.

    Between-sample values are estimated via linear interpolation.  If `warnings` is non-zero, the
    function prints a warning whenever the lookup table is extrapolated.

    This function is very fast and allows extrapolation beyond the lookup table domain.  The
    drawbacks are that the lookup table must be equi-spaced, and the interpolation is linear.
    """
    result = np.empty_like(image)

    lib.internal_pointop(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         image.shape[0] * image.shape[1],
                         lut.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         lut.shape[0],
                         ctypes.c_double(origin),
                         ctypes.c_double(increment), warnings)

    return np.array(result)
