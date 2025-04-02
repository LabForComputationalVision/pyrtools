#!/usr/bin/python
"""functions that interact with the C code, for handling convolutions mostly.
"""
import ctypes
import warnings
import os
import glob
import numpy as np

# the wrapConv.so file can have some system information after it from the compiler, so we just find
# whatever it is called
lib_folder = os.path.dirname(os.path.realpath(__file__))
libpath = glob.glob(os.path.join(lib_folder, 'wrapConv*.so')) + \
    glob.glob(os.path.join(lib_folder, 'wrapConv.*.pyd'))
# print(libpath)

# load the c library
if len(libpath) > 0:
    lib = ctypes.cdll.LoadLibrary(libpath[0])
else:
    warnings.warn("Can't load in C code, something went wrong in your install!")


def corrDn(image, filt, edge_type='reflect1', step=(1, 1), start=(0, 0), stop=None):
    """Compute correlation of image with filt, followed by downsampling.

    These arguments should be 1D or 2D arrays, and image must be larger (in both dimensions) than
    filt.  The origin of filt is assumed to be floor(size(filt)/2)+1.

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

    Arguments
    ---------
    image : `array_like`
        1d or 2d array containing the image to correlate and downsample.
    filt : `array_like`
        1d or 2d array containing the filter to use for correlation and downsampling.
    edge_type : {'circular', 'reflect1', 'reflect2', 'repeat', 'zero', 'extend', 'dont-compute'}
        Specifies how to handle edges. Options are:

        * `'circular'` - circular convolution
        * `'reflect1'` - reflect about the edge pixels
        * `'reflect2'` - reflect, doubling the edge pixels
        * `'repeat'` - repeat the edge pixels
        * `'zero'` - assume values of zero outside image boundary
        * `'extend'` - reflect and invert
        * `'dont-compute'` - zero output when filter overhangs imput boundaries.
    step : `tuple`
        2-tuple (y, x) which determines the downsampling factor
    start : `tuple`
        2-tuple which specifies the start of the window over which we perform the convolution
    start : `tuple` or None
        2-tuple which specifies the end of the window over which we perform the convolution. If
        None, perform convolution over the whole image

    Returns
    -------
    result : `np.array`
        the correlated and downsampled array

    """
    image = image.copy().astype(float)
    filt = filt.copy().astype(float)

    if image.shape[0] < filt.shape[0] or image.shape[1] < filt.shape[1]:
        raise Exception("Signal smaller than filter in corresponding dimension: ", image.shape, filt.shape, " see parse filter")

    if edge_type not in ['circular', 'reflect1', 'reflect2', 'repeat', 'zero', 'extend', 'dont-compute']:
        raise Exception("Don't know how to do convolution with edge_type %s!" % edge_type)

    if filt.ndim == 1:
        filt = filt.reshape(1, -1)

    if stop is None:
        stop = (image.shape[0], image.shape[1])

    rxsz = len(range(start[0], stop[0], step[0]))
    rysz = len(range(start[1], stop[1], step[1]))
    result = np.zeros((rxsz, rysz))

    if edge_type == 'circular':
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
                            edge_type.encode('ascii'))

    return result


def upConv(image, filt, edge_type='reflect1', step=(1, 1), start=(0, 0), stop=None):
    """Upsample matrix image, followed by convolution with matrix filt.

    These arguments should be 1D or 2D matrices, and image must be larger (in both dimensions) than
    filt.  The origin of filt is assumed to be floor(size(filt)/2)+1.

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

    Arguments
    ---------
    image : `array_like`
        1d or 2d array containing the image to upsample and convolve.
    filt : `array_like`
        1d or 2d array containing the filter to use for upsampling and convolution.
    edge_type : {'circular', 'reflect1', 'reflect2', 'repeat', 'zero', 'extend', 'dont-compute'}
        Specifies how to handle edges. Options are:

        * `'circular'` - circular convolution
        * `'reflect1'` - reflect about the edge pixels
        * `'reflect2'` - reflect, doubling the edge pixels
        * `'repeat'` - repeat the edge pixels
        * `'zero'` - assume values of zero outside image boundary
        * `'extend'` - reflect and invert
        * `'dont-compute'` - zero output when filter overhangs imput boundaries.
    step : `tuple`
        2-tuple (y, x) which determines the upsampling factor
    start : `tuple`
        2-tuple which specifies the start of the window over which we perform the convolution.
    start : `tuple` or None
        2-tuple which specifies the end of the window over which we perform the convolution. If
        None, perform convolution over the whole image

    Returns
    -------
    result : `np.array`
        the correlated and downsampled array

    """
    image = image.copy().astype(float)
    filt = filt.copy().astype(float)

    if image.ndim == 1:
        image = image.reshape(-1, 1)

    image_shape = (image.shape[0] * step[0], image.shape[1] * step[1])

    if image_shape[0] < filt.shape[0] or image_shape[1] < filt.shape[1]:
        raise Exception("Signal smaller than filter in corresponding dimension: ", image_shape, filt.shape, " see parse filter")

    if edge_type not in ['circular', 'reflect1', 'reflect2', 'repeat', 'zero', 'extend',
                         'dont-compute']:
        raise Exception("Don't know how to do convolution with edge_type %s!" % edge_type)

    # from upConv.c, the c code that gets compiled in the matlab version: upConv has a bug for
    # even-length kernels when using the reflect1, extend, or repeat edge-handlers
    if ((edge_type in ["reflect1", "extend", "repeat"]) and
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

    if edge_type == 'circular':
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
                            stop[1], stop[0], edge_type.encode('ascii'))
    result = np.reshape(result, stop)

    return result


def pointOp(image, lut, origin, increment, warnings=False):
    """Apply a point operation, specified by lookup table `lut`, to `image`

    This function is very fast and allows extrapolation beyond the lookup table domain.  The
    drawbacks are that the lookup table must be equi-spaced, and the interpolation is linear.

    Arguments
    ---------
    image : `array_like`
        1d or 2d array
    lut : `array_like`
        a row or column vector, assumed to contain (equi-spaced) samples of the function.
    origin : `float`
        specifies the abscissa associated with the first sample
    increment : `float`
        specifies the spacing between samples.
    warnings : `bool`
        whether to print a warning whenever the lookup table is extrapolated

    """
    result = np.empty_like(image)
    # this way we can use python booleans when calling
    if warnings:
        warnings = 1
    else:
        warnings = 0
    lib.internal_pointop(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         image.shape[0] * image.shape[1],
                         lut.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         lut.shape[0],
                         ctypes.c_double(origin),
                         ctypes.c_double(increment), warnings)

    return np.asarray(result)
