#!/usr/bin/python
"""functions that interact with the C code, for handling convolutions mostly.
"""

import ctypes
import os
import glob
import numpy as np

# the wrapConv.so file can have some system information after it from the compiler, so we just find
# whatever it is called
libpath = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'c', 'wrapConv*.so'))
# load the c library
lib = ctypes.cdll.LoadLibrary(libpath[0])


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

    if filt.ndim == 1:
        filt = filt.reshape(1,-1)

    if stop is None:
        stop = (image.shape[0], image.shape[1])

    if result is None:
        rxsz = len(range(start[0], stop[0], step[0]))
        rysz = len(range(start[1], stop[1], step[1]))
        result = np.zeros((rxsz, rysz))
    else:
        result = np.array(result.copy())

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

    WARNING: if both the image and filter are 1d, they must be 1d in the same dimension. E.g., if
    image.shape is (1, 36), then filt.shape must be (1, 5) and NOT (5, 1). If they're both 1d and
    1d in different dimensions, then this may encounter a segfault. I've not been able to find a
    way to avoid that within this function (simply reshaping it does not work).
    """
    image = image.copy()
    filt = filt.copy()

    if filt.ndim == 1:
        filt = filt.reshape(1,-1)

    # TODO: first condition is always TRUE?
    if ((edges != "reflect1" or edges != "extend" or edges != "repeat") and
            (filt.shape[0] % 2 == 0 or filt.shape[1] % 2 == 0)):
        if filt.shape[1] == 1:
            filt = np.append(filt, 0.0)
            filt = np.reshape(filt, (len(filt), 1))
        elif filt.shape[0] == 1:
            filt = np.append(filt, 0.0)
            filt = np.reshape(filt, (1, len(filt)))
        else:
            raise Exception('Even sized 2D filters not yet supported by upConv.')

    if stop is None and result is None:
        stop = [imshape_d * step_d for imshape_d, step_d in zip(image.shape, step)]
    elif stop is None:
        stop = result.shape

    if result is None:
        result = np.zeros((stop[1], stop[0]))
    else:
        result = np.array(result.copy())

    temp = np.zeros((filt.shape[1], filt.shape[0]))

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
                            stop[1], stop[0], edges.encode('ascii'))
        result = np.reshape(result, stop)

    return result


def pointOp(image, lut, origin, increment, warnings):
    """Apply a point operation, specified by lookup table `lut`, to image `im`.

    `lut` must be a row or column vector, and is assumed to contain (equi-spaced) samples of the function.
    `origin` specifies the abscissa associated with the first sample, and `increment` specifies the spacing between samples.
    Between-sample values are estimated via linear interpolation.  If `warnings` is non-zero, the function prints a warning whenever the lookup table is extrapolated.

    This function is very fast and allows extrapolation beyond the lookup table domain.  The drawbacks are that the lookup table must be equi-spaced, and the interpolation is linear.
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

#----------------------------------------------------------------
# slow scipy convolution functions
#----------------------------------------------------------------

import scipy.signal

def cconv2(*args):
    ''' RES = CCONV2(MTX1, MTX2, CTR)

        Circular convolution of two matrices.  Result will be of size of
        LARGER vector.

        The origin of the smaller matrix is assumed to be its center.
        For even dimensions, the origin is determined by the CTR (optional)
        argument:
             CTR   origin
              0     DIM/2      (default)
              1     (DIM/2)+1

        Eero Simoncelli, 6/96.  Modified 2/97.
        Python port by Rob Young, 8/15  '''

    if len(args) < 2:
        print 'Error: cconv2 requires two input matrices!'
        print 'Usage: cconv2(matrix1, matrix2, center)'
        print 'where center parameter is optional'
        return
    else:
        a = np.array(args[0])
        b = np.array(args[1])

    if len(args) == 3:
        ctr = args[2]
    else:
        ctr = 0

    if a.shape[0] >= b.shape[0] and a.shape[1] >= b.shape[1]:
        large = a
        small = b
    elif a.shape[0] <= b.shape[0] and a.shape[1] <= b.shape[1]:
        large = b
        small = a
    else:
        print 'Error: one matrix must be larger than the other in both dimensions!'
        return

    ly = large.shape[0]
    lx = large.shape[1]
    sy = small.shape[0]
    sx = small.shape[1]

    ## These values are the index of the small mtx that falls on the
    ## border pixel of the large matrix when computing the first
    ## convolution response sample:
    sy2 = np.floor((sy+ctr+1)/2.0).astype(int)
    sx2 = np.floor((sx+ctr+1)/2.0).astype(int)

    # pad
    nw = large[ly-sy+sy2:ly, lx-sx+sx2:lx]
    n = large[ly-sy+sy2:ly, :]
    ne = large[ly-sy+sy2:ly, :sx2-1]
    w = large[:, lx-sx+sx2:lx]
    c = large
    e = large[:, :sx2-1]
    sw = large[:sy2-1, lx-sx+sx2:lx]
    s = large[:sy2-1, :]
    se = large[:sy2-1, :sx2-1]

    n = np.column_stack((nw, n, ne))
    c = np.column_stack((w,large,e))
    s = np.column_stack((sw, s, se))

    clarge = np.concatenate((n, c), axis=0)
    clarge = np.concatenate((clarge, s), axis=0)

    c = scipy.signal.convolve(clarge, small, 'valid')

    return c

def rconv2(*args):
    ''' Convolution of two matrices, with boundaries handled via reflection
        about the edge pixels.  Result will be of size of LARGER matrix.

        The origin of the smaller matrix is assumed to be its center.
        For even dimensions, the origin is determined by the CTR (optional)
        argument:
             CTR   origin
              0     DIM/2      (default)
              1   (DIM/2)+1  '''

    if len(args) < 2:
        print "Error: two matrices required as input parameters"
        return

    if len(args) == 2:
        ctr = 0

    if ( args[0].shape[0] >= args[1].shape[0] and
         args[0].shape[1] >= args[1].shape[1] ):
        large = args[0]
        small = args[1]
    elif ( args[0].shape[0] <= args[1].shape[0] and
           args[0].shape[1] <= args[1].shape[1] ):
        large = args[1]
        small = args[0]
    else:
        print 'one arg must be larger than the other in both dimensions!'
        return

    ly = large.shape[0]
    lx = large.shape[1]
    sy = small.shape[0]
    sx = small.shape[1]

    ## These values are one less than the index of the small mtx that falls on
    ## the border pixel of the large matrix when computing the first
    ## convolution response sample:
    sy2 = int(np.floor((sy+ctr-1)/2))
    sx2 = int(np.floor((sx+ctr-1)/2))

    # pad with reflected copies
    nw = large[sy-sy2-1:0:-1, sx-sx2-1:0:-1]
    n = large[sy-sy2-1:0:-1, :]
    ne = large[sy-sy2-1:0:-1, lx-2:lx-sx2-2:-1]
    w = large[:, sx-sx2-1:0:-1]
    e = large[:, lx-2:lx-sx2-2:-1]
    sw = large[ly-2:ly-sy2-2:-1, sx-sx2-1:0:-1]
    s = large[ly-2:ly-sy2-2:-1, :]
    se = large[ly-2:ly-sy2-2:-1, lx-2:lx-sx2-2:-1]

    n = np.column_stack((nw, n, ne))
    c = np.column_stack((w,large,e))
    s = np.column_stack((sw, s, se))

    clarge = np.concatenate((n, c), axis=0)
    clarge = np.concatenate((clarge, s), axis=0)

    return scipy.signal.convolve(clarge, small, 'valid')

def zconv2(*args):
    ''' RES = ZCONV2(MTX1, MTX2, CTR)

        Convolution of two matrices, with boundaries handled as if the larger
        mtx lies in a sea of zeros. Result will be of size of LARGER vector.

        The origin of the smaller matrix is assumed to be its center.
        For even dimensions, the origin is determined by the CTR (optional)
        argument:
             CTR   origin
              0     DIM/2      (default)
              1     (DIM/2)+1  (behaves like conv2(mtx1,mtx2,'same'))

        Eero Simoncelli, 2/97.  Python port by Rob Young, 10/15.  '''

    # REQUIRED ARGUMENTS
    #----------------------------------------------------------------

    if len(args) < 2 or len(args) > 3:
        print 'Usage: zconv2(matrix1, matrix2, center)'
        print 'first two input parameters are required'
        return
    else:
        a = np.array(args[0])
        b = np.array(args[1])

    # OPTIONAL ARGUMENT
    #----------------------------------------------------------------

    if len(args) == 3:
        ctr = args[2]
    else:
        ctr = 0

    #----------------------------------------------------------------

    if (a.shape[0] >= b.shape[0]) and (a.shape[1] >= b.shape[1]):
        large = a
        small = b
    elif (a.shape[0] <= b.shape[0]) and (a.shape[1] <= b.shape[1]):
        large = b
        small = a
    else:
        print 'Error: one arg must be larger than the other in both dimensions!'
        return

    ly = large.shape[0]
    lx = large.shape[1]
    sy = small.shape[0]
    sx = small.shape[1]

    ## These values are the index of the small matrix that falls on the
    ## border pixel of the large matrix when computing the first
    ## convolution response sample:
    sy2 = np.floor((sy+ctr+1)/2.0).astype(int)-1
    sx2 = np.floor((sx+ctr+1)/2.0).astype(int)-1

    clarge = scipy.signal.convolve(large, small, 'full')

    c = clarge[sy2:ly+sy2, sx2:lx+sx2]

    return c
