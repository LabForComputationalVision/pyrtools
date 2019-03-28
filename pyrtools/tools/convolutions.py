import numpy as np
import scipy.signal

# ----------------------------------------------------------------
# Below are (slow) scipy convolution functions
# they are intended for comparison purpose only
# the c code is prefered and used throughout this package
# ----------------------------------------------------------------


def rconv2(mtx1, mtx2, ctr=0):
    '''Convolution of two matrices, with boundaries handled via reflection about the edge pixels.

    Result will be of size of LARGER matrix.

    The origin of the smaller matrix is assumed to be its center.
    For even dimensions, the origin is determined by the CTR (optional)
    argument:
         CTR   origin
          0     DIM/2      (default)
          1   (DIM/2)+1

    In general, you should not use this function, since it will be slow. Instead, use `upConv` or
    `corrDn`, which use the C code and so are much faster.

    '''

    if (mtx1.shape[0] >= mtx2.shape[0] and mtx1.shape[1] >= mtx2.shape[1]):
        large = mtx1
        small = mtx2
    elif (mtx1.shape[0] <= mtx2.shape[0] and mtx1.shape[1] <= mtx2.shape[1]):
        large = mtx2
        small = mtx1
    else:
        print('one matrix must be larger than the other in both dimensions!')
        return

    ly = large.shape[0]
    lx = large.shape[1]
    sy = small.shape[0]
    sx = small.shape[1]

    # These values are one less than the index of the small mtx that falls on
    # the border pixel of the large matrix when computing the first
    # convolution response sample:
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
    c = np.column_stack((w, large, e))
    s = np.column_stack((sw, s, se))

    clarge = np.concatenate((n, c), axis=0)
    clarge = np.concatenate((clarge, s), axis=0)

    return scipy.signal.convolve(clarge, small, 'valid')


# def cconv2(mtx1, mtx2, ctr=0):
#     '''Circular convolution of two matrices.  Result will be of size of
#     LARGER vector.
#
#     The origin of the smaller matrix is assumed to be its center.
#     For even dimensions, the origin is determined by the CTR (optional)
#     argument:
#          CTR   origin
#           0     DIM/2      (default)
#           1     (DIM/2)+1
#
#     Eero Simoncelli, 6/96.  Modified 2/97.
#     Python port by Rob Young, 8/15
#     '''
#
#     if len(args) < 2:
#         print 'Error: cconv2 requires two input matrices!'
#         print 'Usage: cconv2(matrix1, matrix2, center)'
#         print 'where center parameter is optional'
#         return
#     else:
#         a = np.array(args[0])
#         b = np.array(args[1])
#
#     if len(args) == 3:
#         ctr = args[2]
#     else:
#         ctr = 0
#
#     if a.shape[0] >= b.shape[0] and a.shape[1] >= b.shape[1]:
#         large = a
#         small = b
#     elif a.shape[0] <= b.shape[0] and a.shape[1] <= b.shape[1]:
#         large = b
#         small = a
#     else:
#         print 'Error: one matrix must be larger than the other in both dimensions!'
#         return
#
#     ly = large.shape[0]
#     lx = large.shape[1]
#     sy = small.shape[0]
#     sx = small.shape[1]
#
#     ## These values are the index of the small mtx that falls on the
#     ## border pixel of the large matrix when computing the first
#     ## convolution response sample:
#     sy2 = np.floor((sy+ctr+1)/2.0).astype(int)
#     sx2 = np.floor((sx+ctr+1)/2.0).astype(int)
#
#     # pad
#     nw = large[ly-sy+sy2:ly, lx-sx+sx2:lx]
#     n = large[ly-sy+sy2:ly, :]
#     ne = large[ly-sy+sy2:ly, :sx2-1]
#     w = large[:, lx-sx+sx2:lx]
#     c = large
#     e = large[:, :sx2-1]
#     sw = large[:sy2-1, lx-sx+sx2:lx]
#     s = large[:sy2-1, :]
#     se = large[:sy2-1, :sx2-1]
#
#     n = np.column_stack((nw, n, ne))
#     c = np.column_stack((w,large,e))
#     s = np.column_stack((sw, s, se))
#
#     clarge = np.concatenate((n, c), axis=0)
#     clarge = np.concatenate((clarge, s), axis=0)
#
#     c = scipy.signal.convolve(clarge, small, 'valid')
#
#     return c


# def zconv2(mtx1, mtx2, ctr=0):
#     ''' RES = ZCONV2(MTX1, MTX2, CTR)
#
#         Convolution of two matrices, with boundaries handled as if the larger
#         mtx lies in a sea of zeros. Result will be of size of LARGER vector.
#
#         The origin of the smaller matrix is assumed to be its center.
#         For even dimensions, the origin is determined by the CTR (optional)
#         argument:
#              CTR   origin
#               0     DIM/2      (default)
#               1     (DIM/2)+1  (behaves like conv2(mtx1,mtx2,'same'))
#
#         Eero Simoncelli, 2/97.  Python port by Rob Young, 10/15.  '''
#
#     # REQUIRED ARGUMENTS
#     #----------------------------------------------------------------
#
#     if len(args) < 2 or len(args) > 3:
#         print 'Usage: zconv2(matrix1, matrix2, center)'
#         print 'first two input parameters are required'
#         return
#     else:
#         a = np.array(args[0])
#         b = np.array(args[1])
#
#     # OPTIONAL ARGUMENT
#     #----------------------------------------------------------------
#
#     if len(args) == 3:
#         ctr = args[2]
#     else:
#         ctr = 0
#
#     #----------------------------------------------------------------
#
#     if (a.shape[0] >= b.shape[0]) and (a.shape[1] >= b.shape[1]):
#         large = a
#         small = b
#     elif (a.shape[0] <= b.shape[0]) and (a.shape[1] <= b.shape[1]):
#         large = b
#         small = a
#     else:
#         print 'Error: one arg must be larger than the other in both dimensions!'
#         return
#
#     ly = large.shape[0]
#     lx = large.shape[1]
#     sy = small.shape[0]
#     sx = small.shape[1]
#
#     ## These values are the index of the small matrix that falls on the
#     ## border pixel of the large matrix when computing the first
#     ## convolution response sample:
#     sy2 = np.floor((sy+ctr+1)/2.0).astype(int)-1
#     sx2 = np.floor((sx+ctr+1)/2.0).astype(int)-1
#
#     clarge = scipy.signal.convolve(large, small, 'full')
#
#     c = clarge[sy2:ly+sy2, sx2:lx+sx2]
#
#     return c
