import numpy as np
from ..pyramids.filters import parse_filter
from ..pyramids.c.wrapper import corrDn, upConv
import scipy.signal


def blur(image, n_levels=1, filt='binom5'):
    '''blur an image by filtering-downsampling and then upsampling-filtering

    Blur an image, by filtering and downsampling `n_levels` times (default=1), followed by upsampling
    and filtering `n_levels` times.  The blurring is done with filter kernel specified by `filt` (default
    = 'binom5'), which can be a string (to be passed to named_filter), a vector (applied separably
    as a 1D convolution kernel in X and Y), or a matrix (applied as a 2D convolution kernel).  The
    downsampling is always by 2 in each direction.

    This differs from blurDn in that here we upsample afterwards.

    Arguments
    ---------
    image : `array_like`
        1d or 2d image to blur
    n_levels : `int`
        the number of times to filter and downsample. the higher this is, the more blurred the
        resulting image will be
    filt : {`array_like`, 'binomN', 'haar', 'qmf8', 'qmf12', 'qmf16', 'daub2', 'daub3', 'daub4',
            'qmf5', 'qmf9', 'qmf13'}
        filter to use for filtering image. If array_like, can be 1d or 2d. All scaled so L-1 norm
        is 1.0

        * `'binomN'` - binomial coefficient filter of order N-1
        * `'haar'` - Haar wavelet
        * `'qmf8'`, `'qmf12'`, `'qmf16'` - Symmetric Quadrature Mirror Filters [1]_
        * `'daub2'`, `'daub3'`, `'daub4'` - Daubechies wavelet [2]_
        * `'qmf5'`, `'qmf9'`, `'qmf13'`   - Symmetric Quadrature Mirror Filters [3]_, [4]_

    Returns
    -------
    image : `array_like`
        the blurred image

    References
    ----------
    .. [1] J D Johnston, "A filter family designed for use in quadrature mirror filter banks",
       Proc. ICASSP, pp 291-294, 1980.
    .. [2] I Daubechies, "Orthonormal bases of compactly supported wavelets", Commun. Pure Appl.
       Math, vol. 42, pp 909-996, 1988.
    .. [3] E P Simoncelli,  "Orthogonal sub-band image transforms", PhD Thesis, MIT Dept. of Elec.
       Eng. and Comp. Sci. May 1988. Also available as: MIT Media Laboratory Vision and Modeling
       Technical Report #100.
    .. [4] E P Simoncelli and E H Adelson, "Subband image coding", Subband Transforms, chapter 4,
       ed. John W Woods, Kluwer Academic Publishers,  Norwell, MA, 1990, pp 143--192.
    '''

    if image.ndim == 1:
        image = image.reshape(-1, 1)

    filt = parse_filter(filt)

    if n_levels > 0:
        if image.shape[1] == 1:
            # 1D image [M, 1] 1D filter [N, 1]
            imIn = corrDn(image=image, filt=filt, step=(2, 1))
            out = blur(imIn, n_levels-1, filt)
            res = upConv(image=out, filt=filt, step=(2, 1), stop=image.shape)
            return res

        elif image.shape[1] == 1:
            # 1D image [1, M] 1D filter [N, 1]
            imIn = corrDn(image=image, filt=filt.T, step=(1, 2))
            out = blur(imIn, n_levels-1, filt)
            res = upConv(image=out, filt=filt.T, step=(1, 2), stop=image.shape)
            return res

        elif filt.shape[1] == 1:
            # 2D image 1D filter [N, 1]
            imIn = corrDn(image=image, filt=filt, step=(2, 1))
            imIn = corrDn(image=imIn, filt=filt.T, step=(1, 2))
            out = blur(imIn, n_levels-1, filt)
            res = upConv(image=out, filt=filt.T, step=(1, 2), start=(0, 0), stop=[out.shape[0], image.shape[1]])
            res = upConv(image=res, filt=filt, step=(2, 1), start=(0, 0), stop=image.shape)
            return res

        else:
            # 2D image 2D filter
            imIn = corrDn(image=image, filt=filt, step=(2, 2))
            out = blur(imIn, n_levels-1, filt)
            res = upConv(image=out, filt=filt, step=(2, 2), stop=image.shape)
            return res

    else:
        return image


def blurDn(image, n_levels=1, filt='binom5'):
    '''blur and downsample an image

    Blur and downsample an image.  The blurring is done with filter kernel specified by FILT
    (default = 'binom5'), which can be a string (to be passed to named_filter), a vector (applied
    separably as a 1D convolution kernel in X and Y), or a matrix (applied as a 2D convolution
    kernel).  The downsampling is always by 2 in each direction.

    The procedure is applied recursively `n_levels` times (default=1).

    This differs from blur in that we do NOT upsample afterwards.

    Arguments
    ---------
    image : `array_like`
        1d or 2d image to blur and downsample
    n_levels : `int`
        the number of times to filter and downsample. the higher this is, the blurrier and smaller
        the resulting image will be
    filt : {`array_like`, 'binomN', 'haar', 'qmf8', 'qmf12', 'qmf16', 'daub2', 'daub3', 'daub4',
            'qmf5', 'qmf9', 'qmf13'}
        filter to use for filtering image. If array_like, can be 1d or 2d. All scaled so L-1 norm
        is 1.0

        * `'binomN'` - binomial coefficient filter of order N-1
        * `'haar'` - Haar wavelet
        * `'qmf8'`, `'qmf12'`, `'qmf16'` - Symmetric Quadrature Mirror Filters [1]_
        * `'daub2'`, `'daub3'`, `'daub4'` - Daubechies wavelet [2]_
        * `'qmf5'`, `'qmf9'`, `'qmf13'`   - Symmetric Quadrature Mirror Filters [3]_, [4]_

    Returns
    -------
    image : `array_like`
        the blurred and downsampled image

    References
    ----------
    .. [1] J D Johnston, "A filter family designed for use in quadrature mirror filter banks",
       Proc. ICASSP, pp 291-294, 1980.
    .. [2] I Daubechies, "Orthonormal bases of compactly supported wavelets", Commun. Pure Appl.
       Math, vol. 42, pp 909-996, 1988.
    .. [3] E P Simoncelli,  "Orthogonal sub-band image transforms", PhD Thesis, MIT Dept. of Elec.
       Eng. and Comp. Sci. May 1988. Also available as: MIT Media Laboratory Vision and Modeling
       Technical Report #100.
    .. [4] E P Simoncelli and E H Adelson, "Subband image coding", Subband Transforms, chapter 4,
       ed. John W Woods, Kluwer Academic Publishers,  Norwell, MA, 1990, pp 143--192.
    '''

    if image.ndim == 1:
        image = image.reshape(-1, 1)

    filt = parse_filter(filt)

    if n_levels > 1:
        image = blurDn(image, n_levels-1, filt)

    if n_levels >= 1:
        if image.shape[1] == 1:
            # 1D image [M, 1] and 1D filter [N, 1]
            res = corrDn(image=image, filt=filt, step=(2, 1))

        elif image.shape[0] == 1:
            # 1D image [1, M] and 1D filter [N, 1]
            res = corrDn(image=image, filt=filt.T, step=(1, 2))

        elif filt.shape[1] == 1:
            # 2D image and 1D filter [N, 1]
            res = corrDn(image=image, filt=filt, step=(2, 1))
            res = corrDn(image=res, filt=filt.T, step=(1, 2))

        else:
            # 2D image and 2D filter
            res = corrDn(image=image, filt=filt, step=(2, 2))

    else:
        res = image

    return res


def upBlur(image, n_levels=1, filt='binom5'):
    '''upsample and blur an image.

    Upsample and blur an image.  The blurring is done with filter kernel specified by FILT (default
    = 'binom5'), which can be a string (to be passed to named_filter), a vector (applied separably
    as a 1D convolution kernel in X and Y), or a matrix (applied as a 2D convolution kernel).  The
    downsampling is always by 2 in each direction.

    The procedure is applied recursively n_levels times (default=1).

    Arguments
    ---------
    image : `array_like`
        1d or 2d image to upsample and blur
    n_levels : `int`
        the number of times to filter and downsample. the higher this is, the blurrier and larger
        the resulting image will be
    filt : {`array_like`, 'binomN', 'haar', 'qmf8', 'qmf12', 'qmf16', 'daub2', 'daub3', 'daub4',
            'qmf5', 'qmf9', 'qmf13'}
        filter to use for filtering image. If array_like, can be 1d or 2d. All scaled so L-1 norm
        is 1.0

        * `'binomN'` - binomial coefficient filter of order N-1
        * `'haar'` - Haar wavelet
        * `'qmf8'`, `'qmf12'`, `'qmf16'` - Symmetric Quadrature Mirror Filters [1]_
        * `'daub2'`, `'daub3'`, `'daub4'` - Daubechies wavelet [2]_
        * `'qmf5'`, `'qmf9'`, `'qmf13'`   - Symmetric Quadrature Mirror Filters [3]_, [4]_

    Returns
    -------
    image : `array_like`
        the upsampled and blurred image

    References
    ----------
    .. [1] J D Johnston, "A filter family designed for use in quadrature mirror filter banks",
       Proc. ICASSP, pp 291-294, 1980.
    .. [2] I Daubechies, "Orthonormal bases of compactly supported wavelets", Commun. Pure Appl.
       Math, vol. 42, pp 909-996, 1988.
    .. [3] E P Simoncelli,  "Orthogonal sub-band image transforms", PhD Thesis, MIT Dept. of Elec.
       Eng. and Comp. Sci. May 1988. Also available as: MIT Media Laboratory Vision and Modeling
       Technical Report #100.
    .. [4] E P Simoncelli and E H Adelson, "Subband image coding", Subband Transforms, chapter 4,
       ed. John W Woods, Kluwer Academic Publishers,  Norwell, MA, 1990, pp 143--192.

    '''

    if image.ndim == 1:
        image = image.reshape(-1, 1)

    filt = parse_filter(filt, normalize=False)

    if n_levels > 1:
        image = upBlur(image, n_levels-1, filt)

    if n_levels >= 1:
        if image.shape[1] == 1:
            # 1D image [M, 1] and 1D filter [N, 1]
            res = upConv(image=image, filt=filt, step=(2, 1))

        elif image.shape[0] == 1:
            # 1D image [1, M] and 1D filter [N, 1]
            res = upConv(image=image, filt=filt.T, step=(1, 2))

        elif filt.shape[1] == 1:
            # 2D image and 1D filter [N, 1]
            res = upConv(image=image, filt=filt, step=(2, 1))
            res = upConv(image=res, filt=filt.T, step=(1, 2))

        else:
            # 2D image and 2D filter
            res = upConv(image=image, filt=filt, step=(2, 2))

    else:
        res = image

    return res


def image_gradient(image, edge_type="dont-compute"):
    '''Compute the gradient of the image using smooth derivative filters

    Compute the gradient of the image using smooth derivative filters optimized for accurate
    direction estimation.  Coordinate system corresponds to standard pixel indexing: X axis points
    rightward.  Y axis points downward.  `edges` specify boundary handling.

    Notes
    -----
    original filters from Int'l Conf Image Processing, 1994.
    updated filters 10/2003: see Farid & Simoncelli, IEEE Trans Image
                             Processing, 13(4):496-508, April 2004.

    Arguments
    ---------
    image : `array_like`
        2d array to compute the gradients of
    edge_type : {'circular', 'reflect1', 'reflect2', 'repeat', 'zero', 'extend', 'dont-compute'}
        Specifies how to handle edges. Options are:

        * `'circular'` - circular convolution
        * `'reflect1'` - reflect about the edge pixels
        * `'reflect2'` - reflect, doubling the edge pixels
        * `'repeat'` - repeat the edge pixels
        * `'zero'` - assume values of zero outside image boundary
        * `'extend'` - reflect and invert
        * `'dont-compute'` - zero output when filter overhangs imput boundaries.

    Returns
    -------
    dx, dy : `np.array`
        the X derivative and the Y derivative

    '''

    # kernels from Farid & Simoncelli, IEEE Trans Image Processing,
    #   13(4):496-508, April 2004.
    gp = np.asarray([0.037659,  0.249153, 0.426375, 0.249153, 0.037659]).reshape(5, 1)
    gd = np.asarray([-0.109604, -0.276691, 0.000000, 0.276691, 0.109604]).reshape(5, 1)

    dx = corrDn(corrDn(image, gp, edge_type), gd.T, edge_type)
    dy = corrDn(corrDn(image, gd, edge_type), gp.T, edge_type)

    return (dx, dy)


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


# TODO: low priority

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
#         a = np.asarray(args[0])
#         b = np.asarray(args[1])
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
#         a = np.asarray(args[0])
#         b = np.asarray(args[1])
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
