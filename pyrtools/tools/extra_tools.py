#!/usr/bin/python
"""variety of (non-display) image utilities
"""
import numpy as np
from ..pyramids.filters import named_filter
from ..pyramids.c.wrapper import corrDn, upConv


def _init_filt(filt):
    if isinstance(filt, str):
        filt = named_filter(filt)
    else:
        filt = np.array(filt)
    return filt / filt.sum()


def blur(image, n_levels=1, filt='binom5'):
    '''blur an image by filtering and downsampling then by upsampling and filtering

    Blur an image, by filtering and downsampling N_LEVELS times (default=1), followed by upsampling
    and filtering LEVELS times.  The blurring is done with filter kernel specified by FILT (default
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

    filt = _init_filt(filt)

    if n_levels > 0:
        if len(image.shape) == 1 or image.shape[0] == 1 or image.shape[1] == 1:
            # 1D image
            if len(filt) == 2 and (np.asarray(filt.shape) != 1).any():
                raise Exception('Error: can not apply 2D filterer to 1D signal')

            imIn = corrDn(image, filt, 'reflect1', len(image))
            out = blur(imIn, n_levels-1, filt)
            res = upConv(out, filt, 'reflect1', len(image), [0, 0], len(image))
            return res
        elif len(filt.shape) == 1 or filt.shape[0] == 1 or filt.shape[1] == 1:
            # 2D image 1D filter
            imIn = corrDn(image, filt, 'reflect1', [2, 1])
            imIn = corrDn(imIn, filt.T, 'reflect1', [1, 2])
            out = blur(imIn, n_levels-1, filt)
            res = upConv(out, filt.T, 'reflect1', [1, 2], [0, 0],
                         [out.shape[0], image.shape[1]])
            res = upConv(res, filt, 'reflect1', [2, 1], [0, 0],
                         image.shape)
            return res
        else:
            # 2D image 2D filter
            imIn = corrDn(image, filt, 'reflect1', [2, 2])
            out = blur(imIn, n_levels-1, filt)
            res = upConv(out, filt, 'reflect1', [2, 2], [0, 0],
                         image.shape)
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

    filt = _init_filt(filt)

    if n_levels > 1:
        image = blurDn(image, n_levels-1, filt)

    if n_levels >= 1:
        if len(image.shape) == 1 or image.shape[0] == 1 or image.shape[1] == 1:
            # 1D image
            if len(filt.shape) > 1 and (filt.shape[1] != 1 and filt.shape[2] != 1):
                # >1D filter
                raise Exception('Error: Cannot apply 2D filter to 1D signal')

            # orient filter and image correctly
            if image.shape[0] == 1:
                if len(filt.shape) == 1 or filt.shape[1] == 1:
                    filt = filt.T
            else:
                if filt.shape[0] == 1:
                    filt = filt.T

            res = corrDn(image=image, filt=filt, step=(2, 2))
            if len(image.shape) == 1 or image.shape[1] == 1:
                res = np.reshape(res, (np.ceil(image.shape[0]/2.0).astype(int), 1))
            else:
                res = np.reshape(res, (1, np.ceil(image.shape[1]/2.0).astype(int)))
        elif len(filt.shape) == 1 or filt.shape[0] == 1 or filt.shape[1] == 1:
            # 2D image and 1D filter
            res = corrDn(image=image, filt=filt.T, step=(2, 1))
            res = corrDn(image=res, filt=filt, step=(1, 2))

        else:  # 2D image and 2D filterer
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

    if isinstance(filt, str):
        filt = named_filter(filt)
    # print(filt, n_levels)

    if n_levels > 1:
        image = upBlur(image, n_levels-1, filt)

    if n_levels >= 1:
        # 1d image
        if image.shape[0] == 1 or image.shape[1] == 1:
            if image.shape[0] == 1:
                filt = filt.reshape(filt.shape[1], filt.shape[0])
                start = (1, 2)
            else:
                start = (2, 1)
            res = upConv(image, filt, 'reflect1', start)
        elif filt.shape[0] == 1 or filt.shape[1] == 1:
            if filt.shape[0] == 1:
                filt = filt.reshape(filt.shape[1], 1)
            res = upConv(image, filt, 'reflect1', [2, 1])
            res = upConv(res, filt.T, 'reflect1', [1, 2])
        else:
            res = upConv(image, filt, 'reflect1', [2, 2])
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
    gp = np.array([0.037659,  0.249153, 0.426375, 0.249153, 0.037659]).reshape(5, 1)
    gd = np.array([-0.109604, -0.276691, 0.000000, 0.276691, 0.109604]).reshape(5, 1)

    dx = corrDn(corrDn(image, gp, edge_type), gd.T, edge_type)
    dy = corrDn(corrDn(image, gd, edge_type), gp.T, edge_type)

    return (dx, dy)
