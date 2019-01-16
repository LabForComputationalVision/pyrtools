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

    Blur an image, by filtering and downsampling N_LEVELS times
    (default=1), followed by upsampling and filtering LEVELS times.  The
    blurring is done with filter kernel specified by FILT (default =
    'binom5'), which can be a string (to be passed to named_filter), a
    vector (applied separably as a 1D convolution kernel in X and Y), or
    a matrix (applied as a 2D convolution kernel).  The downsampling is
    always by 2 in each direction.

    This differs from blurDn in that here we upsample afterwards.

    Eero Simoncelli, 3/04.  Python port by Rob Young, 10/15  '''

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

    Blur and downsample an image.  The blurring is done with filter
    kernel specified by FILT (default = 'binom5'), which can be a string
    (to be passed to named_filter), a vector (applied separably as a 1D
    convolution kernel in X and Y), or a matrix (applied as a 2D
    convolution kernel).  The downsampling is always by 2 in each
    direction.

    The procedure is applied recursively LEVELS times (default=1).

    This differs from blur in that we do NOT upsample afterwards.

    Eero Simoncelli, 3/97.  Ported to python by Rob Young 4/14'''

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

    Upsample and blur an image.  The blurring is done with filter
    kernel specified by FILT (default = 'binom5'), which can be a string
    (to be passed to named_filter), a vector (applied separably as a 1D
    convolution kernel in X and Y), or a matrix (applied as a 2D
    convolution kernel).  The downsampling is always by 2 in each
    direction.

    The procedure is applied recursively LEVELS times (default=1).

    Eero Simoncelli, 4/97. Python port by Rob Young, 10/15.   '''

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


def image_gradient(im_array, edges="dont-compute"):
    '''Compute the gradient of the image using smooth derivative filters

    Compute the gradient of the image using smooth derivative filters
    optimized for accurate direction estimation.  Coordinate system
    corresponds to standard pixel indexing: X axis points rightward.  Y
    axis points downward.  EDGES specify boundary handling (see corrDn
    for options).

    Returns [dx, dy]: the X derivative and the Y derivative

    edges is a string determining boundary handling:
      'circular' - Circular convolution
      'reflect1' - Reflect about the edge pixels
      'reflect2' - Reflect, doubling the edge pixels
      'repeat'   - Repeat the edge pixels
      'zero'     - Assume values of zero outside image boundary
      'extend'   - Reflect and invert (continuous values and derivs)
      'dont-compute' - Zero output when filter overhangs input boundaries

    EPS, 1997.
    original filters from Int'l Conf Image Processing, 1994.
    updated filters 10/2003: see Farid & Simoncelli, IEEE Trans Image
                             Processing, 13(4):496-508, April 2004.
    Incorporated into matlabPyrTools 10/2004.
    Python port by Rob Young, 10/15  '''

    # kernels from Farid & Simoncelli, IEEE Trans Image Processing,
    #   13(4):496-508, April 2004.
    gp = np.array([0.037659,  0.249153, 0.426375, 0.249153, 0.037659]).reshape(5, 1)
    gd = np.array([-0.109604, -0.276691, 0.000000, 0.276691, 0.109604]).reshape(5, 1)

    dx = corrDn(corrDn(im_array, gp, edges), gd.T, edges)
    dy = corrDn(corrDn(im_array, gd, edges), gp.T, edges)

    return (dx, dy)
