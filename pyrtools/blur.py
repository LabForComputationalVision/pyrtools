import numpy
from .namedFilter import namedFilter
from .corrDn import corrDn
from .upConv import upConv

def blur(*args):
    ''' RES = blur(IM, LEVELS, FILT)
    
        Blur an image, by filtering and downsampling LEVELS times
        (default=1), followed by upsampling and filtering LEVELS times.  The
        blurring is done with filter kernel specified by FILT (default =
        'binom5'), which can be a string (to be passed to namedFilter), a
        vector (applied separably as a 1D convolution kernel in X and Y), or
        a matrix (applied as a 2D convolution kernel).  The downsampling is
        always by 2 in each direction.
    
        Eero Simoncelli, 3/04.  Python port by Rob Young, 10/15  '''

    # REQUIRED ARG:
    if len(args) == 0:
        print("blur(IM, LEVELS, FILT)")
        print("first argument is required")
        exit(1)
    else:
        im = numpy.array(args[0])

    # OPTIONAL ARGS:
    if len(args) > 1:
        nlevs = args[1]
    else:
        nlevs = 1

    if len(args) > 2:
        if isinstance(args[2], str):
            filt = namedFilter(args[2])
        else:
            filt = numpy.array(args[2])
    else:
        filt = namedFilter('binom5')

    #--------------------------------------------------------------------
    
    if len(filt.shape) == 1 or filt.shape[0] == 1 or filt.shape[1] == 1:
        filt = filt / sum(filt)
    else:
        filt = filt / sum(sum(filt))

    if nlevs > 0:
        if len(im.shape) == 1 or im.shape[0] == 1 or im.shape[1] == 1: 
            # 1D image
            if len(filt) == 2 and (numpy.asarray(filt.shape) != 1).any():
                print('Error: can not apply 2D filter to 1D signal')
                return
            
            imIn = corrDn(im, filt, 'reflect1', len(im))
            out = blur(imIn, nlevs-1, filt)
            res = upconv(out, filt, 'reflect1', len(im), [0,0],
                         len(im))
            return res
        elif len(filt.shape) == 1 or filt.shape[0] == 1 or filt.shape[1] == 1:
            # 2D image 1D filter
            imIn = corrDn(im, filt, 'reflect1', [2,1])
            imIn = corrDn(imIn, filt.T, 'reflect1', [1,2])
            out = blur(imIn, nlevs-1, filt)
            res = upConv(out, filt.T, 'reflect1', [1,2], [0,0],
                         [out.shape[0], im.shape[1]])
            res = upConv(res, filt, 'reflect1', [2,1], [0,0],
                         im.shape)
            return res
        else:
            # 2D image 2D filter
            imIn = corrDn(im, filt, 'reflect1', [2,2])
            out = blur(imIn, nlevs-1, filt)
            res = upConv(out, filt, 'reflect1', [2,2], [0,0],
                         im.shape)
            return res
    else:
        return im
