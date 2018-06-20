import numpy
from .namedFilter import namedFilter
from .upConv import upConv

def upBlur(*args):
    ''' RES = upBlur(IM, LEVELS, FILT)
    
        Upsample and blur an image.  The blurring is done with filter
        kernel specified by FILT (default = 'binom5'), which can be a string
        (to be passed to namedFilter), a vector (applied separably as a 1D
        convolution kernel in X and Y), or a matrix (applied as a 2D
        convolution kernel).  The downsampling is always by 2 in each
        direction.
    
        The procedure is applied recursively LEVELS times (default=1).
    
        Eero Simoncelli, 4/97. Python port by Rob Young, 10/15.   '''
    
    #---------------------------------------------------------------
    # REQUIRED ARGS
    
    if len(args) == 0:
        print('Usage: upBlur(image, levels, filter)')
        print('first argument is required')
    else:
        im = numpy.array(args[0])

    #---------------------------------------------------------------
    # OPTIONAL ARGS
    
    if len(args) > 1:
        nlevs = args[1]
    else:
        nlevs = 1

    if len(args) > 2:
        filt = args[2]
    else:
        filt = 'binom5'

    #------------------------------------------------------------------

    if isinstance(filt, str):
        filt = namedFilter(filt)

    if nlevs > 1:
        im = upBlur(im, nlevs-1, filt)

    if nlevs >= 1:
        if im.shape[0] == 1 or im.shape[1] == 1:
            if im.shape[0] == 1:
                filt = filt.reshape(filt.shape[1], filt.shape[0])
                start = (1,2)
            else:
                start = (2,1)
            res = upConv(im, filt, 'reflect1', start)
        elif filt.shape[0] == 1 or filt.shape[1] == 1:
            if filt.shape[0] == 1:
                filt = filt.reshape(filt.shape[1], 1)
            res = upConv(im, filt, 'reflect1', [2,1])
            res = upConv(res, filt.T, 'reflect1', [1,2])
        else:
            res = upConv(im, filt, 'reflect1', [2,2])
    else:
        res = im

    return res
