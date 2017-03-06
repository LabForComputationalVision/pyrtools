import numpy
from namedFilter import namedFilter
from corrDn import corrDn

def blurDn(*args):
    ''' RES = blurDn(IM, LEVELS, FILT)
        Blur and downsample an image.  The blurring is done with filter
        kernel specified by FILT (default = 'binom5'), which can be a string
        (to be passed to namedFilter), a vector (applied separably as a 1D
        convolution kernel in X and Y), or a matrix (applied as a 2D
        convolution kernel).  The downsampling is always by 2 in each
        direction.
        The procedure is applied recursively LEVELS times (default=1).
        Eero Simoncelli, 3/97.  Ported to python by Rob Young 4/14
        function res = blurDn(im, nlevs, filt)  '''

    if len(args) == 0:
        print "Error: image input parameter required."
        return

    im = numpy.array(args[0])
    
    # optional args
    if len(args) > 1:
        nlevs = args[1]
    else:
        nlevs = 1

    if len(args) > 2:
        filt = args[2]
        if isinstance(filt, basestring):
            filt = namedFilter(filt)
    else:
        filt = namedFilter('binom5')

    if filt.shape[0] == 1 or filt.shape[1] == 1:
        filt = [x/sum(filt) for x in filt]
    else:
        filt = [x/sum(sum(filt)) for x in filt]

    filt = numpy.array(filt)
    
    if nlevs > 1:
        im = blurDn(im, nlevs-1, filt)

    if nlevs >= 1:
        if len(im.shape) == 1 or im.shape[0] == 1 or im.shape[1] == 1:
            # 1D image
            if len(filt.shape) > 1 and (filt.shape[1]!=1 and filt.shape[2]!=1):
                # >1D filter
                print 'Error: Cannot apply 2D filter to 1D signal'
                return
            # orient filter and image correctly
            if im.shape[0] == 1:
                if len(filt.shape) == 1 or filt.shape[1] == 1:
                    filt = filt.T
            else:
                if filt.shape[0] == 1:
                    filt = filt.T
                
            res = corrDn(image = im, filt = filt, step = (2, 2))
            if len(im.shape) == 1 or im.shape[1] == 1:
                res = numpy.reshape(res, (numpy.ceil(im.shape[0]/2.0), 1))
            else:
                res = numpy.reshape(res, (1, numpy.ceil(im.shape[1]/2.0)))
        elif len(filt.shape) == 1 or filt.shape[0] == 1 or filt.shape[1] == 1:
            # 2D image and 1D filter
            res = corrDn(image = im, filt = filt.T, step = (2, 1))
            res = corrDn(image = res, filt = filt, step = (1, 2))

        else:  # 2D image and 2D filter
            res = corrDn(image = im, filt = filt, step = (2,2))
    else:
        res = im
            
    return res
