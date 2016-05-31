import numpy

def maxPyrHt(imsz, filtsz):
    ''' Compute maximum pyramid height for given image and filter sizes.
        Specifically: the number of corrDn operations that can be sequentially
        performed when subsampling by a factor of 2. '''

    if not isinstance(imsz, tuple) or not isinstance(filtsz, tuple):
        if imsz < filtsz:
            return 0
    else:
        if len(imsz) == 1:
            imsz = (imsz[0], 1)
        if len(filtsz) == 1:
            filtsz = (filtsz[0], 1)
        #if filtsz[1] == 1:  # new
        #    filtsz = (filtsz[1], filtsz[0])
        if imsz[0] < filtsz[0] or imsz[1] < filtsz[1]:
            return 0

    if not isinstance(imsz, tuple) and not isinstance(filtsz, tuple):
        imsz = imsz
        filtsz = filtsz
    elif imsz[0] == 1 or imsz[1] == 1:         # 1D image
        imsz = imsz[0] * imsz[1]
        filtsz = filtsz[0] * filtsz[1]
    elif filtsz[0] == 1 or filtsz[1] == 1:   # 2D image, 1D filter
        filtsz = (filtsz[0], filtsz[0])

    if ( not isinstance(imsz, tuple) and not isinstance(filtsz, tuple) and
         imsz < filtsz ) :
        height = 0
    elif not isinstance(imsz, tuple) and not isinstance(filtsz, tuple):
        height = 1 + maxPyrHt( numpy.floor(imsz/2.0), filtsz )
    else:
        height = 1 + maxPyrHt( (numpy.floor(imsz[0]/2.0), 
                                numpy.floor(imsz[1]/2.0)), 
                               filtsz )

    return height
