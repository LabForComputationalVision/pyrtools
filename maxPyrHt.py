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
        #if imsz[0] < filtsz[0] or imsz[1] < filtsz[1]:
        #    print 'flag 2'
        #    return 0

    if not isinstance(imsz, tuple) and not isinstance(filtsz, tuple):
        imsz = imsz
        filtsz = filtsz
        if imsz < filtsz:
            return 0
    elif 1 in imsz:         # 1D image
        imsz = imsz[0] * imsz[1]
        filtsz = filtsz[0] * filtsz[1]
        if imsz < filtsz:
            return 0
    #elif 1 in filtsz:   # 2D image, 1D filter
    else:   # 2D image
        #filtsz = (filtsz[0], filtsz[0])
        #print filtsz
        if ( imsz[0] < filtsz[0] or imsz[0] < filtsz[1] or
             imsz[1] < filtsz[0] or imsz[1] < filtsz[1] ):
            return 0

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



''' new - probably not needed    
    # if tuple and 1D, make column vector
    if isinstance(imsz, tuple):
        if 1 in imsz:
            imsz = (max(imsz), 1)
    if isinstance(filtsz, tuple):
        if 1 in filtsz:
            filtsz = (max(filtsz), 1)


    image1D = False
    if isinstance(imsz, tuple) and 1 in imsz:  #1D image
        image1D = True
        imsz = max(imsz)
    elif isinstance(imsz, tuple) and len(imsz) == 1: # 1D image
        image1D = True
        imsz = imsz[0]
    elif isinstance(imsz, int):  # 1D image
        image1D = True
        imsz = imsz
    
    if image1D:
        if isinstance(filtsz, tuple):
            prod = 1
            for t in filtsz:
                prod *= t
            filtsz = prod
    elif isinstance(filtsz, tuple) and 1 in filtsz:   # 2D image, 1D filter
        filtsz = (filtsz[1], filtsz[1])
    elif isinstance(filtsz, tuple) and len(filtsz) == 1:   # 2D image, 1D filter
        filtsz = (filtsz[1], filtsz[1])
    elif isinstance(filtsz, int):   # 2D image, 1D filter
        filtsz = (filtsz, filtsz)


    print imsz
    print filtsz
    
    if isinstance(imsz, tuple) and isinstance(filtsz, tuple):
        if min(imsz) < max(filtsz):
            height = 0
        else:
            height = 1 + maxPyrHt( (numpy.floor(imsz[0]/2.0), 
                                    numpy.floor(imsz[1]/2.0)), 
                                   filtsz )
    elif isinstance(imsz, tuple) and isinstance(filtsz, int):
        if min(imsz) < filtsz:
            height = 0
        else:
            height = 1 + maxPyrHt( (numpy.floor(imsz[0]/2.0), 
                                    numpy.floor(imsz[1]/2.0)), 
                                   filtsz )
    elif isinstance(imsz, int) and isinstance(filtsz, tuple):
        if imsz < max(filtsz):
            height = 0
        else:
            height = 1 + maxPyrHt( (numpy.floor(imsz[0]/2.0), 
                                    numpy.floor(imsz[1]/2.0)), 
                                   filtsz ) 
    elif isinstance(imsz, int) and isinstance(filtsz, int):
        if imsz < filtsz:
            height = 0
        else:
            height = 1 + maxPyrHt( (numpy.floor(imsz[0]/2.0), 
                                    numpy.floor(imsz[1]/2.0)), 
                                   filtsz ) 
        
    return height
'''


