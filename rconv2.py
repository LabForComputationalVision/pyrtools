import numpy
import scipy.signal

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
    sy2 = numpy.floor((sy+ctr-1)/2)
    sx2 = numpy.floor((sx+ctr-1)/2)

    # pad with reflected copies
    nw = large[sy-sy2-1:0:-1, sx-sx2-1:0:-1]
    n = large[sy-sy2-1:0:-1, :]
    ne = large[sy-sy2-1:0:-1, lx-2:lx-sx2-2:-1]
    w = large[:, sx-sx2-1:0:-1]
    e = large[:, lx-2:lx-sx2-2:-1]
    sw = large[ly-2:ly-sy2-2:-1, sx-sx2-1:0:-1]
    s = large[ly-2:ly-sy2-2:-1, :]
    se = large[ly-2:ly-sy2-2:-1, lx-2:lx-sx2-2:-1]

    n = numpy.column_stack((nw, n, ne))
    c = numpy.column_stack((w,large,e))
    s = numpy.column_stack((sw, s, se))

    clarge = numpy.concatenate((n, c), axis=0)
    clarge = numpy.concatenate((clarge, s), axis=0)
    
    return scipy.signal.convolve(clarge, small, 'valid')
