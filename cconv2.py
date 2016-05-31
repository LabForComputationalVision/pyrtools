import numpy
import scipy.signal

def cconv2(*args):
    ''' RES = CCONV2(MTX1, MTX2, CTR)
    
        Circular convolution of two matrices.  Result will be of size of
        LARGER vector.
     
        The origin of the smaller matrix is assumed to be its center.
        For even dimensions, the origin is determined by the CTR (optional) 
        argument:
             CTR   origin
              0     DIM/2      (default)
              1     (DIM/2)+1  
    
        Eero Simoncelli, 6/96.  Modified 2/97.  
        Python port by Rob Young, 8/15  '''
    
    if len(args) < 2:
        print 'Error: cconv2 requires two input matrices!'
        print 'Usage: cconv2(matrix1, matrix2, center)'
        print 'where center parameter is optional'
        return
    else:
        a = numpy.array(args[0])
        b = numpy.array(args[1])

    if len(args) == 3:
        ctr = args[2]
    else:
        ctr = 0

    if a.shape[0] >= b.shape[0] and a.shape[1] >= b.shape[1]:
        large = a
        small = b
    elif a.shape[0] <= b.shape[0] and a.shape[1] <= b.shape[1]:
        large = b
        small = a
    else:
        print 'Error: one matrix must be larger than the other in both dimensions!'
        return
    
    ly = large.shape[0]
    lx = large.shape[1]
    sy = small.shape[0]
    sx = small.shape[1]

    ## These values are the index of the small mtx that falls on the
    ## border pixel of the large matrix when computing the first
    ## convolution response sample:
    sy2 = numpy.floor((sy+ctr+1)/2.0)
    sx2 = numpy.floor((sx+ctr+1)/2.0)

    # pad
    nw = large[ly-sy+sy2:ly, lx-sx+sx2:lx]
    n = large[ly-sy+sy2:ly, :]
    ne = large[ly-sy+sy2:ly, :sx2-1]
    w = large[:, lx-sx+sx2:lx]
    c = large
    e = large[:, :sx2-1]
    sw = large[:sy2-1, lx-sx+sx2:lx]
    s = large[:sy2-1, :]
    se = large[:sy2-1, :sx2-1]

    n = numpy.column_stack((nw, n, ne))
    c = numpy.column_stack((w,large,e))
    s = numpy.column_stack((sw, s, se))

    clarge = numpy.concatenate((n, c), axis=0)
    clarge = numpy.concatenate((clarge, s), axis=0)

    c = scipy.signal.convolve(clarge, small, 'valid')

    return c
