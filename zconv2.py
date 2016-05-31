import numpy
import scipy.signal

def zconv2(*args):
    ''' RES = ZCONV2(MTX1, MTX2, CTR)
    
        Convolution of two matrices, with boundaries handled as if the larger 
        mtx lies in a sea of zeros. Result will be of size of LARGER vector.
     
        The origin of the smaller matrix is assumed to be its center.
        For even dimensions, the origin is determined by the CTR (optional) 
        argument:
             CTR   origin
              0     DIM/2      (default)
              1     (DIM/2)+1  (behaves like conv2(mtx1,mtx2,'same'))
    
        Eero Simoncelli, 2/97.  Python port by Rob Young, 10/15.  '''

    # REQUIRED ARGUMENTS
    #----------------------------------------------------------------
    
    if len(args) < 2 or len(args) > 3:
        print 'Usage: zconv2(matrix1, matrix2, center)'
        print 'first two input parameters are required'
        return
    else:
        a = numpy.array(args[0])
        b = numpy.array(args[1])

    # OPTIONAL ARGUMENT
    #----------------------------------------------------------------

    if len(args) == 3:
        ctr = args[2]
    else:
        ctr = 0

    #----------------------------------------------------------------

    if (a.shape[0] >= b.shape[0]) and (a.shape[1] >= b.shape[1]):
        large = a
        small = b
    elif (a.shape[0] <= b.shape[0]) and (a.shape[1] <= b.shape[1]):
        large = b
        small = a
    else:
        print 'Error: one arg must be larger than the other in both dimensions!'
        return
        
    ly = large.shape[0]
    lx = large.shape[1]
    sy = small.shape[0]
    sx = small.shape[1]

    ## These values are the index of the small matrix that falls on the 
    ## border pixel of the large matrix when computing the first
    ## convolution response sample:
    sy2 = numpy.floor((sy+ctr+1)/2.0)-1
    sx2 = numpy.floor((sx+ctr+1)/2.0)-1

    clarge = scipy.signal.convolve(large, small, 'full')
    
    c = clarge[sy2:ly+sy2, sx2:lx+sx2]

    return c
