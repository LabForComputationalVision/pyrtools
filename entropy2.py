import numpy

def entropy2(*args):
    ''' E = ENTROPY2(MTX,BINSIZE) 
     
        Compute the first-order sample entropy of MTX.  Samples of VEC are
        first discretized.  Optional argument BINSIZE controls the
        discretization, and defaults to 256/(max(VEC)-min(VEC)).
    
        NOTE: This is a heavily  biased estimate of entropy when you
        don't have much data.
    
        Eero Simoncelli, 6/96. Ported to Python by Rob Young, 10/15.  '''
    
    vec = numpy.array(args[0])
    # if 2D flatten to a vector
    if len(vec.shape) != 1 and (vec.shape[0] != 1 or vec.shape[1] != 1):
        vec = vec.flatten()

    (mn, mx) = range2(vec)

    if len(args) > 1:
        binsize = args[1]
        # FIX: why is this max in the Matlab code; it's just a float?
        # we insure that vec isn't 2D above, so this shouldn't be needed
        #nbins = max( float(mx-mn)/float(binsize) )
        nbins = float(mx-mn) / float(binsize)
    else:
        nbins = 256

    [bincount, bins] = histo(vec, nbins)

    ## Collect non-zero bins:
    H = bincount[ numpy.where(bincount > 0) ]
    H = H / float(sum(H))

    return -sum(H * numpy.log2(H))
