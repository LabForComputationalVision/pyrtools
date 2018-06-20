import numpy
from .range2 import range2
from .round import round

def histo(*args):
    ''' [N,X] = histo(MTX, nbinsOrBinsize, binCenter);
    
        Compute a histogram of (all) elements of MTX.  N contains the histogram
        counts, X is a vector containg the centers of the histogram bins.
    
        nbinsOrBinsize (optional, default = 101) specifies either
        the number of histogram bins, or the negative of the binsize.
    
        binCenter (optional, default = mean2(MTX)) specifies a center position
        for (any one of) the histogram bins.
    
        How does this differ from MatLab's HIST function?  This function:
          - allows uniformly spaced bins only.
          +/- operates on all elements of MTX, instead of columnwise.
          + is much faster (approximately a factor of 80 on my machine).
          + allows specification of number of bins OR binsize.  
            Default=101 bins.
          + allows (optional) specification of binCenter.
    
        Eero Simoncelli, 3/97.  ported to Python by Rob Young, 8/15.  '''

    if len(args) == 0 or len(args) > 3:
        print('Usage: histo(mtx, nbins, binCtr)')
        print('first argument is required')
        return
    else:
        mtx = args[0]
    mtx = numpy.array(mtx)

    (mn, mx) = range2(mtx)

    if len(args) > 2:
        binCtr = args[2]
    else:
        binCtr = mtx.mean()

    if len(args) > 1:
        if args[1] < 0:
            binSize = -args[1]
        else:
            binSize = ( float(mx-mn) / float(args[1]) )
            tmpNbins = ( round(float(mx-binCtr) / float(binSize)) - 
                         round(float(mn-binCtr) / float(binSize)) )
            if tmpNbins != args[1]:
                print('Warning: Using %d bins instead of requested number (%d)' % (tmpNbins, args[1]))
    else:
        binSize = float(mx-mn) / 101.0

    firstBin = binCtr + binSize * round( (mn-binCtr)/float(binSize) )
    firstEdge = firstBin - (binSize / 2.0) + (binSize * 0.01)

    tmpNbins = int( round( (mx-binCtr) / binSize ) -
                    round( (mn-binCtr) / binSize ) )
    
    # numpy.histogram uses bin edges, not centers like Matlab's hist
    #bins = firstBin + binSize * numpy.array(range(tmpNbins+1))
    # compute bin edges
    binsE = firstEdge + binSize * numpy.array(list(range(tmpNbins+1)))
    
    [N, X] = numpy.histogram(mtx, binsE)

    # matlab version returns column vectors, so we will too.
    N = N.reshape(1, N.shape[0])
    X = X.reshape(1, X.shape[0])

    return (N, X)
