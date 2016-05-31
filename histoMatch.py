import numpy
import scipy

def histoMatch(*args):
    ''' RES = histoMatch(MTX, N, X, mode)
    
        Modify elements of MTX so that normalized histogram matches that
        specified by vectors X and N, where N contains the histogram counts
        and X the histogram bin positions (see histo).
    
        new input parameter 'mode' can be either 'centers' or 'edges' that tells
        the function if the input X values are bin centers or edges.
    
        Eero Simoncelli, 7/96. Ported to Python by Rob Young, 10/15.  '''
    
    mode = str(args[3])
    mtx = numpy.array(args[0])
    N = numpy.array(args[1])
    X = numpy.array(args[2])
    if mode == 'edges':         # convert to centers
        correction = (X[0][1] - X[0][0]) / 2.0
        X = (X[0][:-1] + correction).reshape(1, X.shape[1]-1)
        
    [oN, oX] = histo(mtx.flatten(), X.flatten().shape[0])
    if mode == 'edges':        # convert to centers
        correction = (oX[0][1] - oX[0][0]) / 2.0
        oX = (oX[0][:-1] + correction).reshape(1, oX.shape[1]-1)

    # remember: histo returns a column vector, so the indexing is thus
    oStep = oX[0][1] - oX[0][0]
    oC = numpy.concatenate((numpy.array([0]), 
                            numpy.array(numpy.cumsum(oN) / 
                                        float(sum(sum(oN))))))
    oX = numpy.concatenate((numpy.array([oX[0][0]-oStep/2.0]), 
                            numpy.array(oX[0]+oStep/2.0)))
    
    N = N.flatten()
    X = X.flatten()
    N = N + N.mean() / 1e8  # HACK: no empty bins ensures nC strictly monotonic
    
    nStep = X[1] - X[0]
    nC = numpy.concatenate((numpy.array([0]), 
                            numpy.array(numpy.cumsum(N) / sum(N))))
    nX = numpy.concatenate((numpy.array([X[0] - nStep / 2.0]),
                            numpy.array(X+nStep / 2.0)))
    
    # unlike in matlab, interp1d returns a function
    func = scipy.interpolate.interp1d(nC, nX, 'linear')
    nnX = func(oC)

    return pointOp(mtx, nnX, oX[0], oStep, 0)
