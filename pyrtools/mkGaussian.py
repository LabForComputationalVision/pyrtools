import numpy

def mkGaussian(*args):
    ''' IM = mkGaussian(SIZE, COVARIANCE, MEAN, AMPLITUDE)
 
        Compute a matrix with dimensions SIZE (a [Y X] 2-vector, or a
        scalar) containing a Gaussian function, centered at pixel position
        specified by MEAN (default = (size+1)/2), with given COVARIANCE (can
        be a scalar, 2-vector, or 2x2 matrix.  Default = (min(size)/6)^2),
        and AMPLITUDE.  AMPLITUDE='norm' (default) will produce a
        probability-normalized function.  All but the first argument are
        optional.

        Eero Simoncelli, 6/96. Python port by Rob Young, 7/15.  '''

    if len(args) == 0:
        print("mkRamp(SIZE, COVARIANCE, MEAN, AMPLITUDE)")
        print("first argument is required")
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print("first argument must be a two element tuple or an integer")
            exit(1)

    # OPTIONAL args:

    if len(args) > 1:
        cov = args[1]
    else:
        cov = (min([sz[0], sz[1]]) / 6.0) ** 2

    if len(args) > 2:
        mn = args[2]
        if isinstance(mn, int):
            mn = [mn, mn]
    else:
        mn = ( (sz[0]+1.0)/2.0, (sz[1]+1.0)/2.0 )

    if len(args) > 3:
        ampl = args[3]
    else:
        ampl = 'norm'

    #---------------------------------------------------------------
        
    (xramp, yramp) = numpy.meshgrid(numpy.array(list(range(1,sz[1]+1)))-mn[1], 
                                    numpy.array(list(range(1,sz[0]+1)))-mn[0])

    if isinstance(cov, (int, float)):
        if 'norm' == ampl:
            ampl = 1.0 / (2.0 * numpy.pi * cov)
        e = ( (xramp**2) + (yramp**2) ) / ( -2.0 * cov )
    elif len(cov) == 2 and isinstance(cov[0], (int, float)):
        if 'norm' == ampl:
            if cov[0]*cov[1] < 0:
                ampl = 1.0 / (2.0 * numpy.pi * 
                              numpy.sqrt(complex(cov[0] * cov[1])))
            else:
                ampl = 1.0 / (2.0 * numpy.pi * numpy.sqrt(cov[0] * cov[1]))
        e = ( (xramp**2) / (-2 * cov[1]) ) + ( (yramp**2) / (-2 * cov[0]) )
    else:
        if 'norm' == ampl:
            detCov = numpy.linalg.det(cov)
            if (detCov < 0).any():
                detCovComplex = numpy.empty(detCov.shape, dtype=complex)
                detCovComplex.real = detCov
                detCovComplex.imag = numpy.zeros(detCov.shape)
                ampl = 1.0 / ( 2.0 * numpy.pi * numpy.sqrt( detCovComplex ) )
            else:
                ampl = 1.0 / (2.0 * numpy.pi * numpy.sqrt( numpy.linalg.det(cov) ) )
        cov = - numpy.linalg.inv(cov) / 2.0
        e = (cov[1,1] * xramp**2) + ( 
            (cov[0,1]+cov[1,0])*(xramp*yramp) ) + ( cov[0,0] * yramp**2)
        
    res = ampl * numpy.exp(e)
    
    return res
