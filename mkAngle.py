import numpy

def mkAngle(*args):
    ''' Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
        containing samples of the polar angle (in radians, CW from the
        X-axis, ranging from -pi to pi), relative to angle PHASE (default =
        0), about ORIGIN pixel (default = (size+1)/2). '''

    if len(args) > 0:
        sz = args[0]
        if not isinstance(sz, tuple):
            sz = (sz, sz)
    else:
        print "Error: first input parameter 'size' is required!"
        print "makeAngle(size, phase, origin)"
        return

    # ------------------------------------------------------------
    # Optional args:

    if len(args) > 1:
        phase = args[1]
    else:
        phase = 'not set'

    if len(args) > 2:
        origin = args[2]
    else:
        origin = (sz[0]+1/2, sz[1]+1/2)

    #------------------------------------------------------------------

    (xramp, yramp) = numpy.meshgrid(numpy.array(range(1,sz[1]+1))-origin[1], 
                                 (numpy.array(range(1,sz[0]+1)))-origin[0])
    xramp = numpy.array(xramp)
    yramp = numpy.array(yramp)

    res = numpy.arctan2(yramp, xramp)
    
    if phase != 'not set':
        res = ((res+(numpy.pi-phase)) % (2*numpy.pi)) - numpy.pi

    return res
