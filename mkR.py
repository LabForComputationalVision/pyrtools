import numpy

def mkR(*args):
    ''' Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
        containing samples of a radial ramp function, raised to power EXPT
        (default = 1), with given ORIGIN (default = (size+1)/2, [1 1] =
        upper left).  All but the first argument are optional.
        Eero Simoncelli, 6/96.  Ported to Python by Rob Young, 5/14.  '''
    
    if len(args) == 0:
        print 'Error: first input parameter is required!'
        return
    else:
        sz = args[0]

    if isinstance(sz, (int, long)) or len(sz) == 1:
        sz = (sz, sz)

    # -----------------------------------------------------------------
    # OPTIONAL args:

    if len(args) < 2:
        expt = 1;
    else:
        expt = args[1]

    if len(args) < 3:
        origin = ((sz[0]+1)/2.0, (sz[1]+1)/2.0)
    else:
        origin = args[2]

    # -----------------------------------------------------------------

    (xramp2, yramp2) = numpy.meshgrid(numpy.array(range(1,sz[1]+1))-origin[1], 
                                   numpy.array(range(1,sz[0]+1))-origin[0])

    res = (xramp2**2 + yramp2**2)**(expt/2.0)
    
    return res
