import numpy
from .mkAngle import mkAngle

def mkAngularSine(*args):
    ''' IM = mkAngularSine(SIZE, HARMONIC, AMPL, PHASE, ORIGIN)

        Make an angular sinusoidal image:
        AMPL * sin( HARMONIC*theta + PHASE),
        where theta is the angle about the origin.
        SIZE specifies the matrix size, as for zeros().  
        AMPL (default = 1) and PHASE (default = 0) are optional.
    
        Eero Simoncelli, 2/97.  Python port by Rob Young, 7/15.  '''

    if len(args) == 0:
        print("mkAngularSine(SIZE, HARMONIC, AMPL, PHASE, ORIGIN)")
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
        harmonic = args[1]
    else:
        harmonic = 1

    if len(args) > 2:
        ampl = args[2]
    else:
        ampl = 1

    if len(args) > 3:
        ph = args[3]
    else:
        ph = 0

    if len(args) > 4:
        origin = args[4]
    else:
        origin = ( (sz[0]+1.0)/2.0, (sz[1]+1.0)/2.0 )
        
    res = ampl * numpy.sin( harmonic * mkAngle(sz, ph, origin) + ph )

    return res
