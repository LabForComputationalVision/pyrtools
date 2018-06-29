import numpy
from .mkR import mkR

def mkZonePlate(*args):
    ''' IM = mkZonePlate(SIZE, AMPL, PHASE)
    
        Make a "zone plate" image:
            AMPL * cos( r^2 + PHASE)
            SIZE specifies the matrix size, as for zeros().  
            AMPL (default = 1) and PHASE (default = 0) are optional.

        Eero Simoncelli, 6/96.  Python port by Rob Young, 7/15.  '''

    # REQUIRED ARGS:

    if len(args) == 0:
        print("mkZonePlate(SIZE, AMPL, PHASE)")
        print("first argument is required")
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print("first argument must be a two element tuple or an integer")
            exit(1)
    
    #---------------------------------------------------------------------
    # OPTIONAL ARGS
    if len(args) > 1:
        ampl = args[1]
    else:
        ampl = 1
    if len(args) > 2:
        ph = args[2]
    else:
        ph = 0

    #---------------------------------------------------------------------

    res = ampl * numpy.cos( (numpy.pi / max(sz)) * mkR(sz, 2) + ph )

    return res
