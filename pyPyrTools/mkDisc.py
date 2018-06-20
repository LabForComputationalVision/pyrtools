import numpy
import sys
from .mkR import mkR
from .rcosFn import rcosFn
from .pointOp import pointOp

def mkDisc(*args):
    ''' IM = mkDisc(SIZE, RADIUS, ORIGIN, TWIDTH, VALS)

        Make a "disk" image.  SIZE specifies the matrix size, as for
        zeros().  RADIUS (default = min(size)/4) specifies the radius of 
        the disk.  ORIGIN (default = (size+1)/2) specifies the 
        location of the disk center.  TWIDTH (in pixels, default = 2) 
        specifies the width over which a soft threshold transition is made.
        VALS (default = [0,1]) should be a 2-vector containing the
        intensity value inside and outside the disk.  

        Eero Simoncelli, 6/96. Python port by Rob Young, 7/15.  '''

    if len(args) == 0:
        print("mkDisc(SIZE, RADIUS, ORIGIN, TWIDTH, VALS)")
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
        rad = args[1]
    else:
        rad = min(sz) / 4.0

    if len(args) > 2:
        origin = args[2]
    else:
        origin = ( (sz[0]+1.0)/2.0, (sz[1]+1.0)/2.0 )

    if len(args) > 3:
        twidth = args[3]
    else:
        twidth = twidth = 2
        
    if len(args) > 4:
        vals = args[4]
    else:
        vals = (1,0)

    #--------------------------------------------------------------

    res = mkR(sz, 1, origin)

    if abs(twidth) < sys.float_info.min:
        res = vals[1] + (vals[0] - vals[1]) * (res <= rad);
    else:
        [Xtbl, Ytbl] = rcosFn(twidth, rad, [vals[0], vals[1]]);
        res = pointOp(res, Ytbl, Xtbl[0], Xtbl[1]-Xtbl[0], 0);

    return numpy.array(res)
