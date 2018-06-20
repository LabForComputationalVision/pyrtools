import numpy
import math
from .mkRamp import mkRamp
from .rcosFn import rcosFn
from .pointOp import pointOp

def mkSquare(*args):
    ''' IM = mkSquare(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN, TWIDTH)
                    or
        IM = mkSquare(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN, TWIDTH)
     
        Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
        containing samples of a 2D square wave, with given PERIOD (in
        pixels), DIRECTION (radians, CW from X-axis, default = 0), AMPLITUDE
        (default = 1), and PHASE (radians, relative to ORIGIN, default = 0).
        ORIGIN defaults to the center of the image.  TWIDTH specifies width
        of raised-cosine edges on the bars of the grating (default =
        min(2,period/3)).
     
        In the second form, FREQ is a 2-vector of frequencies (radians/pixel).
    
        Eero Simoncelli, 6/96. Python port by Rob Young, 7/15.
    
        TODO: Add duty cycle.  '''

    #REQUIRED ARGS:

    if len(args) < 2:
        print("mkSquare(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN, TWIDTH)")
        print("       or")
        print("mkSquare(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN, TWIDTH)")
        print("first two arguments are required")
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print("first argument must be a two element tuple or an integer")
            exit(1)
    
    if isinstance(args[1], (int, float)):
        frequency = (2.0 * numpy.pi) / args[1]
        # OPTIONAL args:
        if len(args) > 2:
            direction = args[2]
        else:
            direction = 0
        if len(args) > 3:
            amplitude = args[3]
        else:
            amplitude = 1
        if len(args) > 4:
            phase = args[4]
        else:
            phase = 0
        if len(args) > 5:
            origin = args[5]
        else:
            origin = 'not set'
        if len(args) > 6:
            transition = args[6]
        else:
            transition = min(2, 2.0 * numpy.pi / (3.0*frequency))
    else:
        frequency = numpy.linalg.norm(args[1])
        direction = math.atan2(args[1][0], args[1][1])
        # OPTIONAL args:
        if len(args) > 2:
            amplitude = args[2]
        else:
            amplitude = 1
        if len(args) > 3:
            phase = args[3]
        else:
            phase = 0
        if len(args) > 4:
            origin = args[4]
        else:
            origin = 'not set'
        if len(args) > 5:
            transition = args[5]
        else:
            transition = min(2, 2.0 * numpy.pi / (3.0*frequency))

    #------------------------------------------------------------

    if origin != 'not set':
        res = mkRamp(sz, direction, frequency, phase, 
                     (origin[0]-1, origin[1]-1)) - numpy.pi/2.0
    else:
        res = mkRamp(sz, direction, frequency, phase) - numpy.pi/2.0

    [Xtbl, Ytbl] = rcosFn(transition * frequency, numpy.pi/2.0, 
                          [-amplitude, amplitude])

    res = pointOp(abs(((res+numpy.pi) % (2.0*numpy.pi))-numpy.pi), Ytbl, 
                  Xtbl[0], Xtbl[1]-Xtbl[0], 0)

    return res
