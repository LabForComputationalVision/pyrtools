import numpy
import math
from .mkRamp import mkRamp

def mkSine(*args):
    ''' IM = mkSine(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN)
                           or
        IM = mkSine(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN)
 
        Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
        containing samples of a 2D sinusoid, with given PERIOD (in pixels),
        DIRECTION (radians, CW from X-axis, default = 0), AMPLITUDE (default
        = 1), and PHASE (radians, relative to ORIGIN, default = 0).  ORIGIN
        defaults to the center of the image.
 
        In the second form, FREQ is a 2-vector of frequencies (radians/pixel).

        Eero Simoncelli, 6/96. Python version by Rob Young, 7/15.  '''

    # REQUIRED args:

    if len(args) < 2:
        print("mkSine(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN)")
        print("       or")
        print("mkSine(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN)")
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

    #----------------------------------------------------------------

    if origin == 'not set':
        res = amplitude * numpy.sin(mkRamp(sz, direction, frequency, phase))
    else:
        res = amplitude * numpy.sin(mkRamp(sz, direction, frequency, phase, 
                                           [origin[0]-1, origin[1]-1]))

    return res
