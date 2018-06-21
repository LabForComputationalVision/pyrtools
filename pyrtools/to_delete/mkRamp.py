import numpy
import math

def mkRamp(*args):
    ''' mkRamp(SIZE, DIRECTION, SLOPE, INTERCEPT, ORIGIN)
        Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
        containing samples of a ramp function, with given gradient DIRECTION
        (radians, CW from X-axis, default = 0), SLOPE (per pixel, default = 
        1), and a value of INTERCEPT (default = 0) at the ORIGIN (default =
        (size+1)/2, [1 1] = upper left). All but the first argument are
        optional '''
    
    if len(args) == 0:
        print("mkRamp(SIZE, DIRECTION, SLOPE, INTERCEPT, ORIGIN)")
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
        direction = args[1]
    else:
        direction = 0

    if len(args) > 2:
        slope = args[2]
    else:
        slope = 1

    if len(args) > 3:
        intercept = args[3]
    else:
        intercept = 0

    if len(args) > 4:
        origin = args[4]
    else:
        origin = ( float(sz[0]-1)/2.0, float(sz[1]-1)/2.0 )

    #--------------------------

    xinc = slope * math.cos(direction)
    yinc = slope * math.sin(direction)

    [xramp, yramp] = numpy.meshgrid( xinc * (numpy.array(list(range(sz[1])))-origin[1]),
                                  yinc * (numpy.array(list(range(sz[0])))-origin[0]) )

    res = intercept + xramp + yramp

    return res
