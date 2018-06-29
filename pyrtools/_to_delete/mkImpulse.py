import numpy

def mkImpulse(*args):
    ''' create an image that is all zeros except for an impulse '''

    if(len(args) == 0):
        print("mkImpulse(size, origin, amplitude)")
        print("first input parameter is required")
        return
    
    if(isinstance(args[0], int)):
        sz = (args[0], args[0])
    elif(isinstance(args[0], tuple)):
        sz = args[0]
    else:
        print("size parameter must be either an integer or a tuple")
        return

    if(len(args) > 1):
        origin = args[1]
    else:
        origin = ( numpy.ceil(sz[0]/2.0), numpy.ceil(sz[1]/2.0) )

    if(len(args) > 2):
        amplitude = args[2]
    else:
        amplitude = 1

    res = numpy.zeros(sz);
    res[origin[0], origin[1]] = amplitude

    return res
