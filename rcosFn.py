import numpy

def rcosFn(*args):
    ''' [X, Y] = rcosFn(WIDTH, POSITION, VALUES)

        Return a lookup table (suitable for use by INTERP1) 
        containing a "raised cosine" soft threshold function:
 
        Y =  VALUES(1) + (VALUES(2)-VALUES(1)) *
             cos^2( PI/2 * (X - POSITION + WIDTH)/WIDTH )

        WIDTH is the width of the region over which the transition occurs
        (default = 1). POSITION is the location of the center of the
        threshold (default = 0).  VALUES (default = [0,1]) specifies the
        values to the left and right of the transition.  '''
    
    if len(args) > 0:
        width = args[0]
    else:
        width = 1

    if len(args) > 1:
        position = args[1]
    else:
        position = 0

    if len(args) > 2:
        values = args[2]
    else:
        values = (0,1)

    #---------------------------------------------

    sz = 256   # arbitrary!

    X = numpy.pi * numpy.array(range(-sz-1,2)) / (2*sz)

    Y = values[0] + (values[1]-values[0]) * numpy.cos(X)**2;

    # make sure end values are repeated, for extrapolation...
    Y[0] = Y[1]
    Y[sz+2] = Y[sz+1]
    
    X = position + (2*width/numpy.pi) * (X + numpy.pi/4)

    return (X,Y)
