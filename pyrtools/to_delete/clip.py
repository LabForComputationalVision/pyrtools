import numpy

def clip(*args):
    ''' [RES] = clip(IM, MINVALorRANGE, MAXVAL)
    
        Clip values of matrix IM to lie between minVal and maxVal:
             RES = max(min(IM,MAXVAL),MINVAL)
        The first argument can also specify both min and max, as a 2-vector.
        If only one argument is passed, the range defaults to [0,1].
        ported to Python by Rob Young, 8/15  '''
    
    if len(args) == 0 or len(args) > 3:
        print('Usage: clip(im, minVal or Range, maxVal)')
        print('first input parameter is required')
        return
        
    im = numpy.array(args[0])

    if len(args) == 1:
        minVal = 0;
        maxVal = 1;
    elif len(args) == 2:
        if isinstance(args[1], (int, float)):
            minVal = args[1]
            maxVal = args[1]+1
        else:
            minVal = args[1][0]
            maxVal = args[1][1]
    elif len(args) == 3:
        minVal = args[1]
        maxVal = args[2]
        
    if maxVal < minVal:
        print('Error: maxVal cannot be less than minVal!')
        return

    im[numpy.where(im < minVal)] = minVal
    im[numpy.where(im > maxVal)] = maxVal

    return im
