import numpy
from .corrDn import corrDn

def imGradient(*args):
    ''' [dx, dy] = imGradient(im, edges) 
    
        Compute the gradient of the image using smooth derivative filters
        optimized for accurate direction estimation.  Coordinate system
        corresponds to standard pixel indexing: X axis points rightward.  Y
        axis points downward.  EDGES specify boundary handling (see corrDn
        for options).
    
        EPS, 1997.
        original filters from Int'l Conf Image Processing, 1994.
        updated filters 10/2003: see Farid & Simoncelli, IEEE Trans Image 
                                 Processing, 13(4):496-508, April 2004.
        Incorporated into matlabPyrTools 10/2004.
        Python port by Rob Young, 10/15  '''
    
    if len(args) == 0 or len(args) > 2:
        print('Usage: imGradient(image, edges)')
        print("'edges' argument is optional")
    elif len(args) == 1:
        edges = "dont-compute"
    elif len(args) == 2:
        edges = str(args[1])
        
    im = numpy.array(args[0])

    # kernels from Farid & Simoncelli, IEEE Trans Image Processing, 
    #   13(4):496-508, April 2004.
    gp = numpy.array([0.037659, 0.249153, 0.426375, 0.249153, 0.037659]).reshape(5,1)
    gd = numpy.array([-0.109604, -0.276691, 0.000000, 0.276691, 0.109604]).reshape(5,1)

    dx = corrDn(corrDn(im, gp, edges), gd.T, edges)
    dy = corrDn(corrDn(im, gd, edges), gp.T, edges)

    return (dx,dy)
