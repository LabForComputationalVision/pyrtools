import numpy

def steer2HarmMtx(*args):
    ''' Compute a steering matrix (maps a directional basis set onto the
        angular Fourier harmonics).  HARMONICS is a vector specifying the
        angular harmonics contained in the steerable basis/filters.  ANGLES 
        (optional) is a vector specifying the angular position of each filter.  
        REL_PHASES (optional, default = 'even') specifies whether the harmonics 
        are cosine or sine phase aligned about those positions.
        The result matrix is suitable for passing to the function STEER.
        mtx = steer2HarmMtx(harmonics, angles, evenorodd)  '''

    if len(args) == 0:
        print "Error: first parameter 'harmonics' is required."
        return
    
    if len(args) > 0:
        harmonics = numpy.array(args[0])

    # optional parameters
    numh = (2*harmonics.shape[0]) - (harmonics == 0).sum()
    if len(args) > 1:
        angles = args[1]
    else:
        angles = numpy.pi * numpy.array(range(numh)) / numh
        
    if len(args) > 2:
        if isinstance(args[2], basestring):
            if args[2] == 'even' or args[2] == 'EVEN':
                evenorodd = 0
            elif args[2] == 'odd' or args[2] == 'ODD':
                evenorodd = 1
            else:
                print "Error: only 'even' and 'odd' are valid entries for the third input parameter."
                return
        else:
            print "Error: third input parameter must be a string (even/odd)."
    else:
        evenorodd = 0

    # Compute inverse matrix, which maps to Fourier components onto 
    #   steerable basis
    imtx = numpy.zeros((angles.shape[0], numh))
    col = 0
    for h in harmonics:
        args = h * angles
        if h == 0:
            imtx[:, col] = numpy.ones(angles.shape)
            col += 1
        elif evenorodd:
            imtx[:, col] = numpy.sin(args)
            imtx[:, col+1] = numpy.negative( numpy.cos(args) )
            col += 2
        else:
            imtx[:, col] = numpy.cos(args)
            imtx[:, col+1] = numpy.sin(args)
            col += 2

    r = numpy.rank(imtx)
    if r != numh and r != angles.shape[0]:
        print "Warning: matrix is not full rank"

    mtx = numpy.linalg.pinv(imtx)
    
    return mtx
