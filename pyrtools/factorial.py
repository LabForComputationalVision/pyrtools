import numpy

def factorial(*args):
    ''' RES = factorial(NUM)
    
        Factorial function that works on matrices (matlab's does not).
    
        EPS, 11/02, Python port by Rob Young, 10/15  '''

    # if scalar input make it a single element array
    if isinstance(args[0], (int, float)):
        num = numpy.array([args[0]])
    else:
        num = numpy.array(args[0])

    res = numpy.ones(num.shape)

    ind = numpy.where(num > 0)
        
    if num.shape[0] != 0:
        subNum = num[ numpy.where(num > 0) ]
        res[ind] = subNum * factorial(subNum-1)

    # if scalar input, return scalar
    if len(res.shape) == 1 and res.shape[0] == 1:
        return res[0]
    else:
        return res
