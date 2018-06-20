import numpy

def kurt2(*args):
    ''' Sample kurtosis (fourth moment divided by squared variance) 
        of a matrix.  Kurtosis of a Gaussian distribution is 3.
        MEAN (optional) and VAR (optional) make the computation faster.  '''

    if len(args) == 0:
        print('Error: input matrix is required')

    if len(args) < 2:
        mn = args[0].mean()
    else:
        mn = args[1]

    if len(args) < 3:
        v = var2(args[0])
    else:
        v = args[2]

    if numpy.isreal(args[0]).all():
        res = (numpy.abs(args[0]-mn)**4).mean() / v**2
    else:
        res = ( (((args[0]-mn).real**4).mean() / v.real**2) + 
                ((numpy.i * (args[0]-mn).imag**4).mean() / v.imag**2) )

    return res
