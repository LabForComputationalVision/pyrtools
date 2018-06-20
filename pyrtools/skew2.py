import numpy
from .var2 import var2

def skew2(*args):
    ''' Sample skew (third moment divided by variance^3/2) of a matrix.
        MEAN (optional) and VAR (optional) make the computation faster.  '''

    if len(args) == 0:
        print('Usage: skew2(matrix, mean, variance)')
        print('mean and variance arguments are optional')
    else:
        mtx = numpy.array(args[0])

    if len(args) > 1:
        mn = args[1]
    else:
        mn = mtx.mean()

    if len(args) > 2:
        v = args[2]
    else:
        v = var2(mtx, mn)

    if isinstance(mtx, complex):
        res = ( ( ((mtx.real - mn.real)**3).mean() / (v.real**(3.0/2.0)) ) +
                ( (1j * (mtx.imag-mn.image)**3) / (v.imag**(3.0/2.0))))
    else:
        res = ((mtx.real - mn.real)**3).mean() / (v.real**(3.0/2.0))

    return res
