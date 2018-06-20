import numpy

def var2(*args):
    ''' Sample variance of a matrix.
        Passing MEAN (optional) makes the calculation faster.  '''

    if len(args) == 1:
        mn = args[0].mean()
    elif len(args) == 2:
        mn = args[1]
    
    if(numpy.isreal(args[0]).all()):
        res = sum(sum((args[0]-mn)**2)) / max(numpy.prod(args[0].shape)-1, 1)
    else:
        res = sum((args[0]-mn).real**2) + 1j*sum((args[0]-mn).imag)**2
        res = res /  max(numpy.prod(args[0].shape)-1, 1)

    return res
