import numpy

def range2(*args):
    ''' compute minimum and maximum values of input matrix, returning them 
        as tuple  '''

    if not numpy.isreal(args[0]).all():
        print('Error: matrix must be real-valued')

    return (args[0].min(), args[0].max())
