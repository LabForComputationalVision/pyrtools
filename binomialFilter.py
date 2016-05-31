import numpy
import scipy.signal

def binomialFilter(size):
    ''' returns a vector of binomial coefficients of order (size-1) '''
    if size < 2:
        print "Error: size argument must be larger than 1"
        exit(1)
    
    kernel = numpy.array([[0.5], [0.5]])

    for i in range(0, size-2):
        kernel = scipy.signal.convolve(numpy.array([[0.5], [0.5]]), kernel)

    return numpy.asarray(kernel)
