import numpy
from .range2 import range2
from .var2 import var2
from .kurt2 import kurt2

def imStats(*args):
    ''' Report image (matrix) statistics.
        When called on a single image IM1, report min, max, mean, stdev, 
        and kurtosis.
        When called on two images (IM1 and IM2), report min, max, mean, 
        stdev of the difference, and also SNR (relative to IM1).  '''

    if len(args) == 0:
        print('Error: at least one input image is required')
        return
    elif len(args) == 1 and not numpy.isreal(args[0]).all():
        print('Error: input images must be real-valued matrices')
        return
    elif len(args) == 2 and ( not numpy.isreal(args[0]).all() or not numpy.isreal(args[1]).all()):
        print('Error: input images must be real-valued matrices')
        return
    elif len(args) > 2:
        print('Error: maximum of two input images allowed')
        return

    if len(args) == 2:
        difference = args[0] - args[1]
        (mn, mx) = range2(difference)
        mean = difference.mean()
        v = var2(difference)
        if v < numpy.finfo(numpy.double).tiny:
            snr = numpy.inf
        else:
            snr = 10 * numpy.log10(var2(args[0])/v)
        print('Difference statistics:')
        print('  Range: [%d, %d]' % (mn, mx))
        print('  Mean: %f,  Stdev (rmse): %f,  SNR (dB): %f' % (mean,
                                                                numpy.sqrt(v),
                                                                snr))
    else:
        (mn, mx) = range2(args[0])
        mean = args[0].mean()
        var = var2(args[0])
        stdev = numpy.sqrt(var.real) + numpy.sqrt(var.imag)
        kurt = kurt2(args[0], mean, stdev**2)
        print('Image statistics:')
        print('  Range: [%f, %f]' % (mn, mx))
        print('  Mean: %f,  Stdev: %f,  Kurtosis: %f' % (mean, stdev, kurt))
