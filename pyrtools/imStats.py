import numpy as np

def range2(np_array):
    ''' compute minimum and maximum values of the input numpy array,
        returning them as tuple
        '''
    if not np.isreal(np_array.all():
        print('Error: matrix must be real-valued')

    return (np_array.min(), np_array.max())

def var2(np_array, mean=None):
    ''' Sample variance of the input numpy array.
        Passing MEAN (optional) makes the calculation faster.  '''

    if mean is None:
        mean = np_array.mean()

    if np.isreal(np_array).all():
        return ((np_array - mean)**2).sum() / max(np_array.size - 1, 1)
    else:
        return var2(np_array.real, mean.real) + 1j * var2(np_array.imag, mean.image)

def skew2(np_array, mean=None, var=None):
    ''' Sample skew (third moment divided by variance^3/2) of the input numpy array.
        MEAN (optional) and VAR (optional) make the computation faster.  '''

    if mean is None:
        mean = np_array.mean()
    if var is None:
        var = var2(np_array, mean)

    if np.isreal(np_array).all():
        return ((np_array - mean)**3).mean() / np.sqrt(var) ** 3
    else:
        return skew2(np_array.real, mean.real, var.real) + 1j * skew2(np_array.imag, mean.imag, var.imag)

def kurt2(np_array, mean=None, var=None):
    ''' Sample kurtosis (fourth moment divided by squared variance)
        of the input numpy array.  Kurtosis of a Gaussian distribution is 3.
        MEAN (optional) and VAR (optional) make the computation faster.  '''

    if mean is None:
        mean = np_array.mean()
    if var is None:
        var = var2(np_array, mean)

    if np.isreal(np_array).all():
        return ((np_array - mean) ** 4).mean() / var ** 2
    else:
        return kurt2(np_array.real, mean.real, var.real) + 1j * kurt2(np_array.imag, mean.imag, var.imag)

def imStats(*args):
    ''' Report image (matrix) statistics.
        When called on a single image IM1, report min, max, mean, stdev,
        and kurtosis.
        When called on two images (IM1 and IM2), report min, max, mean,
        stdev of the difference, and also SNR (relative to IM1).  '''

    if len(args) == 0:
        print('Error: at least one input image is required')
        return
    elif len(args) == 1 and not np.isreal(args[0]).all():
        print('Error: input images must be real-valued matrices')
        return
    elif len(args) == 2 and ( not np.isreal(args[0]).all() or not np.isreal(args[1]).all()):
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
        if v < np.finfo(np.double).tiny:
            snr = np.inf
        else:
            snr = 10 * np.log10(var2(args[0])/v)
        print('Difference statistics:')
        print('  Range: [%d, %d]' % (mn, mx))
        print('  Mean: %f,  Stdev (rmse): %f,  SNR (dB): %f' % (mean,
                                                                np.sqrt(v),
                                                                snr))
    else:
        (mn, mx) = range2(args[0])
        mean = args[0].mean()
        var = var2(args[0])
        stdev = np.sqrt(var.real) + np.sqrt(var.imag)
        kurt = kurt2(args[0], mean, stdev**2)
        print('Image statistics:')
        print('  Range: [%f, %f]' % (mn, mx))
        print('  Mean: %f,  Stdev: %f,  Kurtosis: %f' % (mean, stdev, kurt))
