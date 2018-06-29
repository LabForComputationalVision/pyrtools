import numpy as np

def range2(np_array):
    ''' compute minimum and maximum values of the input numpy array,
        returning them as tuple
        '''
    if not np.isreal(np_array.all()):
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
        return var2(np_array.real, mean.real) + 1j * var2(np_array.imag, mean.imag)

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

def imCompare(im_array0, im_array1):
    ''' Report min, max, mean, stdev of the difference,
        and SNR (relative to IM1).  '''

    if not np.isreal(im_array0).all() or not np.isreal(im_array1).all():
        print('Error: input images must be real-valued matrices')
        return

    difference = im_array0 - im_array1
    (min_diff, max_diff) = range2(difference)
    mean_diff = difference.mean()
    var_diff = var2(difference, mean_diff)
    if var_diff < np.finfo(np.double).tiny:
        snr = np.inf
    else:
        snr = 10 * np.log10(var2(im_array0) / var_diff)
    print('Difference statistics:')
    print('  Range: [%d, %d]' % (min_diff, max_diff))
    print('  Mean: %f,  Stdev (rmse): %f,  SNR (dB): %f' % (mean_diff,
                                                            np.sqrt(var_diff),
                                                            snr))

def imStats(im_array):
    ''' Report image (matrix) statistics: min, max, mean, stdev,
        and kurtosis.
        '''

    if not np.isreal(im_array).all():
        print('Error: input images must be real-valued matrices')
        return

    (mini, maxi) = range2(im_array)
    mean = im_array.mean()
    var = var2(im_array, mean)
    kurt = kurt2(im_array, mean, var)
    print('Image statistics:')
    print('  Range: [%f, %f]' % (mini, maxi))
    print('  Mean: %f,  Stdev: %f,  Kurtosis: %f' % (mean, np.sqrt(var), kurt))
