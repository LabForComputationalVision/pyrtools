import numpy as np
from .utils import matlab_histo


def entropy(vec, binsize=None):
    '''Compute the first-order sample entropy of `vec`

    Samples of `vec` are first discretized.  Optional argument `binsize` controls the
    discretization, and defaults to 256/(max(`vec`)-min(`vec`)).

    NOTE: This is a heavily biased estimate of entropy when you don't have much data.

    Arguments
    ---------
    vec : `array_like`
        the 2d or 2d array to calculate the entropy of
    binsize : `float` or None
        the size of the bins we discretize into. If None, will set to 256/(max(vec)-min(vec))

    Returns
    -------
    entropy : `float`
        estimate of entropy from the data

    '''
    [bincount, _] = matlab_histo(vec, nbins=256, binsize=binsize)

    # Collect non-zero bins:
    H = bincount[np.where(bincount > 0)]
    H = H / H.sum()

    return -(H * np.log2(H)).sum()


def range(array):
    '''compute minimum and maximum values of the input array

    `array` must be real-valued

    Arguments
    ---------
    array : `np.array`
        array to calculate the range of

    Returns
    -------
    array_range : `tuple`
        (min, max)
    '''
    if not np.isreal(array.all()):
        raise Exception('array must be real-valued')

    return (array.min(), array.max())


def var(array, array_mean=None):
    '''Sample variance of the input numpy array.

    Passing `mean` (optional) makes the calculation faster. This works equally well for real and
    complex-valued `array`

    Arguments
    ---------
    array : `np.array`
        array to calculate the variance of
    array_mean : `float` or None
        the mean of `array`. If None, will calculate it.

    Returns
    -------
    array_var : `float`
        the variance of `array`
    '''
    if array_mean is None:
        array_mean = array.mean()

    if np.isreal(array).all():
        return ((array - array_mean)**2).sum() / max(array.size - 1, 1)
    else:
        return var(array.real, array_mean.real) + 1j * var(array.imag, array_mean.imag)


def skew(array, array_mean=None, array_var=None):
    '''Sample skew (third moment divided by variance^3/2) of the input array.

    `mean` (optional) and `var` (optional) make the computation faster. This works equally well for
    real and complex-valued `array`

    Arguments
    ---------
    array : `np.array`
        array to calculate the variance of
    array_mean : `float` or None
        the mean of `array`. If None, will calculate it.
    array_var : `float` or None
        the variance of `array`. If None, will calculate it

    Returns
    -------
    array_skew : `float`
        the skew of `array`.

    '''
    if array_mean is None:
        array_mean = array.mean()
    if array_var is None:
        array_var = var(array, array_mean)

    if np.isreal(array).all():
        return ((array - array_mean)**3).mean() / np.sqrt(array_var) ** 3
    else:
        return (skew(array.real, array_mean.real, array_var.real) + 1j *
                skew(array.imag, array_mean.imag, array_var.imag))


def kurt(array, array_mean=None, array_var=None):
    '''Sample kurtosis (fourth moment divided by squared variance) of the input array.

    For reference, kurtosis of a Gaussian distribution is 3.

    `mean` (optional) and `var` (optional) make the computation faster. This works equally well for
    real and complex-valued `array`

    Arguments
    ---------
    array : `np.array`
        array to calculate the variance of
    array_mean : `float` or None
        the mean of `array`. If None, will calculate it.
    array_var : `float` or None
        the variance of `array`. If None, will calculate it

    Returns
    -------
    array_kurt : `float`
        the kurtosis of `array`.

    '''
    if array_mean is None:
        array_mean = array.mean()
    if array_var is None:
        array_var = var(array, array_mean)

    if np.isreal(array).all():
        return ((array - array_mean) ** 4).mean() / array_var ** 2
    else:
        return (kurt(array.real, array_mean.real, array_var.real) + 1j *
                kurt(array.imag, array_mean.imag, array_var.imag))


def image_compare(im_array0, im_array1):
    '''Prints and returns min, max, mean, stdev of the difference, and SNR (relative to im_array0).

    Arguments
    ---------
    im_array0 : `np.array`
        the first image to compare
    im_array1 : `np.array`
        the second image to compare

    Returns
    -------
    min_diff : `float`
        the minimum difference between `im_array0` and `im_array1`
    max_diff : `float`
        the maximum difference between `im_array0` and `im_array1`
    mean_diff : `float`
        the mean difference between `im_array0` and `im_array1`
    std_diff : `float`
        the standard deviation of the difference between `im_array0` and `im_array1`
    snr : `float`
        the signal-to-noise ratio of the difference between `im_array0` and `im_array0` (relative
        to `im_array0`)
    '''
    if not im_array0.size == im_array1.size:
        raise Exception('Input images must have the same size')

    if not np.isreal(im_array0).all() or not np.isreal(im_array1).all():
        raise Exception('Input images must be real-valued matrices')

    difference = im_array0 - im_array1
    (min_diff, max_diff) = range(difference)
    mean_diff = difference.mean()
    var_diff = var(difference, mean_diff)
    if var_diff < np.finfo(np.double).tiny:
        snr = np.inf
    else:
        snr = 10 * np.log10(var(im_array0) / var_diff)
    print('Difference statistics:')
    print('  Range: [%d, %d]' % (min_diff, max_diff))
    print('  Mean: %f,  Stdev (rmse): %f,  SNR (dB): %f' % (mean_diff, np.sqrt(var_diff), snr))
    return min_diff, max_diff, mean_diff, np.sqrt(var_diff), snr


def image_stats(im_array):
    '''Prints and returns image statistics: min, max, mean, stdev, and kurtosis.

    Arguments
    ---------
    im_array : `np.array`
        the image to summarize

    Returns
    -------
    array_min : `float`
        the minimum of `im_array`
    array_max : `float`
        the maximum of `im_array`
    array_mean : `float`
        the mean of `im_array`
    array_std : `float`
        the standard deviation of `im_array`
    array_kurt : `float`
        the kurtosis of `im_array`
    '''
    if not np.isreal(im_array).all():
        raise Exception('Input images must be real-valued matrices')

    (mini, maxi) = range(im_array)
    array_mean = im_array.mean()
    array_var = var(im_array, array_mean)
    array_kurt = kurt(im_array, array_mean, array_var)
    print('Image statistics:')
    print('  Range: [%f, %f]' % (mini, maxi))
    print('  Mean: %f,  Stdev: %f,  Kurtosis: %f' % (array_mean, np.sqrt(array_var), array_kurt))
    return mini, maxi, array_mean, np.sqrt(array_var), array_kurt
