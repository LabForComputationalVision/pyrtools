import numpy as np
from .utils import matlab_histo


def entropy(vec, binsize=None):
    ''' E = entropy(vec, binsize=None):

        Compute the first-order sample entropy of MTX.  Samples of VEC are
        first discretized.  Optional argument BINSIZE controls the
        discretization, and defaults to 256/(max(VEC)-min(VEC)).

        NOTE: This is a heavily  biased estimate of entropy when you
        don't have much data.

        Eero Simoncelli, 6/96. Ported to Python by Rob Young, 10/15.  '''

    [bincount, _] = matlab_histo(vec, nbins=256, binsize=binsize)

    # Collect non-zero bins:
    H = bincount[np.where(bincount > 0)]
    H = H / H.sum()

    return -(H * np.log2(H)).sum()


def range(np_array):
    ''' compute minimum and maximum values of the input numpy array,
        returning them as tuple
        '''
    if not np.isreal(np_array.all()):
        print('Error: matrix must be real-valued')

    return (np_array.min(), np_array.max())


def var(np_array, img_mean=None):
    ''' Sample variance of the input numpy array.
        Passing MEAN (optional) makes the calculation faster.  '''

    if img_mean is None:
        img_mean = np_array.mean()

    if np.isreal(np_array).all():
        return ((np_array - img_mean)**2).sum() / max(np_array.size - 1, 1)
    else:
        return var(np_array.real, img_mean.real) + 1j * var(np_array.imag, img_mean.imag)


def skew(np_array, img_mean=None, img_var=None):
    ''' Sample skew (third moment divided by variance^3/2) of the input numpy array.
        MEAN (optional) and VAR (optional) make the computation faster.  '''

    if img_mean is None:
        img_mean = np_array.mean()
    if img_var is None:
        img_var = var(np_array, img_mean)

    if np.isreal(np_array).all():
        return ((np_array - img_mean)**3).mean() / np.sqrt(img_var) ** 3
    else:
        return (skew(np_array.real, img_mean.real, img_var.real) + 1j *
                skew(np_array.imag, img_mean.imag, img_var.imag))


def kurt(np_array, img_mean=None, img_var=None):
    ''' Sample kurtosis (fourth moment divided by squared variance)
        of the input numpy array.  Kurtosis of a Gaussian distribution is 3.
        MEAN (optional) and VAR (optional) make the computation faster.  '''

    if img_mean is None:
        img_mean = np_array.mean()
    if img_var is None:
        img_var = var(np_array, img_mean)

    if np.isreal(np_array).all():
        return ((np_array - img_mean) ** 4).mean() / img_var ** 2
    else:
        return (kurt(np_array.real, img_mean.real, img_var.real) + 1j *
                kurt(np_array.imag, img_mean.imag, img_var.imag))


def image_compare(im_array0, im_array1):
    ''' Report min, max, mean, stdev of the difference,
        and SNR (relative to IM1).  '''

    if not im_array0.size == im_array1.size:
        print('Error: input images must have the same size')
        return

    if not np.isreal(im_array0).all() or not np.isreal(im_array1).all():
        print('Error: input images must be real-valued matrices')
        return

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


def image_stats(im_array):
    ''' Report image (matrix) statistics: min, max, mean, stdev,
        and kurtosis.
        '''

    if not np.isreal(im_array).all():
        print('Error: input images must be real-valued matrices')
        return

    (mini, maxi) = range(im_array)
    img_mean = im_array.mean()
    img_var = var(im_array, img_mean)
    img_kurt = kurt(im_array, img_mean, img_var)
    print('Image statistics:')
    print('  Range: [%f, %f]' % (mini, maxi))
    print('  Mean: %f,  Stdev: %f,  Kurtosis: %f' % (img_mean, np.sqrt(img_var), img_kurt))
