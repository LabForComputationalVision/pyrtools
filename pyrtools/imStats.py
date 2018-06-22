import numpy as np
from .utils import matlab_round

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


def matlab_histo(np_array, nbins = 101, binsize = None, center = None):
    ''' [N,edges] = matlab_histo(np_array, nbins = 101, binsize = None, center = None)
        Compute a histogram of np_array.
        N contains the histogram counts,
        edges is a vector containg the centers of the histogram bins.

        nbins (optional, default = 101) specifies the number of histogram bins.
        binsize (optional) specifies the size of each bin.
        binCenter (optional, default = mean2(MTX)) specifies a center position
        for (any one of) the histogram bins.

        How does this differ from MatLab's HIST function?  This function:
          - allows uniformly spaced bins only.
          +/- operates on all elements of MTX, instead of columnwise.
          + is much faster (approximately a factor of 80 on my machine).
          + allows specification of number of bins OR binsize.
            Default=101 bins.
          + allows (optional) specification of binCenter.

        Eero Simoncelli, 3/97.  ported to Python by Rob Young, 8/15.  '''

    mini = np_array.min()
    maxi = np_array.max()

    if center is None:
        center = np_array.mean()

    if binsize is None:
        # use nbins to determine binsize
        binsize = (maxi-mini) / nbins

    nbins2 = int( matlab_round( (maxi - center) / binsize) - matlab_round( (mini - center) / binsize) )
    if nbins2 != nbins:
        print('Warning: Overriding bin number %d (requested %d)' % (nbins2, nbins))
        nbins = nbins2

    # numpy.histogram uses bin edges, not centers like Matlab's hist
    # compute bin edges (nbins + 1 of them)
    edge_left = center + binsize * (-0.499 + matlab_round( (mini - center) / binsize ))
    edges = edge_left + binsize * np.arange(nbins+1)
    N, _ = np.histogram(np_array, edges)

    # matlab version returns column vectors, so we will too.
    # to check: return edges or centers? edit comments.
    return (N.reshape(1,-1), edges.reshape(1,-1))


def entropy2(np_array, binsize=None):
    ''' E = entropy2(np_array, binsize=None):

        Compute the first-order sample entropy of MTX.  Samples of VEC are
        first discretized.  Optional argument BINSIZE controls the
        discretization, and defaults to 256/(max(VEC)-min(VEC)).

        NOTE: This is a heavily  biased estimate of entropy when you
        don't have much data.

        Eero Simoncelli, 6/96. Ported to Python by Rob Young, 10/15.  '''

    [bincount, _] = matlab_histo(vec, nbins=256, binsize=binsize)

    ## Collect non-zero bins:
    H = bincount[ np.where(bincount > 0) ]
    H = H / H.sum()

    return -(H * np.log2(H)).sum()


def imCompare(im_array0, im_array1):
    ''' Report min, max, mean, stdev of the difference,
        and SNR (relative to IM1).  '''
    if not np.isreal(im_array0).all() or not np.isreal(im_array1).all():
        print('Error: input images must be real-valued matrices')
        return

    difference = im_array0 - im_array1
    (min_diff, max_diff) = range2(difference)
    mean_diff = difference.mean()
    var_diff = var2(difference, mean)
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
