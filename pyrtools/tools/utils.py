import numpy as np
import warnings


def matlab_round(array):
    '''round equivalent to matlab function, which rounds .5 away from zero

    (used in matlab_histo so we can unit test against matlab code).

    But np.round() would rounds .5 to nearest even number, e.g.:
    - np.round(0.5) = 0, matlab_round(0.5) = 1
    - np.round(2.5) = 2, matlab_round(2.5) = 3

    Arguments
    ---------
    array : `np.array`
        the array to round

    Returns
    -------
    rounded_array : `np.array`
        the rounded array
    '''
    (fracPart, intPart) = np.modf(array)
    return intPart + (np.abs(fracPart) >= 0.5) * np.sign(fracPart)


def matlab_histo(array, nbins=101, binsize=None, center=None):
    '''Compute a histogram of array.

    How does this differ from MatLab's HIST function?  This function:
      - allows uniformly spaced bins only.
      + operates on all elements of MTX, instead of columnwise.
      + is much faster (approximately a factor of 80 on my machine).
      + allows specification of number of bins OR binsize. Default=101 bins.
      + allows (optional) specification of binCenter.

    Arguments
    ---------
    array : `np.array`
        the array to bin
    nbins : `int`
        the number of histogram bins
    binsize : `float` or None
        the size of each bin. if None, we use nbins to determine it as:
        `(array.max() - array.min()) / nbins`
    center : `float` or None
        the center position for the histogram bins. if None, this is `array.mean()`

    Returns
    -------
    N : `np.array`
        the histogram counts
    edges : `np.array`
        vector containing the centers of the histogram bins
    '''
    mini = array.min()
    maxi = array.max()

    if center is None:
        center = array.mean()

    if binsize is None:
        # use nbins to determine binsize
        binsize = (maxi-mini) / nbins

    nbins2 = int(matlab_round((maxi - center) / binsize) - matlab_round((mini - center) / binsize))
    if nbins2 != nbins:
        warnings.warn('Overriding bin number %d (requested %d)' % (nbins2, nbins))
        nbins = nbins2

    # np.histogram uses bin edges, not centers like Matlab's hist
    # compute bin edges (nbins + 1 of them)
    edge_left = center + binsize * (-0.499 + matlab_round((mini - center) / binsize))
    edges = edge_left + binsize * np.arange(nbins+1)
    N, _ = np.histogram(array, edges)

    # matlab version returns column vectors, so we will too.
    # to check: return edges or centers? edit comments.
    return (N.reshape(1, -1), edges.reshape(1, -1))


def rcosFn(width=1, position=0, values=(0, 1)):
    '''Return a lookup table containing a "raised cosine" soft threshold function

    Y =  VALUES(1) + (VALUES(2)-VALUES(1)) * cos^2( PI/2 * (X - POSITION + WIDTH)/WIDTH )

    this lookup table is suitable for use by `pointOp`

    Arguments
    ---------
    width : `float`
        the width of the region over which the transition occurs
    position : `float`
        the location of the center of the threshold
    values : `tuple`
        2-tuple specifying the values to the left and right of the transition.

    Returns
    -------
    X : `np.array`
        the x valuesof this raised cosine
    Y : `np.array`
        the y valuesof this raised cosine
    '''

    sz = 256   # arbitrary!

    X = np.pi * np.arange(-sz-1, 2) / (2*sz)

    Y = values[0] + (values[1]-values[0]) * np.cos(X)**2

    # make sure end values are repeated, for extrapolation...
    Y[0] = Y[1]
    Y[sz+2] = Y[sz+1]

    X = position + (2*width/np.pi) * (X + np.pi/4)

    return (X, Y)


if __name__ == "__main__":
    X, Y = rcosFn(width=1, position=0, values=(0, 1))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(X, Y)
    plt.show()
