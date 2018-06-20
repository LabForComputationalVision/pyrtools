import numpy

def modulateFlip(*args):
    ''' [HFILT] = modulateFlipShift(LFILT)
        QMF/Wavelet highpass filter construction: modulate by (-1)^n,
        reverse order (and shift by one, which is handled by the convolution
        routines).  This is an extension of the original definition of QMF's
        (e.g., see Simoncelli90).  '''

    if len(args) == 0:
        print("Error: filter input parameter required.")
        return

    lfilt = args[0]
    # reshape lfilt to column vector
    if len(lfilt.shape) == 1:
        lfilt = lfilt.reshape(len(lfilt), 1)
    elif lfilt.shape[0] == 1:
        lfilt = lfilt.reshape(lfilt.shape[1], 1)
    elif len(lfilt.shape) > 2 or lfilt.shape[1] != 1:
        print('Error: only 1D input supported.')
        return

    sz = len(lfilt)
    sz2 = numpy.ceil(sz/2.0);

    ind = numpy.array(list(range(sz-1,-1,-1)))

    hfilt = lfilt[ind].T * (-1)**((ind+1)-sz2)

    # matlab version always returns a column vector
    if len(hfilt.shape) == 1:
        hfilt = hfilt.reshape(len(hfilt), 1)
    elif hfilt.shape[0] == 1:
        hfilt = hfilt.reshape(hfilt.shape[1], 1)

    return hfilt
