import numpy as np

def modulateFlip(lfilt):
    ''' [HFILT] = modulateFlipShift(LFILT)
        QMF/Wavelet highpass filter construction: modulate by (-1)^n,
        reverse order (and shift by one, which is handled by the convolution
        routines).  This is an extension of the original definition of QMF's
        (e.g., see Simoncelli90).  '''
    assert lfilt.size == max(lfilt.shape)
    lfilt = lfilt.flatten()
    ind = np.arange(lfilt.size,0,-1) - (lfilt.size + 1) // 2
    hfilt = lfilt[::-1] * (-1.0) ** ind
    # matlab version always returns a column vector
    return hfilt.reshape(-1,1)
