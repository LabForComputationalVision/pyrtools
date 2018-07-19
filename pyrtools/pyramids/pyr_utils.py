import math
import numpy as np

def LB2idx(lev, band, nlevs, nbands):
    ''' convert level and band to dictionary index '''
    # reset band to match matlab version
    band += (nbands-1)
    if band > nbands-1:
        band = band - nbands

    if lev == 0:
        idx = 0
    elif lev == nlevs-1:
        # (Nlevels - ends)*Nbands + ends -1 (because zero indexed)
        idx = (((nlevs-2)*nbands)+2)-1
    else:
        # (level-first level) * nbands + first level + current band
        idx = (nbands*lev)-band - 1

    return idx

def idx2LB(idx, nlevs, nbands):
    ''' given an index into dictionary return level and band '''

    if idx == 0:
        return ('hi', -1)
    elif idx == ((nlevs-2)*nbands)+1:
        return ('lo', -1)
    else:
        lev = math.ceil(idx/nbands)
        band = (idx % nbands) + 1
        if band == nbands:
            band = 0
        return (lev, band)

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
