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

def convert_pyr_coeffs_to_pyr(pyr_coeffs):
    """this function takes a 'new pyramid' and returns the coefficients as a list

    returns them in original order, so 'residual highpass', all the
    bands, 'residual low pass'
    
    this is to enable backwards compatibility, will be deprecated

    """
    # we first remove the residual high and lowpass from the
    # pyramid coefficients dictionary, then grab the rest of
    # the bands in ascending order, putting those residuals at
    # the end
    coeffs = [pyr_coeffs.pop('residual_highpass')]
    coeffs.append(pyr_coeffs.pop('residual_lowpass'))
    coeffs = [coeffs[0]] + [i[1] for i in sorted(pyr_coeffs.items(), key=lambda x: x[0])] + [coeffs[-1]]
    return coeffs
    
