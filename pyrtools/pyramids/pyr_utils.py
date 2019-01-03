import math
import functools
from operator import mul


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

    returns the in original order, so 'residual highpass', all the
    bands, 'residual low pass'

    this is to enable backwards compatibility, will be deprecated

    """
    highpass = pyr_coeffs.pop('residual_highpass', None)
    lowpass = pyr_coeffs.pop('residual_lowpass', None)
    coeffs = [i[1] for i in sorted(pyr_coeffs.items(), key=lambda x: x[0])]
    return coeffs, highpass, lowpass


def max_pyr_height(imsz, filtsz):
    ''' Compute maximum pyramid height for given image and filter sizes.
        Specifically: the number of corrDn operations that can be sequentially
        performed when subsampling by a factor of 2. '''
    # check if inputs are one of int, tuple and have consistent type
    assert (isinstance(imsz, int) and isinstance(filtsz, int)) or (
            isinstance(imsz, tuple) and isinstance(filtsz, tuple))
    # 1D image case: reduce to the integer case
    if isinstance(imsz, tuple) and (len(imsz) == 1 or 1 in imsz):
        imsz = functools.reduce(mul, imsz)
        filtsz = functools.reduce(mul, filtsz)
    # integer case
    if isinstance(imsz, int):
        if imsz < filtsz:
            return 0
        else:
            return 1 + max_pyr_height(imsz // 2, filtsz)
    # 2D image case
    if isinstance(imsz, tuple):
        if min(imsz) < max(filtsz):
            return 0
        else:
            return 1 + max_pyr_height((imsz[0] // 2, imsz[1] // 2), filtsz)
