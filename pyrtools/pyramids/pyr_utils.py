import math
import functools
from operator import mul


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
