import numpy as np
import math
from ..pyramids import convert_pyr_coeffs_to_pyr


def comparePyr(matPyr, pyPyr, rtol=1e-5, atol=1e-8):
    '''compare two pyramids

    returns True if they are the same with in desired precision and False if not.

    written for unit testing code.

    '''
    # compare two pyramids - return 0 for !=, 1 for ==
    # correct number of elements?
    matSz = sum(matPyr.shape)
    try:
        pySz = 1 + sum([np.asarray(size).prod() for size in pyPyr.pyr_size.values()])
    except AttributeError:
        pySz = 1 + sum([np.asarray(size).prod() for size in pyPyr.pyrSize])

    if(matSz != pySz):
        print("size difference: %d != %d, returning False" % (matSz, pySz))
        return False

    # values are close to each other?
    matStart = 0
    try:
        pyCoeffs, pyHigh, pyLow = convert_pyr_coeffs_to_pyr(pyPyr.pyr_coeffs)
        if pyHigh is not None:
            pyCoeffs.insert(0, pyHigh)
        if pyLow is not None:
            pyCoeffs.append(pyLow)
    except AttributeError:
        pyCoeffs = pyPyr.pyr
    for idx, pyTmp in enumerate(pyCoeffs):
        matTmp = matPyr[matStart:matStart + pyTmp.size]
        matStart = matStart + pyTmp.size
        matTmp = np.reshape(matTmp, pyTmp.shape, order='F')

        # relative tolerance rtol
        # absolute tolerance atol
        isclose = np.isclose(matTmp, pyTmp, rtol, atol)
        if not isclose.all():
            print("some pyramid elements not identical: checking...")
            for i in range(isclose.shape[0]):
                for j in range(isclose.shape[1]):
                    if not isclose[i, j]:
                        print("failed level:%d element:%d %d value:%.15f %.15f" %
                              (idx, i, j, matTmp[i, j], pyTmp[i, j]))
                        return False

    return True


def compareRecon(recon1, recon2, rtol=1e-5, atol=1e-10):
    '''compare two arrays

    returns True is they are the same within specified precision and False if not.  function was
    made to accompany unit test code.

    This function is deprecated. Instead use the builtin numpy:

    np.allclose(recon1, recon2, rtol=1e-05, atol=1e-08, equal_nan=False)

    This will not tell you where the error is, but you can find that yourself

    '''

    # NOTE builtin numpy:

    # BUT won't print where and what the first error is

    if recon1.shape != recon2.shape:
        print('shape is different!')
        print(recon1.shape)
        print(recon2.shape)
        return False

    prec = -1
    for i in range(recon1.shape[0]):
        for j in range(recon2.shape[1]):
            if np.absolute(recon1[i, j].real - recon2[i, j].real) > math.pow(10, -11):
                print("real: i=%d j=%d %.15f %.15f diff=%.15f" %
                      (i, j, recon1[i, j].real, recon2[i, j].real,
                       np.absolute(recon1[i, j].real-recon2[i, j].real)))
                return False
            # FIX: need a better way to test
            # if we have many significant digits to the left of decimal we
            # need to be less stringent about digits to the right.
            # The code below works, but there must be a better way.
            if isinstance(recon1, complex):
                if int(math.log(np.abs(recon1[i, j].imag), 10)) > 1:
                    prec = prec + int(math.log(np.abs(recon1[i, j].imag), 10))
                    if prec > 0:
                        prec = -1
                print(prec)
                if np.absolute(recon1[i, j].imag - recon2[i, j].imag) > math.pow(10, prec):
                    print("imag: i=%d j=%d %.15f %.15f diff=%.15f" %
                          (i, j, recon1[i, j].imag, recon2[i, j].imag,
                           np.absolute(recon1[i, j].imag-recon2[i, j].imag)))
                    return False

    return True
