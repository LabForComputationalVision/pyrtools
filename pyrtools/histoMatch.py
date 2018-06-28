import numpy as np
from scipy.interpolate import interp1d
from .imStats import matlab_histo
from .convolutions import pointOp

def histoMatch(mtx, N, X, mode='edges'):
    '''Modify elements of MTX so that normalized histogram matches that
    specified by vectors X and N, where N contains the histogram counts
    and X the histogram bin positions (see matlab_histo).

    RES = histoMatch(MTX, N, X, mode)

    new input parameter 'mode' can be either 'centers' or 'edges' that tells
    the function if the input X values are bin centers or edges.

    Eero Simoncelli, 7/96. Ported to Python by Rob Young, 10/15.
    '''

    [oN, oX] = matlab_histo(mtx, N.size)
    oStep = oX[0,1] - oX[0,0]
    oC = np.concatenate(( np.array([0]), np.cumsum(oN / oN.sum()) ))

    if mode == 'centers':         # convert to edges
        nStep = X[0,1] - X[0,0]
        nX = np.concatenate((np.array([X[0,0] - 0.5 * nStep]),
                                np.array( X[0,:] + 0.5 * nStep)))
    else:
        nX = X.flatten()

    # HACK: no empty bins ensures nC strictly monotonic
    N = N + N.mean() / 1e8
    nC = np.concatenate((np.array([0]), np.cumsum(N / N.sum()) ))

    # TODO - scipy error
    # ValueError: A value in x_new is above the interpolation range.

    # unlike in matlab, interp1d returns a function
    func = interp1d(nC, nX, 'linear')
    nnX = func(oC)

    return pointOp(image=mtx, lut=nnX, origin=oX[0,0], increment=oStep, warnings=0)
