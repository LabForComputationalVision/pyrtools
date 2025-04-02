import numpy as np
import warnings


def steer_to_harmonics_mtx(harmonics, angles=None, even_phase=True):
    '''Compute a steering matrix

    This maps a directional basis set onto the angular Fourier harmonics.

    Parameters
    ----------
    harmonics: `array_like`
        array specifying the angular harmonics contained in the steerable basis/filters.
    angles: `array_like` or None
        vector specifying the angular position of each filter (in radians). If None, defaults to
        `pi * np.arange(numh) / numh`, where `numh = harmonics.size + np.count_nonzero(harmonics)`
    even_phase : `bool`
        specifies whether the harmonics are cosine or sine phase aligned about those positions.

    Returns
    -------
    imtx : `np.array`
        This matrix is suitable for passing to the function `steer`.

    '''
    # default parameter
    numh = harmonics.size + np.count_nonzero(harmonics)
    if angles is None:
        angles = np.pi * np.arange(numh) / numh

    # Compute inverse matrix, which maps to Fourier components onto
    # steerable basis
    imtx = np.zeros((angles.size, numh))
    col = 0
    for h in harmonics:
        args = h * angles
        if h == 0:
            imtx[:, col] = np.ones(angles.shape)
            col += 1
        elif even_phase:
            imtx[:, col] = np.cos(args)
            imtx[:, col+1] = np.sin(args)
            col += 2
        else:  # odd phase
            imtx[:, col] = np.sin(args)
            imtx[:, col+1] = -1.0 * np.cos(args)
            col += 2

    r = np.linalg.matrix_rank(imtx)
    if r < np.min(imtx.shape):
        warnings.warn("Matrix is not full rank")

    return np.linalg.pinv(imtx)


def steer(basis, angle, harmonics=None, steermtx=None, return_weights=False, even_phase=True):
    '''Steer BASIS to the specfied ANGLE.

    Parameters
    ----------
    basis : `array_like`
        array whose columns are vectorized rotated copies of a steerable function, or the responses
        of a set of steerable filters.
    angle : `array_like` or `int`
        scalar or column vector the size of the basis. specifies the angle(s) (in radians) to
        steer to
    harmonics : `list` or None
        a list of harmonic numbers indicating the angular harmonic content of the basis. if None
        (default), N even or odd low frequencies, as for derivative filters
    steermtx : `array_like` or None.
        matrix which maps the filters onto Fourier series components (ordered [cos0 cos1 sin1 cos2
        sin2 ... sinN]). See steer_to_harmonics_mtx function for more details. If None (default),
        assumes cosine phase harmonic components, and filter positions at 2pi*n/N.
    return_weights : `bool`
        whether to return the weights or not.
    even_phase : `bool`
        specifies whether the harmonics are cosine or sine phase aligned about those positions.

    Returns
    -------
    res : `np.array`
        the resteered basis
    steervect : `np.array`
        the weights used to resteer the basis. only returned if `return_weights` is True
    '''

    num = basis.shape[1]

    if isinstance(angle, (int, float)):
        angle = np.asarray([angle])
    else:
        if angle.shape[0] != basis.shape[0] or angle.shape[1] != 1:
            raise Exception("""ANGLE must be a scalar, or a column vector
                                    the size of the basis elements""")

    # If HARMONICS is not specified, assume derivatives.
    if harmonics is None:
        harmonics = np.arange(1 - (num % 2), num, 2)

    if len(harmonics.shape) == 1 or harmonics.shape[0] == 1:
        # reshape to column matrix
        harmonics = harmonics.reshape(harmonics.shape[0], 1)
    elif harmonics.shape[0] != 1 and harmonics.shape[1] != 1:
        raise Exception('input parameter HARMONICS must be 1D!')

    if 2 * harmonics.shape[0] - (harmonics == 0).sum() != num:
        raise Exception('harmonics list is incompatible with basis size!')

    # If STEERMTX not passed, assume evenly distributed cosine-phase filters:
    if steermtx is None:
        steermtx = steer_to_harmonics_mtx(harmonics, np.pi * np.arange(num) / num,
                                          even_phase=even_phase)

    steervect = np.zeros((angle.shape[0], num))
    arg = angle * harmonics[np.nonzero(harmonics)[0]].T
    if all(harmonics):
        steervect[:, range(0, num, 2)] = np.cos(arg)
        steervect[:, range(1, num, 2)] = np.sin(arg)
    else:
        steervect[:, 0] = np.ones((arg.shape[0], 1))
        steervect[:, range(1, num, 2)] = np.cos(arg)
        steervect[:, range(2, num, 2)] = np.sin(arg)

    steervect = np.dot(steervect, steermtx)

    if steervect.shape[0] > 1:
        tmp = np.dot(basis, steervect)
        res = sum(tmp).T
    else:
        res = np.dot(basis, steervect.T)

    if return_weights:
        return res, np.asarray(steervect).reshape(num)
    else:
        return res
