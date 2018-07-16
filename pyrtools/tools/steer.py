import numpy as np

def steer2HarmMtx(harmonics, angles=None, even_phase=True):
    ''' Compute a steering matrix (maps a directional basis set onto the
        angular Fourier harmonics).

        HARMONICS is a vector specifying the angular harmonics contained in the
        steerable basis/filters.
        ANGLES (optional) is a vector specifying the angular position of each
        filter.
        EVEN_PHASE (optional, default = True) specifies whether the harmonics
        are cosine or sine phase aligned about those positions.

        The result matrix is suitable for passing to the function STEER.
        '''

    # default parameter
    numh = harmonics.size +  np.count_nonzero(harmonics)
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
        else: # odd phase
            imtx[:, col] = np.sin(args)
            imtx[:, col+1] = -1.0 * np.cos(args)
            col += 2

    r = np.linalg.matrix_rank(imtx)
    if r < np.min(imtx.shape):
        print("Warning: matrix is not full rank")

    return np.linalg.pinv(imtx)

def steer(basis, angle, harmonics=None, steermtx=None):
    '''Steer BASIS to the specfied ANGLE.

    BASIS should be a matrix whose columns are vectorized rotated copies of a steerable
    function, or the responses of a set of steerable filters.

    ANGLE can be a scalar, or a column vector the size of the basis.

    HARMONICS (optional, default is N even or odd low frequencies, as for derivative filters)
    should be a list of harmonic numbers indicating the angular harmonic content of the basis.

    STEERMTX (optional, default assumes cosine phase harmonic components, and filter positions at
    2pi*n/N) should be a matrix which maps the filters onto Fourier series components (ordered
    [cos0 cos1 sin1 cos2 sin2 ... sinN]).  See steer2HarmMtx function for more details

    Eero Simoncelli, 7/96. Ported to Python by Rob Young, 5/14.
    '''

    num = basis.shape[1]

    if isinstance(angle, (int, float)):
        angle = np.array([angle])
    else:
        if angle.shape[0] != basis.shape[0] or angle.shape[1] != 1:
            raise Exception('ANGLE must be a scalar, or a column vector the size of the basis elements')

    # If HARMONICS is not specified, assume derivatives.
    if harmonics is None:
        if num % 2 == 0:
            harmonics = np.array(range(num//2))*2+1
        else:
            harmonics = np.array(range((15+1)//2))*2

    if len(harmonics.shape) == 1 or harmonics.shape[0] == 1:
        # reshape to column matrix
        harmonics = harmonics.reshape(harmonics.shape[0], 1)
    elif harmonics.shape[0] != 1 and harmonics.shape[1] != 1:
        raise Exception('input parameter HARMONICS must be 1D!')

    if 2*harmonics.shape[0] - (harmonics == 0).sum() != num:
        raise Exception('harmonics list is incompatible with basis size!')

    # If STEERMTX not passed, assume evenly distributed cosine-phase filters:
    if steermtx is None:
        steermtx = steer2HarmMtx(harmonics, np.pi*np.array(list(range(num)))/num, 'even')

    steervect = np.zeros((angle.shape[0], num))
    arg = angle * harmonics[np.nonzero(harmonics)[0]].T
    if all(harmonics):
        steervect[:, range(0, num, 2)] = np.cos(arg)
        steervect[:, range(1, num, 2)] = np.sin(arg)
    else:
        steervect[:, 1] = np.ones((arg.shape[0], 1))
        steervect[:, range(0, num, 2)] = np.cos(arg)
        steervect[:, range(1, num, 2)] = np.sin(arg)

    steervect = np.dot(steervect, steermtx)

    if steervect.shape[0] > 1:
        tmp = np.dot(basis, steervect)
        res = sum(tmp).T
    else:
        res = np.dot(basis, steervect.T)

    return res
