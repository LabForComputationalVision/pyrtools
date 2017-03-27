import numpy
from steer2HarmMtx import steer2HarmMtx

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

    if isinstance(angle, (int, long, float)):
        angle = numpy.array([angle])
    else:
        if angle.shape[0] != basis.shape[0] or angle.shape[1] != 1:
            raise Exception('ANGLE must be a scalar, or a column vector the size of the basis elements')

    # If HARMONICS is not specified, assume derivatives.
    if harmonics is None:
        if num % 2 == 0:
            harmonics = numpy.array(range(num/2))*2+1
        else:
            harmonics = numpy.array(range((15+1)/2))*2

    if len(harmonics.shape) == 1 or harmonics.shape[0] == 1:
        # reshape to column matrix
        harmonics = harmonics.reshape(harmonics.shape[0], 1)
    elif harmonics.shape[0] != 1 and harmonics.shape[1] != 1:
        raise Exception('input parameter HARMONICS must be 1D!')

    if 2*harmonics.shape[0] - (harmonics == 0).sum() != num:
        raise Exception('harmonics list is incompatible with basis size!')

    # If STEERMTX not passed, assume evenly distributed cosine-phase filters:
    if steermtx is None:
        steermtx = steer2HarmMtx(harmonics, numpy.pi*numpy.array(range(num))/num, 'even')

    steervect = numpy.zeros((angle.shape[0], num))
    arg = angle * harmonics[numpy.nonzero(harmonics)[0]].T
    if all(harmonics):
        steervect[:, range(0, num, 2)] = numpy.cos(arg)
        steervect[:, range(1, num, 2)] = numpy.sin(arg)
    else:
        steervect[:, 1] = numpy.ones((arg.shape[0], 1))
        steervect[:, range(0, num, 2)] = numpy.cos(arg)
        steervect[:, range(1, num, 2)] = numpy.sin(arg)

    steervect = numpy.dot(steervect, steermtx)

    if steervect.shape[0] > 1:
        tmp = numpy.dot(basis, steervect)
        res = sum(tmp).T
    else:
        res = numpy.dot(basis, steervect.T)

    return res
