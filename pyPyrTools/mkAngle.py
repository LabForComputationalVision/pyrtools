import numpy


def mkAngle(size, phase=0, origin=None):
    '''make polar angle matrix (in radians)

    Compute a matrix of dimension SIZE (a [Y X] list/tuple, or a scalar) containing samples of the
    polar angle (in radians, CW from the X-axis, ranging from -pi to pi), relative to angle PHASE
    (default = 0), about ORIGIN pixel (default = (size+1)/2).
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    xramp, yramp = numpy.meshgrid(numpy.array(range(1, size[1]+1))-origin[1],
                                  numpy.array(range(1, size[0]+1))-origin[0])
    xramp = numpy.array(xramp)
    yramp = numpy.array(yramp)

    res = numpy.arctan2(yramp, xramp)

    res = ((res+(numpy.pi-phase)) % (2*numpy.pi)) - numpy.pi

    return res
