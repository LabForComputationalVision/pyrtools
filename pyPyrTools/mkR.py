import numpy


def mkR(size, expt=1, origin=None):
    '''make distance-from-origin (r) matrix

    Compute a matrix of dimension SIZE (a [Y X] list/tuple, or a scalar) containing samples of a
    radial ramp function, raised to power EXPT (default = 1), with given ORIGIN (default =
    (size+1)/2, (0, 0) = upper left).
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    # -----------------------------------------------------------------

    xramp, yramp = numpy.meshgrid(numpy.array(range(1, size[1]+1))-origin[1],
                                  numpy.array(range(1, size[0]+1))-origin[0])

    res = (xramp**2 + yramp**2)**(expt/2.0)

    return res
