import numpy

def shift(mtx, offset):
    ''' Circular shift 2D matrix samples by OFFSET (a [Y,X] 2-tuple),
        such that  RES(POS) = MTX(POS-OFFSET).  '''

    dims = mtx.shape
    if len(dims) == 1:
        mtx = mtx.reshape((1, dims[0]))
        dims = mtx.shape

    offset = numpy.mod(numpy.negative(offset), dims)

    top = numpy.column_stack((mtx[offset[0]:dims[0], offset[1]:dims[1]],
                           mtx[offset[0]:dims[0], 0:offset[1]]))
    bottom = numpy.column_stack((mtx[0:offset[0], offset[1]:dims[1]],
                              mtx[0:offset[0], 0:offset[1]]))

    ret = numpy.concatenate((top, bottom), axis=0)

    return ret
