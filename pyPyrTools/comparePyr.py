import numpy
import math

def comparePyr(matPyr, pyPyr):
    ''' compare two pyramids and return 1 if they are the same with in 
        desired precision and 0 if not.
        written for unit testing code. '''
    # FIX: make precision an input parameter
    prec = math.pow(10,-9)  # desired precision
    # compare two pyramids - return 0 for !=, 1 for == 
    # correct number of elements?
    matSz = sum(matPyr.shape)
    pySz = 1
    for idx in range(len(pyPyr.pyrSize)):
        sz = pyPyr.pyrSize[idx]
        if len(sz) == 1:
            pySz += sz[0]
        else:
            pySz += sz[0] * sz[1]

    if(matSz != pySz):
        print("size difference: %d != %d, returning 0" % (matSz, pySz))
        return 0

    # values are the same?
    matStart = 0
    for idx in range(len(pyPyr.pyrSize)):
        bandSz = pyPyr.pyrSize[idx]
        if len(bandSz) == 1:
            matLen = bandSz[0]
        else:
            matLen = bandSz[0] * bandSz[1]
        matTmp = matPyr[matStart:matStart + matLen]
        matTmp = numpy.reshape(matTmp, bandSz, order='F')
        matStart = matStart+matLen
        if (matTmp != pyPyr.pyr[idx]).any():
            print("some pyramid elements not identical: checking...")
            for i in range(bandSz[0]):
                for j in range(bandSz[1]):
                    if matTmp[i,j] != pyPyr.pyr[idx][i,j]:
                        if ( math.fabs(matTmp[i,j] - pyPyr.pyr[idx][i,j]) > 
                             prec ):
                            print("failed level:%d element:%d %d value:%.15f %.15f" % (idx, i, j, matTmp[i,j], pyPyr.pyr[idx][i,j]))
                            return 0
            print("same to at least %f" % prec)

    return 1
