import numpy as np

def comparePyr(matPyr, pyPyr, rtol=1e-5, atol=1e-8):
    ''' compare two pyramids and return 1 if they are the same with in
        desired precision and 0 if not.
        written for unit testing code. '''
    # compare two pyramids - return 0 for !=, 1 for ==
    # correct number of elements?
    matSz = sum(matPyr.shape)
    pySz = 1 + sum([np.array(size).prod() for size in pyPyr.pyrSize])

    if(matSz != pySz):
        print("size difference: %d != %d, returning 0" % (matSz, pySz))
        return 0

    # values are close to each other?
    matStart = 0
    for idx, pyTmp in enumerate(pyPyr.pyr):
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
                    if not isclose[i,j]:
                        print("failed level:%d element:%d %d value:%.15f %.15f" % (idx, i, j, matTmp[i,j], pyTmp[i,j]))
                        return 0

    return 1
