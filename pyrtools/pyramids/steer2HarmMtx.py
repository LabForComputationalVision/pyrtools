import numpy as np

def steer2HarmMtx(harmonics, angles=None, rel_phases=0):
    ''' Compute a steering matrix (maps a directional basis set onto the
        angular Fourier harmonics).

        HARMONICS is a vector specifying the angular harmonics contained in the
        steerable basis/filters.
        ANGLES (optional) is a vector specifying the angular position of each
        filter.
        REL_PHASES (optional, default = 0, ie. 'even') specifies whether the harmonics
        are cosine or sine phase aligned about those positions.

        The result matrix is suitable for passing to the function STEER.
        '''
    #
    # if len(args) == 0:
    #     print("Error: first parameter 'harmonics' is required.")
    #     return
    #
    # if len(args) > 0:
    #     harmonics = np.array(args[0])

    # default parameter
    numh = (2*harmonics.shape[0]) - (harmonics == 0).sum()
    if angles is None:
        angles = np.pi * np.array(range(numh)) / numh

    # if len(args) > 2:
    #     if isinstance(args[2], str):
    #         if args[2] == 'even' or args[2] == 'EVEN':
    #             evenorodd = 0
    #         elif args[2] == 'odd' or args[2] == 'ODD':
    #             evenorodd = 1
    #         else:
    #             print("Error: only 'even' and 'odd' are valid entries for the third input parameter.")
    #             return
    #     else:
    #         print("Error: third input parameter must be a string (even/odd).")
    # else:
    #     evenorodd = 0

    # Compute inverse matrix, which maps to Fourier components onto
    # steerable basis
    imtx = np.zeros((angles.shape[0], numh))
    col = 0
    for h in harmonics:
        args = h * angles
        if h == 0:
            imtx[:, col] = np.ones(angles.shape)
            col += 1
        elif rel_phases:
            imtx[:, col] = np.sin(args)
            imtx[:, col+1] = np.negative( np.cos(args) )
            col += 2
        else:
            imtx[:, col] = np.cos(args)
            imtx[:, col+1] = np.sin(args)
            col += 2

    r = np.linalg.matrix_rank(imtx)
    if r != numh and r != angles.shape[0]:
        print("Warning: matrix is not full rank")

    mtx = np.linalg.pinv(imtx)

    return mtx
