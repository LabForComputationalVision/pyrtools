import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pylab
import scipy.signal as spsig
import scipy.stats as sps
import math
import struct
import re

def showIm(*args):
    if len(args) == 0:
        print "showIm( matrix, range, zoom, label, colormap, colorbar )"
        print "  matrix is string. It should be the name of a 2D array."
        print "  range is a two element tuple.  It specifies the values that "
        print "    map to the min and max colormap values.  Passing a value "
        print "    of 'auto' (default) sets range=[min,max].  'auto2' sets "
        print "    range=[mean-2*stdev, mean+2*stdev].  'auto3' sets "
        print "    range=[p1-(p2-p1)/8, p2+(p2-p1)/8], where p1 is the 10th "
        print "    percientile value of the sorted matix samples, and p2 is "
        print "    the 90th percentile value."
        print "  zoom specifies the number of matrix smples per screen pixel. "
        print "    It will be rounded to an integer, or 1 divided by an "
        print "    integer.  A value of 'same' or 'auto' (default) causes the "
        print "    zoom value to be chosen automatically to fit the image into "
        print "    the current axes.  A value of 'full' fills the axis region "
        print "    (leaving no room for labels)."
        print "  If label (optional, default = 1, unless zoom = 'full') is "
        print "    non-zero, the range of values that are mapped into the "
        print "    colormap and the dimensions (size) of the matrix and zoom "
        print "    factor are printed below the image.  If label is a string, "
        print "    it is used as a title."
        print "  colormap must contain the string 'auto' (grey colormap with " 
        print "    size = matrix.max() - matrix.min() will be used), "
        print "    or a string that is the name of a colormap variable "
        print "  colorbar is a boolean that specifies whether or not a "
        print "    colorbar is displayed"
    if len(args) > 0:   # matrix entered
        matrix = args[0]
        # defaults for all other values in case they weren't entered
        imRange = ( np.amin(matrix), np.amax(matrix) )
        zoom = 1
        label = 1
        colorbar = False
        colormap = cm.Greys_r
    if len(args) > 1:   # range entered
        if isinstance(args[1], basestring):
            if args[1] is "auto":
                imRange = ( np.amin(matrix), np.amax(matrix) )
            elif args[1] is "auto2":
                imRange = ( matrix.mean()-2*matrix.std(), 
                            matrix.mean()+2*matrix.std() )
            elif args[1] is "auto3":
                #p1 = np.percentile(matrix, 10)  not in python 2.6.6?!
                #p2 = np.percentile(matrix, 90)
                p1 = sps.scoreatpercentile(np.hstack(matrix), 10)
                p2 = sps.scoreatpercentile(np.hstack(matrix), 90)
                imRange = p1-(p2-p1)/8, p2+(p2-p1)/8
            else:
                print "Error: range of %s is not recognized." % args[1]
                print "       please use a two element tuple or "
                print "       'auto', 'auto2' or 'auto3'"
                print "       enter 'showIm' for more info about options"
                return
        else:
            imRange = args[1][0], args[1][1]
    if len(args) > 2:   # zoom entered
        # no equivalent to matlab's pixelAxes in matplotlib. need dpi
        # might work with tkinter, but then have to change everything
        zoom = 1
    if len(args) > 3:   # label entered
        label = args[3]
    if len(args) > 4:   # colormap entered
        if args[4] is "auto":
            colormap = cm.Greys_r
        else:  # got a variable name
            colormap = args[4]
    if len(args) > 5 and args[5]:   # colorbar entered and set to true
        colorbar = args[5]
        
    #imgplot = plt.imshow(matrix, colormap, origin='lower').set_clim(imRange)
    imgplot = plt.imshow(matrix, colormap).set_clim(imRange)
    #plt.gca().invert_yaxis()  # default is inverted y from matlab
    if label != 0 and label != 1:
        plt.title(label)
    if colorbar:
        plt.colorbar(imgplot, cmap=cmap)
    #pylab.show()
    plt.show()
    
# Compute maximum pyramid height for given image and filter sizes.
# Specifically: the number of corrDn operations that can be sequentially
# performed when subsampling by a factor of 2.
def maxPyrHt(imsz, filtsz):
    if isinstance(imsz, int):
        imsz = (imsz, 1)
    if isinstance(filtsz, int):
        filtsz = (filtsz, 1)

    if len(imsz) == 1 and len(filtsz) == 1:
        imsz = (imsz[0], 1)
        filtsz = (filtsz[0], 1)
    elif len(imsz) == 1 and not any(f == 1 for f in filtsz):
            print "Error: cannot have a 1D 'image' and 2D filter"
            exit(1)
    elif len(imsz) == 1:
        imsz = (imsz[0], 1)
    elif len(filtsz) == 1:
        filtsz = (filtsz[0], 1)

    if imsz == 0:
        height = 0
    elif isinstance(imsz, tuple):
        if any( i < f for i,f in zip(imsz, filtsz) ):
            height = 0
        else:
            #if any( i == 1 for i in imsz):
            if imsz[0] == 1:
                imsz = (1, int(math.floor(imsz[1]/2) ) )
            elif imsz[1] == 1:
                imsz = (int( math.floor(imsz[0]/2) ), 1)
            else:
                imsz = ( int( math.floor(imsz[0]/2) ), 
                         int( math.floor(imsz[1]/2) ))
            height = 1 + maxPyrHt(imsz, filtsz)
    else:
        if any(imsz < f for f in filtsz):
            height = 0;
        else:
            imsz = ( int( math.floor(imsz/2) ), 1 )
            height = 1 + maxPyrHt(imsz, filtsz)
            
    return height

# returns a vector of binomial coefficients of order (size-1)
def binomialFilter(size):
    if size < 2:
        print "Error: size argument must be larger than 1"
        exit(1)
    
    kernel = np.array([[0.5], [0.5]])

    for i in range(0, size-2):
        kernel = spsig.convolve(np.array([[0.5], [0.5]]), kernel)

    return np.asarray(kernel)

# Some standard 1D filter kernels. These are scaled such that their L2-norm 
#   is 1.0
#
# binomN              - binomial coefficient filter of order N-1
# haar                - Harr wavelet
# qmf8, qmf12, qmf16  - Symmetric Quadrature Mirror Filters [Johnston80]
# daub2, daub3, daub4 - Daubechies wavelet [Daubechies88]
# qmf5, qmf9, qmf13   - Symmetric Quadrature Mirror Filters [Simoncelli88, 
#                                                            Simoncelli90]
# [Johnston80] - J D Johnston, "A filter family designed for use in quadrature 
#    mirror filter banks", Proc. ICASSP, pp 291-294, 1980.
#
# [Daubechies88] - I Daubechies, "Orthonormal bases of compactly supported wavelets",
#    Commun. Pure Appl. Math, vol. 42, pp 909-996, 1988.
#
# [Simoncelli88] - E P Simoncelli,  "Orthogonal sub-band image transforms",
#     PhD Thesis, MIT Dept. of Elec. Eng. and Comp. Sci. May 1988.
#     Also available as: MIT Media Laboratory Vision and Modeling Technical 
#     Report #100.
#
# [Simoncelli90] -  E P Simoncelli and E H Adelson, "Subband image coding",
#    Subband Transforms, chapter 4, ed. John W Woods, Kluwer Academic 
#    Publishers,  Norwell, MA, 1990, pp 143--192.
#
# Rob Young, 4/13
#
def namedFilter(name):
    if len(name) > 5 and name[:5] == "binom":
        kernel = math.sqrt(2) * binomialFilter(int(name[5:]))
    elif name is "qmf5":
        kernel = np.array([[-0.076103], [0.3535534], [0.8593118], [0.3535534], [-0.076103]])
    elif name is "qmf9":
        kernel = np.array([[0.02807382], [-0.060944743], [-0.073386624], [0.41472545], [0.7973934], [0.41472545], [-0.073386624], [-0.060944743], [0.02807382]])
    elif name is "qmf13":
        kernel = np.array([[-0.014556438], [0.021651438], [0.039045125], [-0.09800052], [-0.057827797], [0.42995453], [0.7737113], [0.42995453], [-0.057827797], [-0.09800052], [0.039045125], [0.021651438], [-0.014556438]])
    elif name is "qmf8":
        kernel = math.sqrt(2) * np.array([[0.00938715], [-0.07065183], [0.06942827], [0.4899808], [0.4899808], [0.06942827], [-0.07065183], [0.00938715]])
    elif name is "qmf12":
        kernel = math.sqrt(2) * np.array([[-0.003809699], [0.01885659], [-0.002710326], [-0.08469594], [0.08846992], [0.4843894], [0.4843894], [0.08846992], [-0.08469594], [-0.002710326], [0.01885659], [-0.003809699]])
    elif name is "qmf16":
        kernel = math.sqrt(2) * np.array([[0.001050167], [-0.005054526], [-0.002589756], [0.0276414], [-0.009666376], [-0.09039223], [0.09779817], [0.4810284], [0.4810284], [0.09779817], [-0.09039223], [-0.009666376], [0.0276414], [-0.002589756], [-0.005054526], [0.001050167]])
    elif name is "haar":
        kernel = np.array([[1], [1]]) / math.sqrt(2)
    elif name is "daub2":
        kernel = np.array([[0.482962913145], [0.836516303738], [0.224143868042], [-0.129409522551]]);
    elif name is "daub3":
        kernel = np.array([[0.332670552950], [0.806891509311], [0.459877502118], [-0.135011020010], [-0.085441273882], [0.035226291882]])
    elif name is "daub4":
        kernel = np.array([[0.230377813309], [0.714846570553], [0.630880767930], [-0.027983769417], [-0.187034811719], [0.030841381836], [0.032883011667], [-0.010597401785]])
    elif name is "gauss5":  # for backward-compatibility
        kernel = math.sqrt(2) * np.array([[0.0625], [0.25], [0.375], [0.25], [0.0625]])
    elif name is "gauss3":  # for backward-compatibility
        kernel = math.sqrt(2) * np.array([[0.25], [0.5], [0.25]])
    else:
        print "Error: Bad filter name: %s" % (name)
        exit(1)
    return np.array(kernel)

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))

def comparePyr(matPyr, pyPyr):
    # compare two pyramids - return 0 for !=, 1 for == 
    # correct number of elements?
    matSz = sum(matPyr.shape)
    pySz = 1
    for key in pyPyr.pyrSize.keys():
        sz = pyPyr.pyrSize[key]
        if len(sz) == 1:
            pySz += sz[0]
        else:
            pySz += sz[0] * sz[1]

    if(matSz != pySz):
        print "size difference: returning 0"
        print matSz
        print pySz
        print pyPyr.pyr.keys()
        return 0

    # values are the same?
    matStart = 0
    for key, value in pyPyr.pyrSize.iteritems():
        bandSz = value
        matLen = bandSz[0] * bandSz[1]
        matTmp = matPyr[matStart:matStart + matLen]
        matTmp = np.reshape(matTmp, bandSz, order='F')
        matStart = matStart+matLen
        if (matTmp != pyPyr.pyr[key]).any():
            print "some pyramid elements not identical: checking..."
            for i in range(value[0]):
                for j in range(value[1]):
                    if matTmp[i,j] != pyPyr.pyr[key][i,j]:
                        #print "%d %d : %.20f" % (i,j,
                        #                         math.fabs(matTmp[i,j]- 
                        #                                  pyPyr.pyr[key][i,j]))
                        if ( math.fabs(matTmp[i,j] - pyPyr.pyr[key][i,j]) > 
                             math.pow(10,-11) ):
                            return 0
            print "same to at least 10^-11"

    return 1

def mkRamp(*args):
    # mkRamp(SIZE, DIRECTION, SLOPE, INTERCEPT, ORIGIN)
    # Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
    # containing samples of a ramp function, with given gradient DIRECTION
    # (radians, CW from X-axis, default = 0), SLOPE (per pixel, default = 
    # 1), and a value of INTERCEPT (default = 0) at the ORIGIN (default =
    # (size+1)/2, [1 1] = upper left). All but the first argument are
    # optional
    
    if len(args) == 0:
        print "mkRamp(SIZE, DIRECTION, SLOPE, INTERCEPT, ORIGIN)"
        print "first argument is required"
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print "first argument must be a two element tuple or an integer"
            exit(1)

    # OPTIONAL args:

    if len(args) > 1:
        direction = args[1]
    else:
        direction = 0

    if len(args) > 2:
        slope = args[2]
    else:
        slope = 1

    if len(args) > 3:
        intercept = args[3]
    else:
        intercept = 0

    if len(args) > 4:
        origin = args[4]
    else:
        origin = ( float(sz[0]-1)/2.0, float(sz[1]-1)/2.0 )

    #--------------------------

    xinc = slope * math.cos(direction)
    yinc = slope * math.sin(direction)

    [xramp, yramp] = np.meshgrid( xinc * (np.array(range(sz[1]))-origin[1]),
                                  yinc * (np.array(range(sz[0]))-origin[0]) )

    res = intercept + xramp + yramp

    return res.copy()

# Steerable pyramid filters.  Transform described  in:
#
# @INPROCEEDINGS{Simoncelli95b,
#	TITLE = "The Steerable Pyramid: A Flexible Architecture for
#		 Multi-Scale Derivative Computation",
#	AUTHOR = "E P Simoncelli and W T Freeman",
#	BOOKTITLE = "Second Int'l Conf on Image Processing",
#	ADDRESS = "Washington, DC", MONTH = "October", YEAR = 1995 }
#
# Filter kernel design described in:
#
#@INPROCEEDINGS{Karasaridis96,
#	TITLE = "A Filter Design Technique for 
#		Steerable Pyramid Image Transforms",
#	AUTHOR = "A Karasaridis and E P Simoncelli",
#	BOOKTITLE = "ICASSP",	ADDRESS = "Atlanta, GA",
#	MONTH = "May",	YEAR = 1996 }
def sp0Filters():
    filters = {}
    filters['harmonics'] = np.array([0])
    filters['lo0filt'] =  ( 
        np.array([[-4.514000e-04, -1.137100e-04, -3.725800e-04, -3.743860e-03, 
                   -3.725800e-04, -1.137100e-04, -4.514000e-04], 
                  [-1.137100e-04, -6.119520e-03, -1.344160e-02, -7.563200e-03, 
                    -1.344160e-02, -6.119520e-03, -1.137100e-04],
                  [-3.725800e-04, -1.344160e-02, 6.441488e-02, 1.524935e-01, 
                    6.441488e-02, -1.344160e-02, -3.725800e-04], 
                  [-3.743860e-03, -7.563200e-03, 1.524935e-01, 3.153017e-01, 
                    1.524935e-01, -7.563200e-03, -3.743860e-03], 
                  [-3.725800e-04, -1.344160e-02, 6.441488e-02, 1.524935e-01, 
                    6.441488e-02, -1.344160e-02, -3.725800e-04],
                  [-1.137100e-04, -6.119520e-03, -1.344160e-02, -7.563200e-03, 
                    -1.344160e-02, -6.119520e-03, -1.137100e-04], 
                  [-4.514000e-04, -1.137100e-04, -3.725800e-04, -3.743860e-03,
                    -3.725800e-04, -1.137100e-04, -4.514000e-04]]) )
    filters['lofilt'] = (
        np.array([[-2.257000e-04, -8.064400e-04, -5.686000e-05, 8.741400e-04, 
                   -1.862800e-04, -1.031640e-03, -1.871920e-03, -1.031640e-03,
                   -1.862800e-04, 8.741400e-04, -5.686000e-05, -8.064400e-04,
                   -2.257000e-04],
                  [-8.064400e-04, 1.417620e-03, -1.903800e-04, -2.449060e-03, 
                    -4.596420e-03, -7.006740e-03, -6.948900e-03, -7.006740e-03,
                    -4.596420e-03, -2.449060e-03, -1.903800e-04, 1.417620e-03,
                    -8.064400e-04],
                  [-5.686000e-05, -1.903800e-04, -3.059760e-03, -6.401000e-03,
                    -6.720800e-03, -5.236180e-03, -3.781600e-03, -5.236180e-03,
                    -6.720800e-03, -6.401000e-03, -3.059760e-03, -1.903800e-04,
                    -5.686000e-05],
                  [8.741400e-04, -2.449060e-03, -6.401000e-03, -5.260020e-03, 
                   3.938620e-03, 1.722078e-02, 2.449600e-02, 1.722078e-02, 
                   3.938620e-03, -5.260020e-03, -6.401000e-03, -2.449060e-03, 
                   8.741400e-04], 
                  [-1.862800e-04, -4.596420e-03, -6.720800e-03, 3.938620e-03,
                    3.220744e-02, 6.306262e-02, 7.624674e-02, 6.306262e-02,
                    3.220744e-02, 3.938620e-03, -6.720800e-03, -4.596420e-03,
                    -1.862800e-04],
                  [-1.031640e-03, -7.006740e-03, -5.236180e-03, 1.722078e-02, 
                    6.306262e-02, 1.116388e-01, 1.348999e-01, 1.116388e-01, 
                    6.306262e-02, 1.722078e-02, -5.236180e-03, -7.006740e-03,
                    -1.031640e-03],
                  [-1.871920e-03, -6.948900e-03, -3.781600e-03, 2.449600e-02,
                    7.624674e-02, 1.348999e-01, 1.576508e-01, 1.348999e-01,
                    7.624674e-02, 2.449600e-02, -3.781600e-03, -6.948900e-03,
                    -1.871920e-03],
                  [-1.031640e-03, -7.006740e-03, -5.236180e-03, 1.722078e-02,
                    6.306262e-02, 1.116388e-01, 1.348999e-01, 1.116388e-01,
                    6.306262e-02, 1.722078e-02, -5.236180e-03, -7.006740e-03,
                    -1.031640e-03], 
                  [-1.862800e-04, -4.596420e-03, -6.720800e-03, 3.938620e-03,
                    3.220744e-02, 6.306262e-02, 7.624674e-02, 6.306262e-02,
                    3.220744e-02, 3.938620e-03, -6.720800e-03, -4.596420e-03,
                    -1.862800e-04],
                  [8.741400e-04, -2.449060e-03, -6.401000e-03, -5.260020e-03,
                   3.938620e-03, 1.722078e-02, 2.449600e-02, 1.722078e-02, 
                   3.938620e-03, -5.260020e-03, -6.401000e-03, -2.449060e-03,
                   8.741400e-04],
                  [-5.686000e-05, -1.903800e-04, -3.059760e-03, -6.401000e-03,
                    -6.720800e-03, -5.236180e-03, -3.781600e-03, -5.236180e-03,
                    -6.720800e-03, -6.401000e-03, -3.059760e-03, -1.903800e-04,
                    -5.686000e-05],
                  [-8.064400e-04, 1.417620e-03, -1.903800e-04, -2.449060e-03,
                    -4.596420e-03, -7.006740e-03, -6.948900e-03, -7.006740e-03,
                    -4.596420e-03, -2.449060e-03, -1.903800e-04, 1.417620e-03,
                    -8.064400e-04], 
                  [-2.257000e-04, -8.064400e-04, -5.686000e-05, 8.741400e-04,
                   -1.862800e-04, -1.031640e-03, -1.871920e-03, -1.031640e-03,
                    -1.862800e-04, 8.741400e-04, -5.686000e-05, -8.064400e-04,
                    -2.257000e-04]]) )
    filters['mtx'] = np.array([ 1.000000 ])
    filters['hi0filt'] = ( 
        np.array([[5.997200e-04, -6.068000e-05, -3.324900e-04, -3.325600e-04, 
                   -2.406600e-04, -3.325600e-04, -3.324900e-04, -6.068000e-05, 
                   5.997200e-04],
                  [-6.068000e-05, 1.263100e-04, 4.927100e-04, 1.459700e-04, 
                    -3.732100e-04, 1.459700e-04, 4.927100e-04, 1.263100e-04, 
                    -6.068000e-05],
                  [-3.324900e-04, 4.927100e-04, -1.616650e-03, -1.437358e-02, 
                    -2.420138e-02, -1.437358e-02, -1.616650e-03, 4.927100e-04, 
                    -3.324900e-04], 
                  [-3.325600e-04, 1.459700e-04, -1.437358e-02, -6.300923e-02, 
                    -9.623594e-02, -6.300923e-02, -1.437358e-02, 1.459700e-04, 
                    -3.325600e-04],
                  [-2.406600e-04, -3.732100e-04, -2.420138e-02, -9.623594e-02, 
                    8.554893e-01, -9.623594e-02, -2.420138e-02, -3.732100e-04, 
                    -2.406600e-04],
                  [-3.325600e-04, 1.459700e-04, -1.437358e-02, -6.300923e-02, 
                    -9.623594e-02, -6.300923e-02, -1.437358e-02, 1.459700e-04, 
                    -3.325600e-04], 
                  [-3.324900e-04, 4.927100e-04, -1.616650e-03, -1.437358e-02, 
                    -2.420138e-02, -1.437358e-02, -1.616650e-03, 4.927100e-04, 
                    -3.324900e-04], 
                  [-6.068000e-05, 1.263100e-04, 4.927100e-04, 1.459700e-04, 
                    -3.732100e-04, 1.459700e-04, 4.927100e-04, 1.263100e-04, 
                    -6.068000e-05], 
                  [5.997200e-04, -6.068000e-05, -3.324900e-04, -3.325600e-04, 
                   -2.406600e-04, -3.325600e-04, -3.324900e-04, -6.068000e-05, 
                   5.997200e-04]]) )
    filters['bfilts'] = ( 
        np.array([-9.066000e-05, -1.738640e-03, -4.942500e-03, -7.889390e-03, 
                   -1.009473e-02, -7.889390e-03, -4.942500e-03, -1.738640e-03, 
                   -9.066000e-05, -1.738640e-03, -4.625150e-03, -7.272540e-03, 
                   -7.623410e-03, -9.091950e-03, -7.623410e-03, -7.272540e-03, 
                   -4.625150e-03, -1.738640e-03, -4.942500e-03, -7.272540e-03, 
                   -2.129540e-02, -2.435662e-02, -3.487008e-02, -2.435662e-02, 
                   -2.129540e-02, -7.272540e-03, -4.942500e-03, -7.889390e-03, 
                   -7.623410e-03, -2.435662e-02, -1.730466e-02, -3.158605e-02, 
                   -1.730466e-02, -2.435662e-02, -7.623410e-03, -7.889390e-03,
                   -1.009473e-02, -9.091950e-03, -3.487008e-02, -3.158605e-02, 
                   9.464195e-01, -3.158605e-02, -3.487008e-02, -9.091950e-03, 
                   -1.009473e-02, -7.889390e-03, -7.623410e-03, -2.435662e-02, 
                   -1.730466e-02, -3.158605e-02, -1.730466e-02, -2.435662e-02, 
                   -7.623410e-03, -7.889390e-03, -4.942500e-03, -7.272540e-03, 
                   -2.129540e-02, -2.435662e-02, -3.487008e-02, -2.435662e-02, 
                   -2.129540e-02, -7.272540e-03, -4.942500e-03, -1.738640e-03, 
                   -4.625150e-03, -7.272540e-03, -7.623410e-03, -9.091950e-03, 
                   -7.623410e-03, -7.272540e-03, -4.625150e-03, -1.738640e-03,
                   -9.066000e-05, -1.738640e-03, -4.942500e-03, -7.889390e-03,
                   -1.009473e-02, -7.889390e-03, -4.942500e-03, -1.738640e-03,
                   -9.066000e-05]) )
    filters['bfilts'] = filters['bfilts'].reshape(len(filters['bfilts']),1)
    return filters

def sp1Filters():
    filters = {}
    filters['harmonics'] = np.array([ 1 ])
    filters['mtx'] = np.eye(2)
    filters['lo0filt'] = ( 
        np.array([[-8.701000e-05, -1.354280e-03, -1.601260e-03, -5.033700e-04, 
                    2.524010e-03, -5.033700e-04, -1.601260e-03, -1.354280e-03, 
                    -8.701000e-05],
                  [-1.354280e-03, 2.921580e-03, 7.522720e-03, 8.224420e-03, 
                    1.107620e-03, 8.224420e-03, 7.522720e-03, 2.921580e-03, 
                    -1.354280e-03],
                  [-1.601260e-03, 7.522720e-03, -7.061290e-03, -3.769487e-02,
                    -3.297137e-02, -3.769487e-02, -7.061290e-03, 7.522720e-03,
                    -1.601260e-03],
                  [-5.033700e-04, 8.224420e-03, -3.769487e-02, 4.381320e-02,
                    1.811603e-01, 4.381320e-02, -3.769487e-02, 8.224420e-03,
                    -5.033700e-04], 
                  [2.524010e-03, 1.107620e-03, -3.297137e-02, 1.811603e-01, 
                   4.376250e-01, 1.811603e-01, -3.297137e-02, 1.107620e-03, 
                   2.524010e-03],
                  [-5.033700e-04, 8.224420e-03, -3.769487e-02, 4.381320e-02, 
                    1.811603e-01, 4.381320e-02, -3.769487e-02, 8.224420e-03,
                    -5.033700e-04],
                  [-1.601260e-03, 7.522720e-03, -7.061290e-03, -3.769487e-02,
                    -3.297137e-02, -3.769487e-02, -7.061290e-03, 7.522720e-03,
                    -1.601260e-03],
                  [-1.354280e-03, 2.921580e-03, 7.522720e-03, 8.224420e-03, 
                    1.107620e-03, 8.224420e-03, 7.522720e-03, 2.921580e-03,
                    -1.354280e-03], 
                  [-8.701000e-05, -1.354280e-03, -1.601260e-03, -5.033700e-04, 
                    2.524010e-03, -5.033700e-04, -1.601260e-03, -1.354280e-03, 
                    -8.701000e-05]]) )
    filters['lofilt'] = (
        np.array([[-4.350000e-05, 1.207800e-04, -6.771400e-04, -1.243400e-04, 
                    -8.006400e-04, -1.597040e-03, -2.516800e-04, -4.202000e-04,
                    1.262000e-03, -4.202000e-04, -2.516800e-04, -1.597040e-03,
                    -8.006400e-04, -1.243400e-04, -6.771400e-04, 1.207800e-04,
                    -4.350000e-05], 
                  [1.207800e-04, 4.460600e-04, -5.814600e-04, 5.621600e-04, 
                   -1.368800e-04, 2.325540e-03, 2.889860e-03, 4.287280e-03, 
                   5.589400e-03, 4.287280e-03, 2.889860e-03, 2.325540e-03, 
                   -1.368800e-04, 5.621600e-04, -5.814600e-04, 4.460600e-04, 
                   1.207800e-04],
                  [-6.771400e-04, -5.814600e-04, 1.460780e-03, 2.160540e-03, 
                    3.761360e-03, 3.080980e-03, 4.112200e-03, 2.221220e-03, 
                    5.538200e-04, 2.221220e-03, 4.112200e-03, 3.080980e-03, 
                    3.761360e-03, 2.160540e-03, 1.460780e-03, -5.814600e-04, 
                    -6.771400e-04],
                  [-1.243400e-04, 5.621600e-04, 2.160540e-03, 3.175780e-03, 
                    3.184680e-03, -1.777480e-03, -7.431700e-03, -9.056920e-03,
                    -9.637220e-03, -9.056920e-03, -7.431700e-03, -1.777480e-03,
                    3.184680e-03, 3.175780e-03, 2.160540e-03, 5.621600e-04, 
                    -1.243400e-04],
                  [-8.006400e-04, -1.368800e-04, 3.761360e-03, 3.184680e-03, 
                    -3.530640e-03, -1.260420e-02, -1.884744e-02, -1.750818e-02,
                    -1.648568e-02, -1.750818e-02, -1.884744e-02, -1.260420e-02,
                    -3.530640e-03, 3.184680e-03, 3.761360e-03, -1.368800e-04,
                    -8.006400e-04],
                  [-1.597040e-03, 2.325540e-03, 3.080980e-03, -1.777480e-03, 
                    -1.260420e-02, -2.022938e-02, -1.109170e-02, 3.955660e-03, 
                    1.438512e-02, 3.955660e-03, -1.109170e-02, -2.022938e-02, 
                    -1.260420e-02, -1.777480e-03, 3.080980e-03, 2.325540e-03, 
                    -1.597040e-03],
                  [-2.516800e-04, 2.889860e-03, 4.112200e-03, -7.431700e-03, 
                    -1.884744e-02, -1.109170e-02, 2.190660e-02, 6.806584e-02, 
                    9.058014e-02, 6.806584e-02, 2.190660e-02, -1.109170e-02, 
                    -1.884744e-02, -7.431700e-03, 4.112200e-03, 2.889860e-03, 
                    -2.516800e-04],
                  [-4.202000e-04, 4.287280e-03, 2.221220e-03, -9.056920e-03, 
                    -1.750818e-02, 3.955660e-03, 6.806584e-02, 1.445500e-01, 
                    1.773651e-01, 1.445500e-01, 6.806584e-02, 3.955660e-03, 
                    -1.750818e-02, -9.056920e-03, 2.221220e-03, 4.287280e-03, 
                    -4.202000e-04],
                  [1.262000e-03, 5.589400e-03, 5.538200e-04, -9.637220e-03, 
                   -1.648568e-02, 1.438512e-02, 9.058014e-02, 1.773651e-01, 
                   2.120374e-01, 1.773651e-01, 9.058014e-02, 1.438512e-02, 
                   -1.648568e-02, -9.637220e-03, 5.538200e-04, 5.589400e-03, 
                   1.262000e-03],
                  [-4.202000e-04, 4.287280e-03, 2.221220e-03, -9.056920e-03, 
                    -1.750818e-02, 3.955660e-03, 6.806584e-02, 1.445500e-01, 
                    1.773651e-01, 1.445500e-01, 6.806584e-02, 3.955660e-03, 
                    -1.750818e-02, -9.056920e-03, 2.221220e-03, 4.287280e-03, 
                    -4.202000e-04],
                  [-2.516800e-04, 2.889860e-03, 4.112200e-03, -7.431700e-03, 
                    -1.884744e-02, -1.109170e-02, 2.190660e-02, 6.806584e-02, 
                    9.058014e-02, 6.806584e-02, 2.190660e-02, -1.109170e-02, 
                    -1.884744e-02, -7.431700e-03, 4.112200e-03, 2.889860e-03, 
                    -2.516800e-04],
                  [-1.597040e-03, 2.325540e-03, 3.080980e-03, -1.777480e-03, 
                    -1.260420e-02, -2.022938e-02, -1.109170e-02, 3.955660e-03, 
                    1.438512e-02, 3.955660e-03, -1.109170e-02, -2.022938e-02, 
                    -1.260420e-02, -1.777480e-03, 3.080980e-03, 2.325540e-03, 
                    -1.597040e-03],
                  [-8.006400e-04, -1.368800e-04, 3.761360e-03, 3.184680e-03, 
                    -3.530640e-03, -1.260420e-02, -1.884744e-02, -1.750818e-02,
                    -1.648568e-02, -1.750818e-02, -1.884744e-02, -1.260420e-02,
                    -3.530640e-03, 3.184680e-03, 3.761360e-03, -1.368800e-04,
                    -8.006400e-04],
                  [-1.243400e-04, 5.621600e-04, 2.160540e-03, 3.175780e-03, 
                    3.184680e-03, -1.777480e-03, -7.431700e-03, -9.056920e-03,
                    -9.637220e-03, -9.056920e-03, -7.431700e-03, -1.777480e-03,
                    3.184680e-03, 3.175780e-03, 2.160540e-03, 5.621600e-04,
                    -1.243400e-04],
                  [-6.771400e-04, -5.814600e-04, 1.460780e-03, 2.160540e-03, 
                    3.761360e-03, 3.080980e-03, 4.112200e-03, 2.221220e-03, 
                    5.538200e-04, 2.221220e-03, 4.112200e-03, 3.080980e-03, 
                    3.761360e-03, 2.160540e-03, 1.460780e-03, -5.814600e-04, 
                    -6.771400e-04],
                  [1.207800e-04, 4.460600e-04, -5.814600e-04, 5.621600e-04, 
                   -1.368800e-04, 2.325540e-03, 2.889860e-03, 4.287280e-03, 
                   5.589400e-03, 4.287280e-03, 2.889860e-03, 2.325540e-03, 
                   -1.368800e-04, 5.621600e-04, -5.814600e-04, 4.460600e-04, 
                   1.207800e-04],
                  [-4.350000e-05, 1.207800e-04, -6.771400e-04, -1.243400e-04, 
                    -8.006400e-04, -1.597040e-03, -2.516800e-04, -4.202000e-04,
                    1.262000e-03, -4.202000e-04, -2.516800e-04, -1.597040e-03,
                    -8.006400e-04, -1.243400e-04, -6.771400e-04, 1.207800e-04,
                    -4.350000e-05] ]) )
    filters['hi0filt'] = (
        np.array([[-9.570000e-04, -2.424100e-04, -1.424720e-03, -8.742600e-04, 
                    -1.166810e-03, -8.742600e-04, -1.424720e-03, -2.424100e-04,
                    -9.570000e-04],
                  [-2.424100e-04, -4.317530e-03, 8.998600e-04, 9.156420e-03, 
                    1.098012e-02, 9.156420e-03, 8.998600e-04, -4.317530e-03, 
                    -2.424100e-04],
                  [-1.424720e-03, 8.998600e-04, 1.706347e-02, 1.094866e-02, 
                    -5.897780e-03, 1.094866e-02, 1.706347e-02, 8.998600e-04, 
                    -1.424720e-03],
                  [-8.742600e-04, 9.156420e-03, 1.094866e-02, -7.841370e-02, 
                    -1.562827e-01, -7.841370e-02, 1.094866e-02, 9.156420e-03, 
                    -8.742600e-04],
                  [-1.166810e-03, 1.098012e-02, -5.897780e-03, -1.562827e-01, 
                    7.282593e-01, -1.562827e-01, -5.897780e-03, 1.098012e-02, 
                    -1.166810e-03],
                  [-8.742600e-04, 9.156420e-03, 1.094866e-02, -7.841370e-02, 
                    -1.562827e-01, -7.841370e-02, 1.094866e-02, 9.156420e-03, 
                    -8.742600e-04],
                  [-1.424720e-03, 8.998600e-04, 1.706347e-02, 1.094866e-02, 
                    -5.897780e-03, 1.094866e-02, 1.706347e-02, 8.998600e-04, 
                    -1.424720e-03],
                  [-2.424100e-04, -4.317530e-03, 8.998600e-04, 9.156420e-03, 
                    1.098012e-02, 9.156420e-03, 8.998600e-04, -4.317530e-03, 
                    -2.424100e-04],
                  [-9.570000e-04, -2.424100e-04, -1.424720e-03, -8.742600e-04, 
                    -1.166810e-03, -8.742600e-04, -1.424720e-03, -2.424100e-04,
                    -9.570000e-04]]) )
    filters['bfilts'] = (
        np.array([[6.125880e-03, -8.052600e-03, -2.103714e-02, -1.536890e-02, 
                   -1.851466e-02, -1.536890e-02, -2.103714e-02, -8.052600e-03, 
                   6.125880e-03, -1.287416e-02, -9.611520e-03, 1.023569e-02, 
                   6.009450e-03, 1.872620e-03, 6.009450e-03, 1.023569e-02, 
                   -9.611520e-03, -1.287416e-02, -5.641530e-03, 4.168400e-03, 
                   -2.382180e-02, -5.375324e-02, -2.076086e-02, -5.375324e-02,
                   -2.382180e-02, 4.168400e-03, -5.641530e-03, -8.957260e-03, 
                   -1.751170e-03, -1.836909e-02, 1.265655e-01, 2.996168e-01, 
                   1.265655e-01, -1.836909e-02, -1.751170e-03, -8.957260e-03, 
                   0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 
                   0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 
                   0.000000e+00, 8.957260e-03, 1.751170e-03, 1.836909e-02, 
                   -1.265655e-01, -2.996168e-01, -1.265655e-01, 1.836909e-02, 
                   1.751170e-03, 8.957260e-03, 5.641530e-03, -4.168400e-03, 
                   2.382180e-02, 5.375324e-02, 2.076086e-02, 5.375324e-02, 
                   2.382180e-02, -4.168400e-03, 5.641530e-03, 1.287416e-02, 
                   9.611520e-03, -1.023569e-02, -6.009450e-03, -1.872620e-03, 
                   -6.009450e-03, -1.023569e-02, 9.611520e-03, 1.287416e-02, 
                   -6.125880e-03, 8.052600e-03, 2.103714e-02, 1.536890e-02, 
                   1.851466e-02, 1.536890e-02, 2.103714e-02, 8.052600e-03, 
                   -6.125880e-03],
                  [-6.125880e-03, 1.287416e-02, 5.641530e-03, 8.957260e-03, 
                    0.000000e+00, -8.957260e-03, -5.641530e-03, -1.287416e-02, 
                    6.125880e-03, 8.052600e-03, 9.611520e-03, -4.168400e-03, 
                    1.751170e-03, 0.000000e+00, -1.751170e-03, 4.168400e-03, 
                    -9.611520e-03, -8.052600e-03, 2.103714e-02, -1.023569e-02, 
                    2.382180e-02, 1.836909e-02, 0.000000e+00, -1.836909e-02, 
                    -2.382180e-02, 1.023569e-02, -2.103714e-02, 1.536890e-02, 
                    -6.009450e-03, 5.375324e-02, -1.265655e-01, 0.000000e+00, 
                    1.265655e-01, -5.375324e-02, 6.009450e-03, -1.536890e-02, 
                    1.851466e-02, -1.872620e-03, 2.076086e-02, -2.996168e-01, 
                    0.000000e+00, 2.996168e-01, -2.076086e-02, 1.872620e-03, 
                    -1.851466e-02, 1.536890e-02, -6.009450e-03, 5.375324e-02, 
                    -1.265655e-01, 0.000000e+00, 1.265655e-01, -5.375324e-02, 
                    6.009450e-03, -1.536890e-02, 2.103714e-02, -1.023569e-02, 
                    2.382180e-02, 1.836909e-02, 0.000000e+00, -1.836909e-02, 
                    -2.382180e-02, 1.023569e-02, -2.103714e-02, 8.052600e-03, 
                    9.611520e-03, -4.168400e-03, 1.751170e-03, 0.000000e+00, 
                    -1.751170e-03, 4.168400e-03, -9.611520e-03, -8.052600e-03, 
                    -6.125880e-03, 1.287416e-02, 5.641530e-03, 8.957260e-03, 
                    0.000000e+00, -8.957260e-03, -5.641530e-03, -1.287416e-02, 
                    6.125880e-03]]).T )
    filters['bfilts'] = -filters['bfilts']

    return filters

# convert level and band to dictionary index
def LB2idx(lev,band,nlevs,nbands):
    if lev == 0:
        idx = 0
    elif lev == nlevs-1:
        # (Nlevels - ends)*Nbands + ends -1 (because zero indexed)
        idx = (((nlevs-2)*nbands)+2)-1
    else:
        # (level-first level) * nbands + first level + current band 
        #idx = (nbands*(lev-1))+1+band
        idx = (nbands*(lev-1))+1-band + 1
    return idx

# given and index into dictionary return level and band
def idx2LB(idx, nlevs, nbands):
    if idx == 0:
        return ('hi', -1)
    elif idx == ((nlevs-2)*nbands)+1:
        return ('lo', -1)
    else:
        lev = math.ceil(idx/nbands)
        band = (idx % nbands) + 1
        if band == nbands:
            band = 0
        return (lev, band)

# find next largest size in list
def nextSz(size, sizeList):
    ## make sure sizeList is strictly increasing
    if sizeList[0] > sizeList[len(sizeList)-1]:
        sizeList = sizeList[::-1]
    outSize = (0,0)
    idx = 0;
    while outSize == (0,0) and idx < len(sizeList):
        if sizeList[idx] > size:
            outSize = sizeList[idx]
        idx += 1
    return outSize

def mkImpulse(*args):
    # create an image that is all zeros except for an impulse
    if(len(args) == 0):
        print "mkImpulse(size, origin, amplitude)"
        print "first input parameter is required"
        return
    
    if(isinstance(args[0], int)):
        sz = (args[0], args[0])
    elif(isinstance(args[0], tuple)):
        sz = args[0]
    else:
        print "size parameter must be either an integer or a tuple"
        return

    if(len(args) > 1):
        origin = args[1]
    else:
        origin = ( np.ceil(sz[0]/2.0), np.ceil(sz[1]/2.0) )

    if(len(args) > 2):
        amplitude = args[2]
    else:
        amplitude = 1

    res = np.zeros(sz);
    res[origin[0], origin[1]] = amplitude

    return res
