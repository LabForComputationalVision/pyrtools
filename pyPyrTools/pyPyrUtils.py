import matplotlib.pyplot
import matplotlib.cm
import numpy
import pylab
import scipy.signal
import scipy.stats
from scipy import interpolate
import math
import struct
import re
import sys
from PyQt4 import QtGui
from PyQt4 import QtCore
import JBhelpers
import PIL
import ImageTk
import Tkinter
import ctypes
lib = ctypes.cdll.LoadLibrary('./wrapConv.so')


##############
#  This code uses C code from wrapConv.so.  To compile type:
#  gcc -shared -L/users-local/ryoung/anaconda2/lib -I/users-local/ryoung/anaconda2/include/python2.7/ -lpython2.7 -o wrapConv.so -fPIC convolve.c edges.c wrap.c internal_pointOp.c


# Compute maximum pyramid height for given image and filter sizes.
# Specifically: the number of corrDn operations that can be sequentially
# performed when subsampling by a factor of 2.
def maxPyrHt_old(imsz, filtsz):
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

    if filtsz[0] == 1 or filtsz[1] == 1:
        filtsz = (max(filtsz), max(filtsz))

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

def maxPyrHt(imsz, filtsz):
    if not isinstance(imsz, tuple) or not isinstance(filtsz, tuple):
        if imsz < filtsz:
            return 0
    else:
        if len(imsz) == 1:
            imsz = (imsz[0], 1)
        if len(filtsz) == 1:
            filtsz = (filtsz[0], 1)
        #if filtsz[1] == 1:  # new
        #    filtsz = (filtsz[1], filtsz[0])
        if imsz[0] < filtsz[0] or imsz[1] < filtsz[1]:
            return 0


    if not isinstance(imsz, tuple) and not isinstance(filtsz, tuple):
        imsz = imsz
        filtsz = filtsz
    elif imsz[0] == 1 or imsz[1] == 1:         # 1D image
        imsz = imsz[0] * imsz[1]
        filtsz = filtsz[0] * filtsz[1]
    elif filtsz[0] == 1 or filtsz[1] == 1:   # 2D image, 1D filter
        filtsz = (filtsz[0], filtsz[0])

    if not isinstance(imsz, tuple) and not isinstance(filtsz, tuple) and imsz < filtsz:
        height = 0
    elif not isinstance(imsz, tuple) and not isinstance(filtsz, tuple):
        height = 1 + maxPyrHt( numpy.floor(imsz/2.0), filtsz )
    else:
        height = 1 + maxPyrHt( (numpy.floor(imsz[0]/2.0), 
                                numpy.floor(imsz[1]/2.0)), 
                               filtsz )

    return height

# returns a vector of binomial coefficients of order (size-1)
def binomialFilter(size):
    if size < 2:
        print "Error: size argument must be larger than 1"
        exit(1)
    
    kernel = numpy.array([[0.5], [0.5]])

    for i in range(0, size-2):
        kernel = scipy.signal.convolve(numpy.array([[0.5], [0.5]]), kernel)

    return numpy.asarray(kernel)

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
def namedFilter(name):
    if len(name) > 5 and name[:5] == "binom":
        kernel = math.sqrt(2) * binomialFilter(int(name[5:]))
    elif name is "qmf5":
        kernel = numpy.array([[-0.076103], [0.3535534], [0.8593118], [0.3535534], [-0.076103]])
    elif name is "qmf9":
        kernel = numpy.array([[0.02807382], [-0.060944743], [-0.073386624], [0.41472545], [0.7973934], [0.41472545], [-0.073386624], [-0.060944743], [0.02807382]])
    elif name is "qmf13":
        kernel = numpy.array([[-0.014556438], [0.021651438], [0.039045125], [-0.09800052], [-0.057827797], [0.42995453], [0.7737113], [0.42995453], [-0.057827797], [-0.09800052], [0.039045125], [0.021651438], [-0.014556438]])
    elif name is "qmf8":
        kernel = math.sqrt(2) * numpy.array([[0.00938715], [-0.07065183], [0.06942827], [0.4899808], [0.4899808], [0.06942827], [-0.07065183], [0.00938715]])
    elif name is "qmf12":
        kernel = math.sqrt(2) * numpy.array([[-0.003809699], [0.01885659], [-0.002710326], [-0.08469594], [0.08846992], [0.4843894], [0.4843894], [0.08846992], [-0.08469594], [-0.002710326], [0.01885659], [-0.003809699]])
    elif name is "qmf16":
        kernel = math.sqrt(2) * numpy.array([[0.001050167], [-0.005054526], [-0.002589756], [0.0276414], [-0.009666376], [-0.09039223], [0.09779817], [0.4810284], [0.4810284], [0.09779817], [-0.09039223], [-0.009666376], [0.0276414], [-0.002589756], [-0.005054526], [0.001050167]])
    elif name is "haar":
        kernel = numpy.array([[1], [1]]) / math.sqrt(2)
    elif name is "daub2":
        kernel = numpy.array([[0.482962913145], [0.836516303738], [0.224143868042], [-0.129409522551]]);
    elif name is "daub3":
        kernel = numpy.array([[0.332670552950], [0.806891509311], [0.459877502118], [-0.135011020010], [-0.085441273882], [0.035226291882]])
    elif name is "daub4":
        kernel = numpy.array([[0.230377813309], [0.714846570553], [0.630880767930], [-0.027983769417], [-0.187034811719], [0.030841381836], [0.032883011667], [-0.010597401785]])
    elif name is "gauss5":  # for backward-compatibility
        kernel = math.sqrt(2) * numpy.array([[0.0625], [0.25], [0.375], [0.25], [0.0625]])
    elif name is "gauss3":  # for backward-compatibility
        kernel = math.sqrt(2) * numpy.array([[0.25], [0.5], [0.25]])
    else:
        print "Error: Bad filter name: %s" % (name)
        exit(1)
    return numpy.array(kernel)

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))

def compareRecon(recon1, recon2):
    prec = -11
    if recon1.shape != recon2.shape:
        print 'shape is different!'
        print recon1.shape
        print recon2.shape
        return 0

    for i in range(recon1.shape[0]):
        for j in range(recon2.shape[1]):
            if numpy.absolute(recon1[i,j].real - recon2[i,j].real) > math.pow(10,-11):
                print "real: i=%d j=%d %.15f %.15f diff=%.15f" % (i, j, recon1[i,j].real, recon2[i,j].real, numpy.absolute(recon1[i,j].real-recon2[i,j].real))
                return 0
            ## FIX: need a better way to test
            # if we have many significant digits to the left of decimal we 
            #   need to be less stringent about digits to the right.
            # The code below works, but there must be a better way.
            if isinstance(recon1, complex):
                if int(math.log(numpy.abs(recon1[i,j].imag), 10)) > 1:
                    prec = prec + int(math.log(numpy.abs(recon1[i,j].imag), 10))
                    if prec > 0:
                        prec = -1
                print prec
                if numpy.absolute(recon1[i,j].imag - recon2[i,j].imag) > math.pow(10, prec):
                    print "imag: i=%d j=%d %.15f %.15f diff=%.15f" % (i, j, recon1[i,j].imag, recon2[i,j].imag, numpy.absolute(recon1[i,j].imag-recon2[i,j].imag))
                    return 0

    return 1

def comparePyr(matPyr, pyPyr):
    prec = math.pow(10,-9)
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
        print "size difference: %d != %d, returning 0" % (matSz, pySz)
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
            print "some pyramid elements not identical: checking..."
            for i in range(bandSz[0]):
                for j in range(bandSz[1]):
                    if matTmp[i,j] != pyPyr.pyr[idx][i,j]:
                        if ( math.fabs(matTmp[i,j] - pyPyr.pyr[idx][i,j]) > 
                             prec ):
                            print "failed level:%d element:%d %d value:%.15f %.15f" % (idx, i, j, matTmp[i,j], pyPyr.pyr[idx][i,j])
                            return 0
            print "same to at least %f" % prec

    return 1

def mkAngularSine(*args):
    # IM = mkAngularSine(SIZE, HARMONIC, AMPL, PHASE, ORIGIN)
    #
    # Make an angular sinusoidal image:
    #     AMPL * sin( HARMONIC*theta + PHASE),
    # where theta is the angle about the origin.
    # SIZE specifies the matrix size, as for zeros().  
    # AMPL (default = 1) and PHASE (default = 0) are optional.
    
    # Eero Simoncelli, 2/97.  Python port by Rob Young, 7/15.

    if len(args) == 0:
        print "mkAngularSine(SIZE, HARMONIC, AMPL, PHASE, ORIGIN)"
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
        harmonic = args[1]
    else:
        harmonic = 1

    if len(args) > 2:
        ampl = args[2]
    else:
        ampl = 1

    if len(args) > 3:
        ph = args[3]
    else:
        ph = 0

    if len(args) > 4:
        origin = args[4]
    else:
        origin = ( (sz[0]+1.0)/2.0, (sz[1]+1.0)/2.0 )
        
    res = ampl * numpy.sin( harmonic * mkAngle(sz, ph, origin) + ph )

    return res

def mkGaussian(*args):
# IM = mkGaussian(SIZE, COVARIANCE, MEAN, AMPLITUDE)
# 
# Compute a matrix with dimensions SIZE (a [Y X] 2-vector, or a
#   scalar) containing a Gaussian function, centered at pixel position
# specified by MEAN (default = (size+1)/2), with given COVARIANCE (can
# be a scalar, 2-vector, or 2x2 matrix.  Default = (min(size)/6)^2),
# and AMPLITUDE.  AMPLITUDE='norm' (default) will produce a
# probability-normalized function.  All but the first argument are
# optional.
#
# Eero Simoncelli, 6/96. Python port by Rob Young, 7/15.

    if len(args) == 0:
        print "mkRamp(SIZE, COVARIANCE, MEAN, AMPLITUDE)"
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
        cov = args[1]
    else:
        cov = (min([sz[0], sz[1]]) / 6.0) ** 2

    if len(args) > 2:
        mn = args[2]
        if isinstance(mn, int):
            mn = [mn, mn]
    else:
        mn = ( (sz[0]+1.0)/2.0, (sz[1]+1.0)/2.0 )

    if len(args) > 3:
        ampl = args[3]
    else:
        ampl = 'norm'

    #---------------------------------------------------------------
        
    (xramp, yramp) = numpy.meshgrid(numpy.array(range(1,sz[1]+1))-mn[1], 
                                    numpy.array(range(1,sz[0]+1))-mn[0])

    if isinstance(cov, (int, long, float)):
        if 'norm' == ampl:
            ampl = 1.0 / (2.0 * numpy.pi * cov)
        e = ( (xramp**2) + (yramp**2) ) / ( -2.0 * cov )
    elif len(cov) == 2 and isinstance(cov[0], (int, long, float)):
        if 'norm' == ampl:
            if cov[0]*cov[1] < 0:
                ampl = 1.0 / (2.0 * numpy.pi * 
                              numpy.sqrt(complex(cov[0] * cov[1])))
            else:
                ampl = 1.0 / (2.0 * numpy.pi * numpy.sqrt(cov[0] * cov[1]))
        e = ( (xramp**2) / (-2 * cov[1]) ) + ( (yramp**2) / (-2 * cov[0]) )
    else:
        if 'norm' == ampl:
            detCov = numpy.linalg.det(cov)
            if (detCov < 0).any():
                detCovComplex = numpy.empty(detCov.shape, dtype=complex)
                detCovComplex.real = detCov
                detCovComplex.imag = numpy.zeros(detCov.shape)
                ampl = 1.0 / ( 2.0 * numpy.pi * numpy.sqrt( detCovComplex ) )
            else:
                ampl = 1.0 / (2.0 * numpy.pi * numpy.sqrt( numpy.linalg.det(cov) ) )
        cov = - numpy.linalg.inv(cov) / 2.0
        e = (cov[1,1] * xramp**2) + ( 
            (cov[0,1]+cov[1,0])*(xramp*yramp) ) + ( cov[0,0] * yramp**2)
        
    res = ampl * numpy.exp(e)
    
    return res


def mkDisc(*args):
 # IM = mkDisc(SIZE, RADIUS, ORIGIN, TWIDTH, VALS)

 # Make a "disk" image.  SIZE specifies the matrix size, as for
 # zeros().  RADIUS (default = min(size)/4) specifies the radius of 
 # the disk.  ORIGIN (default = (size+1)/2) specifies the 
 # location of the disk center.  TWIDTH (in pixels, default = 2) 
 # specifies the width over which a soft threshold transition is made.
 # VALS (default = [0,1]) should be a 2-vector containing the
 # intensity value inside and outside the disk.  

 # Eero Simoncelli, 6/96. Python port by Rob Young, 7/15.

    if len(args) == 0:
        print "mkDisc(SIZE, RADIUS, ORIGIN, TWIDTH, VALS)"
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
        rad = args[1]
    else:
        rad = min(sz) / 4.0

    if len(args) > 2:
        origin = args[2]
    else:
        origin = ( (sz[0]+1.0)/2.0, (sz[1]+1.0)/2.0 )

    if len(args) > 3:
        twidth = args[3]
    else:
        twidth = twidth = 2
        
    if len(args) > 4:
        vals = args[4]
    else:
        vals = (1,0)

    #--------------------------------------------------------------

    res = mkR(sz, 1, origin)

    if abs(twidth) < sys.float_info.min:
        res = vals[1] + (vals[0] - vals[1]) * (res <= rad);
    else:
        [Xtbl, Ytbl] = rcosFn(twidth, rad, [vals[0], vals[1]]);
        res = pointOp(res, Ytbl, Xtbl[0], Xtbl[1]-Xtbl[0], 0);

    return res

def mkSine(*args):
# IM = mkSine(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN)
#      or
# IM = mkSine(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN)
# 
# Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
# containing samples of a 2D sinusoid, with given PERIOD (in pixels),
# DIRECTION (radians, CW from X-axis, default = 0), AMPLITUDE (default
# = 1), and PHASE (radians, relative to ORIGIN, default = 0).  ORIGIN
# defaults to the center of the image.
# 
# In the second form, FREQ is a 2-vector of frequencies (radians/pixel).
#
# Eero Simoncelli, 6/96. Python version by Rob Young, 7/15.

    # REQUIRED args:

    if len(args) < 2:
        print "mkSine(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN)"
        print "       or"
        print "mkSine(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN)"
        print "first two arguments are required"
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print "first argument must be a two element tuple or an integer"
            exit(1)

    if isinstance(args[1], (int, float, long)):
        frequency = (2.0 * numpy.pi) / args[1]
        # OPTIONAL args:
        if len(args) > 2:
            direction = args[2]
        else:
            direction = 0
        if len(args) > 3:
            amplitude = args[3]
        else:
            amplitude = 1
        if len(args) > 4:
            phase = args[4]
        else:
            phase = 0
        if len(args) > 5:
            origin = args[5]
        else:
            origin = 'not set'
    else:
        frequency = numpy.linalg.norm(args[1])
        direction = math.atan2(args[1][0], args[1][1])
        # OPTIONAL args:
        if len(args) > 2:
            amplitude = args[2]
        else:
            amplitude = 1
        if len(args) > 3:
            phase = args[3]
        else:
            phase = 0
        if len(args) > 4:
            origin = args[4]
        else:
            origin = 'not set'

    #----------------------------------------------------------------

    if origin == 'not set':
        res = amplitude * numpy.sin(mkRamp(sz, direction, frequency, phase))
    else:
        res = amplitude * numpy.sin(mkRamp(sz, direction, frequency, phase, 
                                           [origin[0]-1, origin[1]-1]))

    return res

def mkZonePlate(*args):
    # IM = mkZonePlate(SIZE, AMPL, PHASE)
    #
    # Make a "zone plate" image:
    #     AMPL * cos( r^2 + PHASE)
    # SIZE specifies the matrix size, as for zeros().  
    # AMPL (default = 1) and PHASE (default = 0) are optional.
    #
    # Eero Simoncelli, 6/96.  Python port by Rob Young, 7/15.

    # REQUIRED ARGS:

    if len(args) == 0:
        print "mkZonePlate(SIZE, AMPL, PHASE)"
        print "first argument is required"
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print "first argument must be a two element tuple or an integer"
            exit(1)
    
    #---------------------------------------------------------------------
    # OPTIONAL ARGS
    if len(args) > 1:
        ampl = args[1]
    else:
        ampl = 1
    if len(args) > 2:
        ph = args[2]
    else:
        ph = 0

    #---------------------------------------------------------------------

    res = ampl * numpy.cos( (numpy.pi / max(sz)) * mkR(sz, 2) + ph )

    return res

def mkSquare(*args):
    # IM = mkSquare(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN, TWIDTH)
    #      or
    # IM = mkSquare(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN, TWIDTH)
    # 
    # Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
    # containing samples of a 2D square wave, with given PERIOD (in
    # pixels), DIRECTION (radians, CW from X-axis, default = 0), AMPLITUDE
    # (default = 1), and PHASE (radians, relative to ORIGIN, default = 0).
    # ORIGIN defaults to the center of the image.  TWIDTH specifies width
    # of raised-cosine edges on the bars of the grating (default =
    # min(2,period/3)).
    # 
    # In the second form, FREQ is a 2-vector of frequencies (radians/pixel).
    #
    # Eero Simoncelli, 6/96. Python port by Rob Young, 7/15.
    #
    # TODO: Add duty cycle.  

    # REQUIRED ARGS:

    if len(args) < 2:
        print "mkSquare(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN, TWIDTH)"
        print "       or"
        print "mkSquare(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN, TWIDTH)"
        print "first two arguments are required"
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print "first argument must be a two element tuple or an integer"
            exit(1)
    
    if isinstance(args[1], (int, float, long)):
        frequency = (2.0 * numpy.pi) / args[1]
        # OPTIONAL args:
        if len(args) > 2:
            direction = args[2]
        else:
            direction = 0
        if len(args) > 3:
            amplitude = args[3]
        else:
            amplitude = 1
        if len(args) > 4:
            phase = args[4]
        else:
            phase = 0
        if len(args) > 5:
            origin = args[5]
        else:
            origin = 'not set'
        if len(args) > 6:
            transition = args[6]
        else:
            transition = min(2, 2.0 * numpy.pi / (3.0*frequency))
    else:
        frequency = numpy.linalg.norm(args[1])
        direction = math.atan2(args[1][0], args[1][1])
        # OPTIONAL args:
        if len(args) > 2:
            amplitude = args[2]
        else:
            amplitude = 1
        if len(args) > 3:
            phase = args[3]
        else:
            phase = 0
        if len(args) > 4:
            origin = args[4]
        else:
            origin = 'not set'
        if len(args) > 5:
            transition = args[5]
        else:
            transition = min(2, 2.0 * numpy.pi / (3.0*frequency))

    #------------------------------------------------------------

    if origin != 'not set':
        res = mkRamp(sz, direction, frequency, phase, 
                     (origin[0]-1, origin[1]-1)) - numpy.pi/2.0
    else:
        res = mkRamp(sz, direction, frequency, phase) - numpy.pi/2.0

    [Xtbl, Ytbl] = rcosFn(transition * frequency, numpy.pi/2.0, 
                          [-amplitude, amplitude])

    res = pointOp(abs(((res+numpy.pi) % (2.0*numpy.pi))-numpy.pi), Ytbl, 
                  Xtbl[0], Xtbl[1]-Xtbl[0], 0)

    return res

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

    [xramp, yramp] = numpy.meshgrid( xinc * (numpy.array(range(sz[1]))-origin[1]),
                                  yinc * (numpy.array(range(sz[0]))-origin[0]) )

    res = intercept + xramp + yramp

    return res

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
    filters['harmonics'] = numpy.array([0])
    filters['lo0filt'] =  ( 
        numpy.array([[-4.514000e-04, -1.137100e-04, -3.725800e-04, -3.743860e-03, 
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
        numpy.array([[-2.257000e-04, -8.064400e-04, -5.686000e-05, 8.741400e-04, 
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
    filters['mtx'] = numpy.array([ 1.000000 ])
    filters['hi0filt'] = ( 
        numpy.array([[5.997200e-04, -6.068000e-05, -3.324900e-04, -3.325600e-04, 
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
        numpy.array([-9.066000e-05, -1.738640e-03, -4.942500e-03, -7.889390e-03, 
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
    filters['harmonics'] = numpy.array([ 1 ])
    filters['mtx'] = numpy.eye(2)
    filters['lo0filt'] = ( 
        numpy.array([[-8.701000e-05, -1.354280e-03, -1.601260e-03, -5.033700e-04, 
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
        numpy.array([[-4.350000e-05, 1.207800e-04, -6.771400e-04, -1.243400e-04, 
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
        numpy.array([[-9.570000e-04, -2.424100e-04, -1.424720e-03, -8.742600e-04, 
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
        numpy.array([[6.125880e-03, -8.052600e-03, -2.103714e-02, -1.536890e-02, 
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
    filters['bfilts'] = numpy.negative(filters['bfilts'])

    return filters

def sp3Filters():
    filters = {}
    filters['harmonics'] = numpy.array([1, 3])
    filters['mtx'] = (
        numpy.array([[0.5000, 0.3536, 0, -0.3536],
                  [-0.0000, 0.3536, 0.5000, 0.3536],
                  [0.5000, -0.3536, 0, 0.3536],
                  [-0.0000, 0.3536, -0.5000, 0.3536]]))
    filters['hi0filt'] = (
        numpy.array([[-4.0483998600E-4, -6.2596000498E-4, -3.7829999201E-5,
                    8.8387000142E-4, 1.5450799838E-3, 1.9235999789E-3,
                    2.0687500946E-3, 2.0898699295E-3, 2.0687500946E-3,
                    1.9235999789E-3, 1.5450799838E-3, 8.8387000142E-4,
                    -3.7829999201E-5, -6.2596000498E-4, -4.0483998600E-4],
                  [-6.2596000498E-4, -3.2734998967E-4, 7.7435001731E-4,
                    1.5874400269E-3, 2.1750701126E-3, 2.5626500137E-3,
                    2.2892199922E-3, 1.9755100366E-3, 2.2892199922E-3,
                    2.5626500137E-3, 2.1750701126E-3, 1.5874400269E-3,
                    7.7435001731E-4, -3.2734998967E-4, -6.2596000498E-4],
                  [-3.7829999201E-5, 7.7435001731E-4, 1.1793200392E-3,
                    1.4050999889E-3, 2.2253401112E-3, 2.1145299543E-3,
                    3.3578000148E-4, -8.3368999185E-4, 3.3578000148E-4,
                    2.1145299543E-3, 2.2253401112E-3, 1.4050999889E-3,
                    1.1793200392E-3, 7.7435001731E-4, -3.7829999201E-5],
                  [8.8387000142E-4, 1.5874400269E-3, 1.4050999889E-3,
                   1.2960999738E-3, -4.9274001503E-4, -3.1295299996E-3,
                   -4.5751798898E-3, -5.1014497876E-3, -4.5751798898E-3,
                   -3.1295299996E-3, -4.9274001503E-4, 1.2960999738E-3,
                   1.4050999889E-3, 1.5874400269E-3, 8.8387000142E-4],
                  [1.5450799838E-3, 2.1750701126E-3, 2.2253401112E-3,
                   -4.9274001503E-4, -6.3222697936E-3, -2.7556000277E-3,
                   5.3632198833E-3, 7.3032598011E-3, 5.3632198833E-3,
                   -2.7556000277E-3, -6.3222697936E-3, -4.9274001503E-4,
                   2.2253401112E-3, 2.1750701126E-3, 1.5450799838E-3],
                  [1.9235999789E-3, 2.5626500137E-3, 2.1145299543E-3,
                   -3.1295299996E-3, -2.7556000277E-3, 1.3962360099E-2,
                   7.8046298586E-3, -9.3812197447E-3, 7.8046298586E-3,
                   1.3962360099E-2, -2.7556000277E-3, -3.1295299996E-3,
                   2.1145299543E-3, 2.5626500137E-3, 1.9235999789E-3],
                  [2.0687500946E-3, 2.2892199922E-3, 3.3578000148E-4,
                   -4.5751798898E-3, 5.3632198833E-3, 7.8046298586E-3,
                   -7.9501636326E-2, -0.1554141641, -7.9501636326E-2,
                   7.8046298586E-3, 5.3632198833E-3, -4.5751798898E-3,
                   3.3578000148E-4, 2.2892199922E-3, 2.0687500946E-3],
                  [2.0898699295E-3, 1.9755100366E-3, -8.3368999185E-4,
                   -5.1014497876E-3, 7.3032598011E-3, -9.3812197447E-3,
                   -0.1554141641, 0.7303866148, -0.1554141641, 
                   -9.3812197447E-3, 7.3032598011E-3, -5.1014497876E-3,
                   -8.3368999185E-4, 1.9755100366E-3, 2.0898699295E-3],
                  [2.0687500946E-3, 2.2892199922E-3, 3.3578000148E-4,
                   -4.5751798898E-3, 5.3632198833E-3, 7.8046298586E-3,
                   -7.9501636326E-2, -0.1554141641, -7.9501636326E-2,
                   7.8046298586E-3, 5.3632198833E-3, -4.5751798898E-3,
                   3.3578000148E-4, 2.2892199922E-3, 2.0687500946E-3],
                  [1.9235999789E-3, 2.5626500137E-3, 2.1145299543E-3,
                   -3.1295299996E-3, -2.7556000277E-3, 1.3962360099E-2,
                   7.8046298586E-3, -9.3812197447E-3, 7.8046298586E-3,
                   1.3962360099E-2, -2.7556000277E-3, -3.1295299996E-3,
                   2.1145299543E-3, 2.5626500137E-3, 1.9235999789E-3],
                  [1.5450799838E-3, 2.1750701126E-3, 2.2253401112E-3,
                   -4.9274001503E-4, -6.3222697936E-3, -2.7556000277E-3,
                   5.3632198833E-3, 7.3032598011E-3, 5.3632198833E-3,
                   -2.7556000277E-3, -6.3222697936E-3, -4.9274001503E-4,
                   2.2253401112E-3, 2.1750701126E-3, 1.5450799838E-3],
                  [8.8387000142E-4, 1.5874400269E-3, 1.4050999889E-3,
                   1.2960999738E-3, -4.9274001503E-4, -3.1295299996E-3,
                   -4.5751798898E-3, -5.1014497876E-3, -4.5751798898E-3,
                   -3.1295299996E-3, -4.9274001503E-4, 1.2960999738E-3,
                   1.4050999889E-3, 1.5874400269E-3, 8.8387000142E-4],
                  [-3.7829999201E-5, 7.7435001731E-4, 1.1793200392E-3,
                    1.4050999889E-3, 2.2253401112E-3, 2.1145299543E-3,
                    3.3578000148E-4, -8.3368999185E-4, 3.3578000148E-4,
                    2.1145299543E-3, 2.2253401112E-3, 1.4050999889E-3,
                    1.1793200392E-3, 7.7435001731E-4, -3.7829999201E-5],
                  [-6.2596000498E-4, -3.2734998967E-4, 7.7435001731E-4,
                    1.5874400269E-3, 2.1750701126E-3, 2.5626500137E-3,
                    2.2892199922E-3, 1.9755100366E-3, 2.2892199922E-3,
                    2.5626500137E-3, 2.1750701126E-3, 1.5874400269E-3,
                    7.7435001731E-4, -3.2734998967E-4, -6.2596000498E-4],
                  [-4.0483998600E-4, -6.2596000498E-4, -3.7829999201E-5,
                    8.8387000142E-4, 1.5450799838E-3, 1.9235999789E-3,
                    2.0687500946E-3, 2.0898699295E-3, 2.0687500946E-3,
                    1.9235999789E-3, 1.5450799838E-3, 8.8387000142E-4,
                    -3.7829999201E-5, -6.2596000498E-4, -4.0483998600E-4]]))
    filters['lo0filt'] = (
        numpy.array([[-8.7009997515E-5, -1.3542800443E-3, -1.6012600390E-3,
                    -5.0337001448E-4, 2.5240099058E-3, -5.0337001448E-4,
                    -1.6012600390E-3, -1.3542800443E-3, -8.7009997515E-5],
                  [-1.3542800443E-3, 2.9215801042E-3, 7.5227199122E-3,
                    8.2244202495E-3, 1.1076199589E-3, 8.2244202495E-3,
                    7.5227199122E-3, 2.9215801042E-3, -1.3542800443E-3],
                  [-1.6012600390E-3, 7.5227199122E-3, -7.0612900890E-3,
                    -3.7694871426E-2, -3.2971370965E-2, -3.7694871426E-2,
                    -7.0612900890E-3, 7.5227199122E-3, -1.6012600390E-3],
                  [-5.0337001448E-4, 8.2244202495E-3, -3.7694871426E-2,
                    4.3813198805E-2, 0.1811603010, 4.3813198805E-2,
                    -3.7694871426E-2, 8.2244202495E-3, -5.0337001448E-4],
                  [2.5240099058E-3, 1.1076199589E-3, -3.2971370965E-2,
                   0.1811603010, 0.4376249909, 0.1811603010,
                   -3.2971370965E-2, 1.1076199589E-3, 2.5240099058E-3],
                  [-5.0337001448E-4, 8.2244202495E-3, -3.7694871426E-2,
                    4.3813198805E-2, 0.1811603010, 4.3813198805E-2,
                    -3.7694871426E-2, 8.2244202495E-3, -5.0337001448E-4],
                  [-1.6012600390E-3, 7.5227199122E-3, -7.0612900890E-3,
                    -3.7694871426E-2, -3.2971370965E-2, -3.7694871426E-2,
                    -7.0612900890E-3, 7.5227199122E-3, -1.6012600390E-3],
                  [-1.3542800443E-3, 2.9215801042E-3, 7.5227199122E-3,
                    8.2244202495E-3, 1.1076199589E-3, 8.2244202495E-3,
                    7.5227199122E-3, 2.9215801042E-3, -1.3542800443E-3],
                  [-8.7009997515E-5, -1.3542800443E-3, -1.6012600390E-3,
                    -5.0337001448E-4, 2.5240099058E-3, -5.0337001448E-4,
                    -1.6012600390E-3, -1.3542800443E-3, -8.7009997515E-5]]))
    filters['lofilt'] = (
        numpy.array([[-4.3500000174E-5, 1.2078000145E-4, -6.7714002216E-4,
                    -1.2434000382E-4, -8.0063997302E-4, -1.5970399836E-3,
                    -2.5168000138E-4, -4.2019999819E-4, 1.2619999470E-3,
                    -4.2019999819E-4, -2.5168000138E-4, -1.5970399836E-3,
                    -8.0063997302E-4, -1.2434000382E-4, -6.7714002216E-4,
                    1.2078000145E-4, -4.3500000174E-5],
                  [1.2078000145E-4, 4.4606000301E-4, -5.8146001538E-4,
                   5.6215998484E-4, -1.3688000035E-4, 2.3255399428E-3,
                   2.8898599558E-3, 4.2872801423E-3, 5.5893999524E-3,
                   4.2872801423E-3, 2.8898599558E-3, 2.3255399428E-3,
                   -1.3688000035E-4, 5.6215998484E-4, -5.8146001538E-4,
                   4.4606000301E-4, 1.2078000145E-4],
                  [-6.7714002216E-4, -5.8146001538E-4, 1.4607800404E-3,
                    2.1605400834E-3, 3.7613599561E-3, 3.0809799209E-3,
                    4.1121998802E-3, 2.2212199401E-3, 5.5381999118E-4,
                    2.2212199401E-3, 4.1121998802E-3, 3.0809799209E-3,
                    3.7613599561E-3, 2.1605400834E-3, 1.4607800404E-3,
                    -5.8146001538E-4, -6.7714002216E-4],
                  [-1.2434000382E-4, 5.6215998484E-4, 2.1605400834E-3,
                    3.1757799443E-3, 3.1846798956E-3, -1.7774800071E-3,
                    -7.4316998944E-3, -9.0569201857E-3, -9.6372198313E-3,
                    -9.0569201857E-3, -7.4316998944E-3, -1.7774800071E-3,
                    3.1846798956E-3, 3.1757799443E-3, 2.1605400834E-3,
                    5.6215998484E-4, -1.2434000382E-4],
                  [-8.0063997302E-4, -1.3688000035E-4, 3.7613599561E-3,
                    3.1846798956E-3, -3.5306399222E-3, -1.2604200281E-2,
                    -1.8847439438E-2, -1.7508180812E-2, -1.6485679895E-2,
                    -1.7508180812E-2, -1.8847439438E-2, -1.2604200281E-2,
                    -3.5306399222E-3, 3.1846798956E-3, 3.7613599561E-3,
                    -1.3688000035E-4, -8.0063997302E-4],
                  [-1.5970399836E-3, 2.3255399428E-3, 3.0809799209E-3,
                    -1.7774800071E-3, -1.2604200281E-2, -2.0229380578E-2,
                    -1.1091699824E-2, 3.9556599222E-3, 1.4385120012E-2,
                    3.9556599222E-3, -1.1091699824E-2, -2.0229380578E-2,
                    -1.2604200281E-2, -1.7774800071E-3, 3.0809799209E-3,
                    2.3255399428E-3, -1.5970399836E-3],
                  [-2.5168000138E-4, 2.8898599558E-3, 4.1121998802E-3,
                    -7.4316998944E-3, -1.8847439438E-2, -1.1091699824E-2,
                    2.1906599402E-2, 6.8065837026E-2, 9.0580143034E-2,
                    6.8065837026E-2, 2.1906599402E-2, -1.1091699824E-2,
                    -1.8847439438E-2, -7.4316998944E-3, 4.1121998802E-3,
                    2.8898599558E-3, -2.5168000138E-4],
                  [-4.2019999819E-4, 4.2872801423E-3, 2.2212199401E-3,
                    -9.0569201857E-3, -1.7508180812E-2, 3.9556599222E-3,
                    6.8065837026E-2, 0.1445499808, 0.1773651242,
                    0.1445499808, 6.8065837026E-2, 3.9556599222E-3,
                    -1.7508180812E-2, -9.0569201857E-3, 2.2212199401E-3,
                    4.2872801423E-3, -4.2019999819E-4],
                  [1.2619999470E-3, 5.5893999524E-3, 5.5381999118E-4,
                   -9.6372198313E-3, -1.6485679895E-2, 1.4385120012E-2,
                   9.0580143034E-2, 0.1773651242, 0.2120374441,
                   0.1773651242, 9.0580143034E-2, 1.4385120012E-2,
                   -1.6485679895E-2, -9.6372198313E-3, 5.5381999118E-4,
                   5.5893999524E-3, 1.2619999470E-3],
                  [-4.2019999819E-4, 4.2872801423E-3, 2.2212199401E-3,
                    -9.0569201857E-3, -1.7508180812E-2, 3.9556599222E-3,
                    6.8065837026E-2, 0.1445499808, 0.1773651242,
                    0.1445499808, 6.8065837026E-2, 3.9556599222E-3,
                    -1.7508180812E-2, -9.0569201857E-3, 2.2212199401E-3,
                    4.2872801423E-3, -4.2019999819E-4],
                  [-2.5168000138E-4, 2.8898599558E-3, 4.1121998802E-3,
                    -7.4316998944E-3, -1.8847439438E-2, -1.1091699824E-2,
                    2.1906599402E-2, 6.8065837026E-2, 9.0580143034E-2,
                    6.8065837026E-2, 2.1906599402E-2, -1.1091699824E-2,
                    -1.8847439438E-2, -7.4316998944E-3, 4.1121998802E-3,
                    2.8898599558E-3, -2.5168000138E-4],
                  [-1.5970399836E-3, 2.3255399428E-3, 3.0809799209E-3,
                    -1.7774800071E-3, -1.2604200281E-2, -2.0229380578E-2,
                    -1.1091699824E-2, 3.9556599222E-3, 1.4385120012E-2,
                    3.9556599222E-3, -1.1091699824E-2, -2.0229380578E-2,
                    -1.2604200281E-2, -1.7774800071E-3, 3.0809799209E-3,
                    2.3255399428E-3, -1.5970399836E-3],
                  [-8.0063997302E-4, -1.3688000035E-4, 3.7613599561E-3,
                    3.1846798956E-3, -3.5306399222E-3, -1.2604200281E-2,
                    -1.8847439438E-2, -1.7508180812E-2, -1.6485679895E-2,
                    -1.7508180812E-2, -1.8847439438E-2, -1.2604200281E-2,
                    -3.5306399222E-3, 3.1846798956E-3, 3.7613599561E-3,
                    -1.3688000035E-4, -8.0063997302E-4],
                  [-1.2434000382E-4, 5.6215998484E-4, 2.1605400834E-3,
                    3.1757799443E-3, 3.1846798956E-3, -1.7774800071E-3,
                    -7.4316998944E-3, -9.0569201857E-3, -9.6372198313E-3,
                    -9.0569201857E-3, -7.4316998944E-3, -1.7774800071E-3,
                    3.1846798956E-3, 3.1757799443E-3, 2.1605400834E-3,
                    5.6215998484E-4, -1.2434000382E-4],
                  [-6.7714002216E-4, -5.8146001538E-4, 1.4607800404E-3,
                    2.1605400834E-3, 3.7613599561E-3, 3.0809799209E-3,
                    4.1121998802E-3, 2.2212199401E-3, 5.5381999118E-4,
                    2.2212199401E-3, 4.1121998802E-3, 3.0809799209E-3,
                    3.7613599561E-3, 2.1605400834E-3, 1.4607800404E-3,
                    -5.8146001538E-4, -6.7714002216E-4],
                  [1.2078000145E-4, 4.4606000301E-4, -5.8146001538E-4,
                   5.6215998484E-4, -1.3688000035E-4, 2.3255399428E-3,
                   2.8898599558E-3, 4.2872801423E-3, 5.5893999524E-3,
                   4.2872801423E-3, 2.8898599558E-3, 2.3255399428E-3,
                   -1.3688000035E-4, 5.6215998484E-4, -5.8146001538E-4,
                   4.4606000301E-4, 1.2078000145E-4],
                  [-4.3500000174E-5, 1.2078000145E-4, -6.7714002216E-4,
                    -1.2434000382E-4, -8.0063997302E-4, -1.5970399836E-3,
                    -2.5168000138E-4, -4.2019999819E-4, 1.2619999470E-3,
                    -4.2019999819E-4, -2.5168000138E-4, -1.5970399836E-3,
                    -8.0063997302E-4, -1.2434000382E-4, -6.7714002216E-4,
                    1.2078000145E-4, -4.3500000174E-5]]))
    filters['bfilts'] = (
        numpy.array([[-8.1125000725E-4, 4.4451598078E-3, 1.2316980399E-2,
                    1.3955879956E-2,  1.4179450460E-2, 1.3955879956E-2,
                    1.2316980399E-2, 4.4451598078E-3, -8.1125000725E-4,
                    3.9103501476E-3, 4.4565401040E-3, -5.8724298142E-3,
                    -2.8760801069E-3, 8.5267601535E-3, -2.8760801069E-3,
                    -5.8724298142E-3, 4.4565401040E-3, 3.9103501476E-3,
                    1.3462699717E-3, -3.7740699481E-3, 8.2581602037E-3,
                    3.9442278445E-2, 5.3605638444E-2, 3.9442278445E-2,
                    8.2581602037E-3, -3.7740699481E-3, 1.3462699717E-3,
                    7.4700999539E-4, -3.6522001028E-4, -2.2522680461E-2,
                    -0.1105690673, -0.1768419296, -0.1105690673,
                    -2.2522680461E-2, -3.6522001028E-4, 7.4700999539E-4,
                    0.0000000000, 0.0000000000, 0.0000000000,
                    0.0000000000, 0.0000000000, 0.0000000000,
                    0.0000000000, 0.0000000000, 0.0000000000,
                    -7.4700999539E-4, 3.6522001028E-4, 2.2522680461E-2,
                    0.1105690673, 0.1768419296, 0.1105690673,
                    2.2522680461E-2, 3.6522001028E-4, -7.4700999539E-4,
                    -1.3462699717E-3, 3.7740699481E-3, -8.2581602037E-3,
                    -3.9442278445E-2, -5.3605638444E-2, -3.9442278445E-2,
                    -8.2581602037E-3, 3.7740699481E-3, -1.3462699717E-3,
                    -3.9103501476E-3, -4.4565401040E-3, 5.8724298142E-3,
                    2.8760801069E-3, -8.5267601535E-3, 2.8760801069E-3,
                    5.8724298142E-3, -4.4565401040E-3, -3.9103501476E-3,
                    8.1125000725E-4, -4.4451598078E-3, -1.2316980399E-2,
                    -1.3955879956E-2, -1.4179450460E-2, -1.3955879956E-2,
                    -1.2316980399E-2, -4.4451598078E-3, 8.1125000725E-4],
                  [0.0000000000, -8.2846998703E-4, -5.7109999034E-5,
                   4.0110000555E-5, 4.6670897864E-3, 8.0871898681E-3,
                   1.4807609841E-2, 8.6204400286E-3, -3.1221499667E-3,
                   8.2846998703E-4, 0.0000000000, -9.7479997203E-4,
                   -6.9718998857E-3, -2.0865600090E-3, 2.3298799060E-3,
                   -4.4814897701E-3, 1.4917500317E-2, 8.6204400286E-3,
                   5.7109999034E-5, 9.7479997203E-4, 0.0000000000,
                   -1.2145539746E-2, -2.4427289143E-2, 5.0797060132E-2,
                   3.2785870135E-2, -4.4814897701E-3, 1.4807609841E-2,
                   -4.0110000555E-5, 6.9718998857E-3, 1.2145539746E-2,
                   0.0000000000, -0.1510555595, -8.2495503128E-2,
                   5.0797060132E-2, 2.3298799060E-3, 8.0871898681E-3,
                   -4.6670897864E-3, 2.0865600090E-3, 2.4427289143E-2,
                   0.1510555595, 0.0000000000, -0.1510555595,
                   -2.4427289143E-2, -2.0865600090E-3, 4.6670897864E-3,
                   -8.0871898681E-3, -2.3298799060E-3, -5.0797060132E-2,
                   8.2495503128E-2, 0.1510555595, 0.0000000000,
                   -1.2145539746E-2, -6.9718998857E-3, 4.0110000555E-5,
                   -1.4807609841E-2, 4.4814897701E-3, -3.2785870135E-2,
                   -5.0797060132E-2, 2.4427289143E-2, 1.2145539746E-2,
                   0.0000000000, -9.7479997203E-4, -5.7109999034E-5,
                   -8.6204400286E-3, -1.4917500317E-2, 4.4814897701E-3,
                   -2.3298799060E-3, 2.0865600090E-3, 6.9718998857E-3,
                   9.7479997203E-4, 0.0000000000, -8.2846998703E-4,
                   3.1221499667E-3, -8.6204400286E-3, -1.4807609841E-2,
                   -8.0871898681E-3, -4.6670897864E-3, -4.0110000555E-5,
                   5.7109999034E-5, 8.2846998703E-4, 0.0000000000],
                  [8.1125000725E-4, -3.9103501476E-3, -1.3462699717E-3,
                   -7.4700999539E-4, 0.0000000000, 7.4700999539E-4,
                   1.3462699717E-3, 3.9103501476E-3, -8.1125000725E-4,
                   -4.4451598078E-3, -4.4565401040E-3, 3.7740699481E-3,
                   3.6522001028E-4, 0.0000000000, -3.6522001028E-4,
                   -3.7740699481E-3, 4.4565401040E-3, 4.4451598078E-3,
                   -1.2316980399E-2, 5.8724298142E-3, -8.2581602037E-3,
                   2.2522680461E-2, 0.0000000000, -2.2522680461E-2,
                   8.2581602037E-3, -5.8724298142E-3, 1.2316980399E-2,
                   -1.3955879956E-2, 2.8760801069E-3, -3.9442278445E-2,
                   0.1105690673, 0.0000000000, -0.1105690673,
                   3.9442278445E-2, -2.8760801069E-3, 1.3955879956E-2,
                   -1.4179450460E-2, -8.5267601535E-3, -5.3605638444E-2,
                   0.1768419296, 0.0000000000, -0.1768419296,
                   5.3605638444E-2, 8.5267601535E-3, 1.4179450460E-2,
                   -1.3955879956E-2, 2.8760801069E-3, -3.9442278445E-2,
                   0.1105690673, 0.0000000000, -0.1105690673,
                   3.9442278445E-2, -2.8760801069E-3, 1.3955879956E-2,
                   -1.2316980399E-2, 5.8724298142E-3, -8.2581602037E-3,
                   2.2522680461E-2, 0.0000000000, -2.2522680461E-2,
                   8.2581602037E-3, -5.8724298142E-3, 1.2316980399E-2,
                   -4.4451598078E-3, -4.4565401040E-3, 3.7740699481E-3,
                   3.6522001028E-4, 0.0000000000, -3.6522001028E-4,
                   -3.7740699481E-3, 4.4565401040E-3, 4.4451598078E-3,
                   8.1125000725E-4, -3.9103501476E-3, -1.3462699717E-3,
                   -7.4700999539E-4, 0.0000000000, 7.4700999539E-4,
                   1.3462699717E-3, 3.9103501476E-3, -8.1125000725E-4],
                  [3.1221499667E-3, -8.6204400286E-3, -1.4807609841E-2,
                   -8.0871898681E-3, -4.6670897864E-3, -4.0110000555E-5,
                   5.7109999034E-5, 8.2846998703E-4, 0.0000000000,
                   -8.6204400286E-3, -1.4917500317E-2, 4.4814897701E-3,
                   -2.3298799060E-3, 2.0865600090E-3, 6.9718998857E-3,
                   9.7479997203E-4, -0.0000000000, -8.2846998703E-4,
                   -1.4807609841E-2, 4.4814897701E-3, -3.2785870135E-2,
                   -5.0797060132E-2, 2.4427289143E-2, 1.2145539746E-2,
                   0.0000000000, -9.7479997203E-4, -5.7109999034E-5,
                   -8.0871898681E-3, -2.3298799060E-3, -5.0797060132E-2,
                   8.2495503128E-2, 0.1510555595, -0.0000000000,
                   -1.2145539746E-2, -6.9718998857E-3, 4.0110000555E-5,
                   -4.6670897864E-3, 2.0865600090E-3, 2.4427289143E-2,
                   0.1510555595, 0.0000000000, -0.1510555595,
                   -2.4427289143E-2, -2.0865600090E-3, 4.6670897864E-3,
                   -4.0110000555E-5, 6.9718998857E-3, 1.2145539746E-2,
                   0.0000000000, -0.1510555595, -8.2495503128E-2,
                   5.0797060132E-2, 2.3298799060E-3, 8.0871898681E-3,
                   5.7109999034E-5, 9.7479997203E-4, -0.0000000000,
                   -1.2145539746E-2, -2.4427289143E-2, 5.0797060132E-2,
                   3.2785870135E-2, -4.4814897701E-3, 1.4807609841E-2,
                   8.2846998703E-4, -0.0000000000, -9.7479997203E-4,
                   -6.9718998857E-3, -2.0865600090E-3, 2.3298799060E-3,
                   -4.4814897701E-3, 1.4917500317E-2, 8.6204400286E-3,
                   0.0000000000, -8.2846998703E-4, -5.7109999034E-5,
                   4.0110000555E-5, 4.6670897864E-3, 8.0871898681E-3,
                   1.4807609841E-2, 8.6204400286E-3, -3.1221499667E-3]]).T)
    return filters

def sp5Filters():
    filters = {}
    filters['harmonics'] = numpy.array([1, 3, 5])
    filters['mtx'] = (
        numpy.array([[0.3333, 0.2887, 0.1667, 0.0000, -0.1667, -0.2887],
                  [0.0000, 0.1667, 0.2887, 0.3333, 0.2887, 0.1667],
                  [0.3333, -0.0000, -0.3333, -0.0000, 0.3333, -0.0000],
                  [0.0000, 0.3333, 0.0000, -0.3333, 0.0000, 0.3333],
                  [0.3333, -0.2887, 0.1667, -0.0000, -0.1667, 0.2887],
                  [-0.0000, 0.1667, -0.2887, 0.3333, -0.2887, 0.1667]]))
    filters['hi0filt'] = (
        numpy.array([[-0.00033429, -0.00113093, -0.00171484,
                    -0.00133542, -0.00080639, -0.00133542,
                    -0.00171484, -0.00113093, -0.00033429],
                  [-0.00113093, -0.00350017, -0.00243812,
                    0.00631653, 0.01261227, 0.00631653,
                    -0.00243812,-0.00350017, -0.00113093],
                  [-0.00171484, -0.00243812, -0.00290081,
                    -0.00673482, -0.00981051, -0.00673482,
                    -0.00290081, -0.00243812, -0.00171484],
                  [-0.00133542, 0.00631653, -0.00673482,
                    -0.07027679, -0.11435863, -0.07027679,
                    -0.00673482, 0.00631653, -0.00133542],
                  [-0.00080639, 0.01261227, -0.00981051,
                    -0.11435863, 0.81380200, -0.11435863,
                    -0.00981051, 0.01261227, -0.00080639],
                  [-0.00133542, 0.00631653, -0.00673482,
                    -0.07027679, -0.11435863, -0.07027679,
                    -0.00673482, 0.00631653, -0.00133542],
                  [-0.00171484, -0.00243812, -0.00290081,
                    -0.00673482, -0.00981051, -0.00673482,
                    -0.00290081, -0.00243812, -0.00171484],
                  [-0.00113093, -0.00350017, -0.00243812,
                    0.00631653, 0.01261227, 0.00631653,
                    -0.00243812, -0.00350017, -0.00113093],
                  [-0.00033429, -0.00113093, -0.00171484,
                    -0.00133542, -0.00080639, -0.00133542,
                    -0.00171484, -0.00113093, -0.00033429]]))
    filters['lo0filt'] = (
        numpy.array([[0.00341614, -0.01551246, -0.03848215, -0.01551246,
                  0.00341614],
                 [-0.01551246, 0.05586982, 0.15925570, 0.05586982,
                   -0.01551246],
                 [-0.03848215, 0.15925570, 0.40304148, 0.15925570,
                   -0.03848215],
                 [-0.01551246, 0.05586982, 0.15925570, 0.05586982,
                   -0.01551246],
                 [0.00341614, -0.01551246, -0.03848215, -0.01551246,
                  0.00341614]]))
    filters['lofilt'] = (
        2 * numpy.array([[0.00085404, -0.00244917, -0.00387812, -0.00944432,
                       -0.00962054, -0.00944432, -0.00387812, -0.00244917,
                       0.00085404],
                      [-0.00244917, -0.00523281, -0.00661117, 0.00410600,
                        0.01002988, 0.00410600, -0.00661117, -0.00523281,
                        -0.00244917],
                      [-0.00387812, -0.00661117, 0.01396746, 0.03277038,
                        0.03981393, 0.03277038, 0.01396746, -0.00661117,
                        -0.00387812],
                      [-0.00944432, 0.00410600, 0.03277038, 0.06426333,
                        0.08169618, 0.06426333, 0.03277038, 0.00410600,
                        -0.00944432],
                      [-0.00962054, 0.01002988, 0.03981393, 0.08169618,
                        0.10096540, 0.08169618, 0.03981393, 0.01002988,
                        -0.00962054],
                      [-0.00944432, 0.00410600, 0.03277038, 0.06426333,
                        0.08169618, 0.06426333, 0.03277038, 0.00410600,
                        -0.00944432],
                      [-0.00387812, -0.00661117, 0.01396746, 0.03277038,
                        0.03981393, 0.03277038, 0.01396746, -0.00661117,
                        -0.00387812],
                      [-0.00244917, -0.00523281, -0.00661117, 0.00410600,
                        0.01002988, 0.00410600, -0.00661117, -0.00523281,
                        -0.00244917],
                      [0.00085404, -0.00244917, -0.00387812, -0.00944432,
                       -0.00962054, -0.00944432, -0.00387812, -0.00244917,
                       0.00085404]]))
    filters['bfilts'] = (
        numpy.array([[0.00277643, 0.00496194, 0.01026699, 0.01455399, 0.01026699,
                   0.00496194, 0.00277643, -0.00986904, -0.00893064, 
                   0.01189859, 0.02755155, 0.01189859, -0.00893064,
                   -0.00986904, -0.01021852, -0.03075356, -0.08226445,
                   -0.11732297, -0.08226445, -0.03075356, -0.01021852,
                   0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                   0.00000000, 0.00000000, 0.01021852, 0.03075356, 0.08226445,
                   0.11732297, 0.08226445, 0.03075356, 0.01021852, 0.00986904,
                   0.00893064, -0.01189859, -0.02755155, -0.01189859, 
                   0.00893064, 0.00986904, -0.00277643, -0.00496194,
                   -0.01026699, -0.01455399, -0.01026699, -0.00496194,
                   -0.00277643],
                  [-0.00343249, -0.00640815, -0.00073141, 0.01124321,
                    0.00182078, 0.00285723, 0.01166982, -0.00358461,
                    -0.01977507, -0.04084211, -0.00228219, 0.03930573,
                    0.01161195, 0.00128000, 0.01047717, 0.01486305,
                    -0.04819057, -0.12227230, -0.05394139, 0.00853965,
                    -0.00459034, 0.00790407, 0.04435647, 0.09454202,
                    -0.00000000, -0.09454202, -0.04435647, -0.00790407,
                    0.00459034, -0.00853965, 0.05394139, 0.12227230,
                    0.04819057, -0.01486305, -0.01047717, -0.00128000,
                    -0.01161195, -0.03930573, 0.00228219, 0.04084211,
                    0.01977507, 0.00358461, -0.01166982, -0.00285723,
                    -0.00182078, -0.01124321, 0.00073141, 0.00640815,
                    0.00343249],
                  [0.00343249, 0.00358461, -0.01047717, -0.00790407,
                   -0.00459034, 0.00128000, 0.01166982, 0.00640815,
                   0.01977507, -0.01486305, -0.04435647, 0.00853965,
                   0.01161195, 0.00285723, 0.00073141, 0.04084211, 0.04819057,
                   -0.09454202, -0.05394139, 0.03930573, 0.00182078,
                   -0.01124321, 0.00228219, 0.12227230, -0.00000000,
                   -0.12227230, -0.00228219, 0.01124321, -0.00182078,
                   -0.03930573, 0.05394139, 0.09454202, -0.04819057,
                   -0.04084211, -0.00073141, -0.00285723, -0.01161195,
                   -0.00853965, 0.04435647, 0.01486305, -0.01977507,
                   -0.00640815, -0.01166982, -0.00128000, 0.00459034,
                   0.00790407, 0.01047717, -0.00358461, -0.00343249],
                  [-0.00277643, 0.00986904, 0.01021852, -0.00000000,
                    -0.01021852, -0.00986904, 0.00277643, -0.00496194,
                    0.00893064, 0.03075356, -0.00000000, -0.03075356,
                    -0.00893064, 0.00496194, -0.01026699, -0.01189859,
                    0.08226445, -0.00000000, -0.08226445, 0.01189859,
                    0.01026699, -0.01455399, -0.02755155, 0.11732297,
                    -0.00000000, -0.11732297, 0.02755155, 0.01455399,
                    -0.01026699, -0.01189859, 0.08226445, -0.00000000,
                    -0.08226445, 0.01189859, 0.01026699, -0.00496194,
                    0.00893064, 0.03075356, -0.00000000, -0.03075356,
                    -0.00893064, 0.00496194, -0.00277643, 0.00986904,
                    0.01021852, -0.00000000, -0.01021852, -0.00986904,
                    0.00277643],
                  [-0.01166982, -0.00128000, 0.00459034, 0.00790407,
                    0.01047717, -0.00358461, -0.00343249, -0.00285723,
                    -0.01161195, -0.00853965, 0.04435647, 0.01486305,
                    -0.01977507, -0.00640815, -0.00182078, -0.03930573,
                    0.05394139, 0.09454202, -0.04819057, -0.04084211,
                    -0.00073141, -0.01124321, 0.00228219, 0.12227230,
                    -0.00000000, -0.12227230, -0.00228219, 0.01124321,
                    0.00073141, 0.04084211, 0.04819057, -0.09454202,
                    -0.05394139, 0.03930573, 0.00182078, 0.00640815,
                    0.01977507, -0.01486305, -0.04435647, 0.00853965,
                    0.01161195, 0.00285723, 0.00343249, 0.00358461,
                    -0.01047717, -0.00790407, -0.00459034, 0.00128000,
                    0.01166982],
                  [-0.01166982, -0.00285723, -0.00182078, -0.01124321,
                    0.00073141, 0.00640815, 0.00343249, -0.00128000,
                    -0.01161195, -0.03930573, 0.00228219, 0.04084211,
                    0.01977507, 0.00358461, 0.00459034, -0.00853965,
                    0.05394139, 0.12227230, 0.04819057, -0.01486305,
                    -0.01047717, 0.00790407, 0.04435647, 0.09454202,
                    -0.00000000, -0.09454202, -0.04435647, -0.00790407,
                    0.01047717, 0.01486305, -0.04819057, -0.12227230,
                    -0.05394139, 0.00853965, -0.00459034, -0.00358461,
                    -0.01977507, -0.04084211, -0.00228219, 0.03930573,
                    0.01161195, 0.00128000, -0.00343249, -0.00640815,
                    -0.00073141, 0.01124321, 0.00182078, 0.00285723,
                    0.01166982]]).T) 
    return filters

# convert level and band to dictionary index
def LB2idx(lev,band,nlevs,nbands):
    # reset band to match matlab version
    band += (nbands-1)
    if band > nbands-1:
        band = band - nbands

    if lev == 0:
        idx = 0
    elif lev == nlevs-1:
        # (Nlevels - ends)*Nbands + ends -1 (because zero indexed)
        idx = (((nlevs-2)*nbands)+2)-1
    else:
        # (level-first level) * nbands + first level + current band 
        idx = (nbands*lev)-band - 1

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
        origin = ( numpy.ceil(sz[0]/2.0), numpy.ceil(sz[1]/2.0) )

    if(len(args) > 2):
        amplitude = args[2]
    else:
        amplitude = 1

    res = numpy.zeros(sz);
    res[origin[0], origin[1]] = amplitude

    return res

# Compute a steering matrix (maps a directional basis set onto the
# angular Fourier harmonics).  HARMONICS is a vector specifying the
# angular harmonics contained in the steerable basis/filters.  ANGLES 
# (optional) is a vector specifying the angular position of each filter.  
# REL_PHASES (optional, default = 'even') specifies whether the harmonics 
# are cosine or sine phase aligned about those positions.
# The result matrix is suitable for passing to the function STEER.
# mtx = steer2HarmMtx(harmonics, angles, evenorodd)
def steer2HarmMtx(*args):

    if len(args) == 0:
        print "Error: first parameter 'harmonics' is required."
        return
    
    if len(args) > 0:
        harmonics = numpy.array(args[0])

    # optional parameters
    numh = (2*harmonics.shape[0]) - (harmonics == 0).sum()
    if len(args) > 1:
        angles = args[1]
    else:
        angles = numpy.pi * numpy.array(range(numh)) / numh
        
    if len(args) > 2:
        if isinstance(args[2], basestring):
            if args[2] == 'even' or args[2] == 'EVEN':
                evenorodd = 0
            elif args[2] == 'odd' or args[2] == 'ODD':
                evenorodd = 1
            else:
                print "Error: only 'even' and 'odd' are valid entries for the third input parameter."
                return
        else:
            print "Error: third input parameter must be a string (even/odd)."
    else:
        evenorodd = 0

    # Compute inverse matrix, which maps to Fourier components onto 
    #   steerable basis
    imtx = numpy.zeros((angles.shape[0], numh))
    col = 0
    for h in harmonics:
        args = h * angles
        if h == 0:
            imtx[:, col] = numpy.ones(angles.shape)
            col += 1
        elif evenorodd:
            imtx[:, col] = numpy.sin(args)
            imtx[:, col+1] = numpy.negative( numpy.cos(args) )
            col += 2
        else:
            imtx[:, col] = numpy.cos(args)
            imtx[:, col+1] = numpy.sin(args)
            col += 2

    r = numpy.rank(imtx)
    if r != numh and r != angles.shape[0]:
        print "Warning: matrix is not full rank"

    mtx = numpy.linalg.pinv(imtx)
    
    return mtx

# [X, Y] = rcosFn(WIDTH, POSITION, VALUES)
#
# Return a lookup table (suitable for use by INTERP1) 
# containing a "raised cosine" soft threshold function:
# 
#    Y =  VALUES(1) + (VALUES(2)-VALUES(1)) *
#              cos^2( PI/2 * (X - POSITION + WIDTH)/WIDTH )
#
# WIDTH is the width of the region over which the transition occurs
# (default = 1). POSITION is the location of the center of the
# threshold (default = 0).  VALUES (default = [0,1]) specifies the
# values to the left and right of the transition.
def rcosFn(*args):
    
    if len(args) > 0:
        width = args[0]
    else:
        width = 1

    if len(args) > 1:
        position = args[1]
    else:
        position = 0

    if len(args) > 2:
        values = args[2]
    else:
        values = (0,1)

    #---------------------------------------------

    sz = 256   # arbitrary!

    X = numpy.pi * numpy.array(range(-sz-1,2)) / (2*sz)

    Y = values[0] + (values[1]-values[0]) * numpy.cos(X)**2;

    # make sure end values are repeated, for extrapolation...
    Y[0] = Y[1]
    Y[sz+2] = Y[sz+1]
    
    X = position + (2*width/numpy.pi) * (X + numpy.pi/4)

    return (X,Y)

# Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
# containing samples of the polar angle (in radians, CW from the
# X-axis, ranging from -pi to pi), relative to angle PHASE (default =
# 0), about ORIGIN pixel (default = (size+1)/2).
def mkAngle(*args):

    if len(args) > 0:
        sz = args[0]
        if not isinstance(sz, tuple):
            sz = (sz, sz)
    else:
        print "Error: first input parameter 'size' is required!"
        print "makeAngle(size, phase, origin)"
        return

    # ------------------------------------------------------------
    # Optional args:

    if len(args) > 1:
        phase = args[1]
    else:
        phase = 'not set'

    if len(args) > 2:
        origin = args[2]
    else:
        origin = (sz[0]+1/2, sz[1]+1/2)

    #------------------------------------------------------------------

    (xramp, yramp) = numpy.meshgrid(numpy.array(range(1,sz[1]+1))-origin[1], 
                                 (numpy.array(range(1,sz[0]+1)))-origin[0])
    xramp = numpy.array(xramp)
    yramp = numpy.array(yramp)

    res = numpy.arctan2(yramp, xramp)
    
    if phase != 'not set':
        res = ((res+(numpy.pi-phase)) % (2*numpy.pi)) - numpy.pi

    return res

# [HFILT] = modulateFlipShift(LFILT)
# QMF/Wavelet highpass filter construction: modulate by (-1)^n,
# reverse order (and shift by one, which is handled by the convolution
# routines).  This is an extension of the original definition of QMF's
# (e.g., see Simoncelli90).
def modulateFlip(*args):

    if len(args) == 0:
        print "Error: filter input parameter required."
        return

    lfilt = args[0]
    # reshape lfilt to column vector
    if len(lfilt.shape) == 1:
        lfilt = lfilt.reshape(len(lfilt), 1)
    elif lfilt.shape[0] == 1:
        lfilt = lfilt.reshape(lfilt.shape[1], 1)
    elif len(lfilt.shape) > 2 or lfilt.shape[1] != 1:
        print 'Error: only 1D input supported.'
        return

    sz = len(lfilt)
    sz2 = numpy.ceil(sz/2.0);

    ind = numpy.array(range(sz-1,-1,-1))

    hfilt = lfilt[ind].T * (-1)**((ind+1)-sz2)

    # matlab version always returns a column vector
    if len(hfilt.shape) == 1:
        hfilt = hfilt.reshape(len(hfilt), 1)
    elif hfilt.shape[0] == 1:
        hfilt = hfilt.reshape(hfilt.shape[1], 1)

    return hfilt

# RES = blurDn(IM, LEVELS, FILT)
# Blur and downsample an image.  The blurring is done with filter
# kernel specified by FILT (default = 'binom5'), which can be a string
# (to be passed to namedFilter), a vector (applied separably as a 1D
# convolution kernel in X and Y), or a matrix (applied as a 2D
# convolution kernel).  The downsampling is always by 2 in each
# direction.
# The procedure is applied recursively LEVELS times (default=1).
# Eero Simoncelli, 3/97.  Ported to python by Rob Young 4/14
# function res = blurDn(im, nlevs, filt)
def blurDn(*args):
    if len(args) == 0:
        print "Error: image input parameter required."
        return

    im = numpy.array(args[0])
    
    # optional args
    if len(args) > 1:
        nlevs = args[1]
    else:
        nlevs = 1

    if len(args) > 2:
        filt = args[2]
        if isinstance(filt, basestring):
            filt = namedFilter(filt)
    else:
        filt = namedFilter('binom5')

    if filt.shape[0] == 1 or filt.shape[1] == 1:
        filt = [x/sum(filt) for x in filt]
    else:
        filt = [x/sum(sum(filt)) for x in filt]

    filt = numpy.array(filt)
    
    if nlevs > 1:
        im = blurDn(im, nlevs-1, filt)

    if nlevs >= 1:
        if len(im.shape) == 1 or im.shape[0] == 1 or im.shape[1] == 1:
            # 1D image
            if len(filt.shape) > 1 and (filt.shape[1]!=1 and filt.shape[2]!=1):
                # >1D filter
                print 'Error: Cannot apply 2D filter to 1D signal'
                return
            # orient filter and image correctly
            if im.shape[0] == 1:
                if len(filt.shape) == 1 or filt.shape[1] == 1:
                    filt = filt.T
            else:
                if filt.shape[0] == 1:
                    filt = filt.T
                
            res = corrDn(image = im, filt = filt, step = (2, 2))
            if len(im.shape) == 1 or im.shape[1] == 1:
                res = numpy.reshape(res, (numpy.ceil(im.shape[0]/2.0), 1))
            else:
                res = numpy.reshape(res, (1, numpy.ceil(im.shape[1]/2.0)))
        elif len(filt.shape) == 1 or filt.shape[0] == 1 or filt.shape[1] == 1:
            # 2D image and 1D filter
            res = corrDn(image = im, filt = filt.T, step = (2, 1))
            res = corrDn(image = res, filt = filt, step = (1, 2))

        else:  # 2D image and 2D filter
            res = corrDn(image = im, filt = filt, step = (2,2))
    else:
        res = im
            
    return res

def blur(*args):
    # RES = blur(IM, LEVELS, FILT)
    #
    # Blur an image, by filtering and downsampling LEVELS times
    # (default=1), followed by upsampling and filtering LEVELS times.  The
    # blurring is done with filter kernel specified by FILT (default =
    # 'binom5'), which can be a string (to be passed to namedFilter), a
    # vector (applied separably as a 1D convolution kernel in X and Y), or
    # a matrix (applied as a 2D convolution kernel).  The downsampling is
    # always by 2 in each direction.
    #
    # Eero Simoncelli, 3/04.  Python port by Rob Young, 10/15

    # REQUIRED ARG:
    if len(args) == 0:
        print "blur(IM, LEVELS, FILT)"
        print "first argument is required"
        exit(1)
    else:
        im = numpy.array(args[0])

    # OPTIONAL ARGS:
    if len(args) > 1:
        nlevs = args[1]
    else:
        nlevs = 1

    if len(args) > 2:
        if isinstance(args[2], basestring):
            filt = namedFilter(args[2])
        else:
            filt = numpy.array(args[2])
    else:
        filt = namedFilter('binom5')

    #--------------------------------------------------------------------
    
    if len(filt.shape) == 1 or filt.shape[0] == 1 or filt.shape[1] == 1:
        filt = filt / sum(filt)
    else:
        filt = filt / sum(sum(filt))

    if nlevs > 0:
        if len(im.shape) == 1 or im.shape[0] == 1 or im.shape[1] == 1: 
            # 1D image
            if len(filt) == 2 and (numpy.asarray(filt.shape) != 1).any():
                print 'Error: can not apply 2D filter to 1D signal'
                return
            
            imIn = corrDn(im, filt, 'reflect1', len(im))
            out = blur(imIn, nlevs-1, filt)
            res = upconv(out, filt, 'reflect1', len(im), [0,0],
                         len(im))
            return res
        elif len(filt.shape) == 1 or filt.shape[0] == 1 or filt.shape[1] == 1:
            # 2D image 1D filter
            imIn = corrDn(im, filt, 'reflect1', [2,1])
            imIn = corrDn(imIn, filt.T, 'reflect1', [1,2])
            out = blur(imIn, nlevs-1, filt)
            res = upConv(out, filt.T, 'reflect1', [1,2], [0,0],
                         [out.shape[0], im.shape[1]])
            res = upConv(res, filt, 'reflect1', [2,1], [0,0],
                         im.shape)
            return res
        else:
            # 2D image 2D filter
            imIn = corrDn(im, filt, 'reflect1', [2,2])
            out = blur(imIn, nlevs-1, filt)
            res = upConv(out, filt, 'reflect1', [2,2], [0,0],
                         im.shape)
            return res
    else:
        return im
            

def rconv2(*args):
    # Convolution of two matrices, with boundaries handled via reflection
    # about the edge pixels.  Result will be of size of LARGER matrix.
    # 
    # The origin of the smaller matrix is assumed to be its center.
    # For even dimensions, the origin is determined by the CTR (optional) 
    # argument:
    #      CTR   origin
    #       0     DIM/2      (default)
    #       1   (DIM/2)+1  
    
    if len(args) < 2:
        print "Error: two matrices required as input parameters"
        return

    if len(args) == 2:
        ctr = 0

    if ( args[0].shape[0] >= args[1].shape[0] and 
         args[0].shape[1] >= args[1].shape[1] ):
        large = args[0]
        small = args[1]
    elif ( args[0].shape[0] <= args[1].shape[0] and 
           args[0].shape[1] <= args[1].shape[1] ):
        large = args[1]
        small = args[0]
    else:
        print 'one arg must be larger than the other in both dimensions!'
        return

    ly = large.shape[0]
    lx = large.shape[1]
    sy = small.shape[0]
    sx = small.shape[1]

    ## These values are one less than the index of the small mtx that falls on 
    ## the border pixel of the large matrix when computing the first 
    ## convolution response sample:
    sy2 = numpy.floor((sy+ctr-1)/2)
    sx2 = numpy.floor((sx+ctr-1)/2)

    # pad with reflected copies
    nw = large[sy-sy2-1:0:-1, sx-sx2-1:0:-1]
    n = large[sy-sy2-1:0:-1, :]
    ne = large[sy-sy2-1:0:-1, lx-2:lx-sx2-2:-1]
    w = large[:, sx-sx2-1:0:-1]
    e = large[:, lx-2:lx-sx2-2:-1]
    sw = large[ly-2:ly-sy2-2:-1, sx-sx2-1:0:-1]
    s = large[ly-2:ly-sy2-2:-1, :]
    se = large[ly-2:ly-sy2-2:-1, lx-2:lx-sx2-2:-1]

    n = numpy.column_stack((nw, n, ne))
    c = numpy.column_stack((w,large,e))
    s = numpy.column_stack((sw, s, se))

    clarge = numpy.concatenate((n, c), axis=0)
    clarge = numpy.concatenate((clarge, s), axis=0)
    
    return scipy.signal.convolve(clarge, small, 'valid')

# compute minimum and maximum values of input matrix, returning them as tuple
def range2(*args):
    if not numpy.isreal(args[0]).all():
        print 'Error: matrix must be real-valued'

    return (args[0].min(), args[0].max())

# Sample variance of a matrix.
#  Passing MEAN (optional) makes the calculation faster.
def var2(*args):
    if len(args) == 1:
        mn = args[0].mean()
    elif len(args) == 2:
        mn = args[1]
    
    if(numpy.isreal(args[0]).all()):
        res = sum(sum((args[0]-mn)**2)) / max(numpy.prod(args[0].shape)-1, 1)
    else:
        res = sum((args[0]-mn).real**2) + 1j*sum((args[0]-mn).imag)**2
        res = res /  max(numpy.prod(args[0].shape)-1, 1)

    return res

# Sample kurtosis (fourth moment divided by squared variance) 
# of a matrix.  Kurtosis of a Gaussian distribution is 3.
#  MEAN (optional) and VAR (optional) make the computation faster.
def kurt2(*args):
    if len(args) == 0:
        print 'Error: input matrix is required'

    if len(args) < 2:
        mn = args[0].mean()
    else:
        mn = args[1]

    if len(args) < 3:
        v = var2(args[0])
    else:
        v = args[2]

    if numpy.isreal(args[0]).all():
        res = (numpy.abs(args[0]-mn)**4).mean() / v**2
    else:
        res = ( (((args[0]-mn).real**4).mean() / v.real**2) + 
                ((numpy.i * (args[0]-mn).imag**4).mean() / v.imag**2) )

    return res

# Report image (matrix) statistics.
# When called on a single image IM1, report min, max, mean, stdev, 
# and kurtosis.
# When called on two images (IM1 and IM2), report min, max, mean, 
# stdev of the difference, and also SNR (relative to IM1).
def imStats(*args):

    if len(args) == 0:
        print 'Error: at least one input image is required'
        return
    elif len(args) == 1 and not numpy.isreal(args[0]).all():
        print 'Error: input images must be real-valued matrices'
        return
    elif len(args) == 2 and ( not numpy.isreal(args[0]).all() or not numpy.isreal(args[1]).all()):
        print 'Error: input images must be real-valued matrices'
        return
    elif len(args) > 2:
        print 'Error: maximum of two input images allowed'
        return

    if len(args) == 2:
        difference = args[0] - args[1]
        (mn, mx) = range2(difference)
        mean = difference.mean()
        v = var2(difference)
        if v < numpy.finfo(numpy.double).tiny:
            snr = numpy.inf
        else:
            snr = 10 * numpy.log10(var2(args[0])/v)
        print 'Difference statistics:'
        print '  Range: [%d, %d]' % (mn, mx)
        print '  Mean: %f,  Stdev (rmse): %f,  SNR (dB): %f' % (mean, numpy.sqrt(v), snr)
    else:
        (mn, mx) = range2(args[0])
        mean = args[0].mean()
        var = var2(args[0])
        stdev = numpy.sqrt(var.real) + numpy.sqrt(var.imag)
        kurt = kurt2(args[0], mean, stdev**2)
        print 'Image statistics:'
        print '  Range: [%f, %f]' % (mn, mx)
        print '  Mean: %f,  Stdev: %f,  Kurtosis: %f' % (mean, stdev, kurt)
        
# makes image the same as read in by matlab
def correctImage(img):
    #tmpcol = img[:,0]
    #for i in range(img.shape[1]-1):
    #    img[:,i] = img[:,i+1]
    #img[:, img.shape[1]-1] = tmpcol
    #return img
    return numpy.roll(img, -1)

# Circular shift 2D matrix samples by OFFSET (a [Y,X] 2-tuple),
# such that  RES(POS) = MTX(POS-OFFSET).
def shift(mtx, offset):
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

# Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
# containing samples of a radial ramp function, raised to power EXPT
# (default = 1), with given ORIGIN (default = (size+1)/2, [1 1] =
# upper left).  All but the first argument are optional.
# Eero Simoncelli, 6/96.  Ported to Python by Rob Young, 5/14.
def mkR(*args):
    if len(args) == 0:
        print 'Error: first input parameter is required!'
        return
    else:
        sz = args[0]

    if isinstance(sz, (int, long)) or len(sz) == 1:
        sz = (sz, sz)

    # -----------------------------------------------------------------
    # OPTIONAL args:

    if len(args) < 2:
        expt = 1;
    else:
        expt = args[1]

    if len(args) < 3:
        origin = ((sz[0]+1)/2.0, (sz[1]+1)/2.0)
    else:
        origin = args[2]

    # -----------------------------------------------------------------

    (xramp2, yramp2) = numpy.meshgrid(numpy.array(range(1,sz[1]+1))-origin[1], 
                                   numpy.array(range(1,sz[0]+1))-origin[0])

    res = (xramp2**2 + yramp2**2)**(expt/2.0)
    
    return res

# Make a matrix of dimensions SIZE (a [Y X] 2-vector, or a scalar)
# containing fractal (pink) noise with power spectral density of the
# form: 1/f^(5-2*FRACT_DIM).  Image variance is normalized to 1.0.
# FRACT_DIM defaults to 1.0
# Eero Simoncelli, 6/96. Ported to Python by Rob Young, 5/14.

# TODO: Verify that this  matches Mandelbrot defn of fractal dimension.
#       Make this more efficient!

def mkFract(*args):
    if len(args) == 0:
        print 'Error: input parameter dims required'
    else:
        if isinstance(args[0], (int, long)) or len(args[0]) == 1:
            dims = (args[0], args[0])
        elif args[0] == 1:
            dims = (args[1], args[1])
        elif args[1] == 1:
            dims = (args[0], args[0])
        else:
            dims = args[0]

    if len(args) < 2:
        fract_dim = 1.0
    else:
        fract_dim = args[1]

    res = numpy.random.randn(dims[0], dims[1])
    fres = numpy.fft.fft2(res)

    sz = res.shape
    ctr = (int(numpy.ceil((sz[0]+1)/2.0)), int(numpy.ceil((sz[1]+1)/2.0)))

    sh = numpy.fft.ifftshift(mkR(sz, -(2.5-fract_dim), ctr))
    sh[0,0] = 1;  #DC term

    fres = sh * fres
    fres = numpy.fft.ifft2(fres)

    #if any(max(max(abs(fres.imag))) > 1e-10):
    if abs(fres.imag).max() > 1e-10:
        print 'Symmetry error in creating fractal'
    else:
        res = numpy.real(fres)
        res = res / numpy.sqrt(var2(res))

    return res

# Steer BASIS to the specfied ANGLE.  
#function res = steer(basis,angle,harmonics,steermtx)
# 
# BASIS should be a matrix whose columns are vectorized rotated copies of a 
# steerable function, or the responses of a set of steerable filters.
# 
# ANGLE can be a scalar, or a column vector the size of the basis.
# 
# HARMONICS (optional, default is N even or odd low frequencies, as for 
# derivative filters) should be a list of harmonic numbers indicating
# the angular harmonic content of the basis.
# 
# STEERMTX (optional, default assumes cosine phase harmonic components,
# and filter positions at 2pi*n/N) should be a matrix which maps
# the filters onto Fourier series components (ordered [cos0 cos1 sin1 
# cos2 sin2 ... sinN]).  See steer2HarmMtx.m
#
# Eero Simoncelli, 7/96. Ported to Python by Rob Young, 5/14.
def steer(*args):
    
    if len(args) < 2:
        print 'Error: input parameters basis and angle are required!'
        return

    basis = args[0]

    num = basis.shape[1]

    #if ( any(size(angle) ~= [size(basis,1) 1]) & any(size(angle) ~= [1 1]) )
    angle = args[1]
    if isinstance(angle, (int, long, float)):
        angle = numpy.array([angle])
    else:
        if angle.shape[0] != basis.shape[0] or angle.shape[1] != 1:
            print 'ANGLE must be a scalar, or a column vector the size of the basis elements'
            return

    # If HARMONICS are not passed, assume derivatives.
    if len(args) < 3:
        if num%2 == 0:
            harmonics = numpy.array(range(num/2))*2+1
        else:
            harmonics = numpy.array(range((15+1)/2))*2
    else:
        harmonics = args[2]
    if len(harmonics.shape) == 1 or harmonics.shape[0] == 1:
        harmonics = harmonics.reshape(harmonics.shape[0], 1)
    elif harmonics.shape[0] != 1 and harmonics.shape[1] != 1:
        print 'Error: input parameter HARMONICS must be 1D!'
        return

    if 2*harmonics.shape[0] - (harmonics == 0).sum() != num:
        print 'harmonics list is incompatible with basis size!'
        return

    # If STEERMTX not passed, assume evenly distributed cosine-phase filters:
    if len(args) < 4:
        steermtx = steer2HarmMtx(harmonics, numpy.pi*numpy.array(range(num))/num, 
                                 'even')
    else:
        steermtx = args[3]

    steervect = numpy.zeros((angle.shape[0], num))
    arg = angle * harmonics[numpy.nonzero(harmonics)[0]].T
    if all(harmonics):
	steervect[:, range(0,num,2)] = numpy.cos(arg)
	steervect[:, range(1,num,2)] = numpy.sin(arg)
    else:
	steervect[:, 1] = numpy.ones((arg.shape[0],1))
	steervect[:, range(0,num,2)] = numpy.cos(arg)
	steervect[:, range(1,num,2)] = numpy.sin(arg)

    steervect = numpy.dot(steervect,steermtx)

    if steervect.shape[0] > 1:
	tmp = numpy.dot(basis, steervect)
	res = sum(tmp).T
    else:
	res = numpy.dot(basis, steervect.T)

    return res

def showIm_old(*args):
    # check and set input parameters
    if len(args) == 0:
        print "showIm( matrix, range, zoom, label, nshades )"
        print "  matrix is string. It should be the name of a 2D array."
        print "  range is a two element tuple.  It specifies the values that "
        print "    map to the min and max colormap values.  Passing a value "
        print "    of 'auto' (default) sets range=[min,max].  'auto2' sets "
        print "    range=[mean-2*stdev, mean+2*stdev].  'auto3' sets "
        print "    range=[p1-(p2-p1)/8, p2+(p2-p1)/8], where p1 is the 10th "
        print "    percientile value of the sorted matix samples, and p2 is "
        print "    the 90th percentile value."
        print "  zoom specifies the number of matrix samples per screen pixel."
        print "    It will be rounded to an integer, or 1 divided by an "
        print "    integer."
        #print "    A value of 'same' or 'auto' (default) causes the "
        #print "    zoom value to be chosen automatically to fit the image into"
        #print "    the current axes."
        #print "    A value of 'full' fills the axis region "
        #print "    (leaving no room for labels)."
        print "  label - A string that is used as a figure title."
        print "  NSHADES (optional) specifies the number of gray shades, "
        print "    and defaults to the size of the current colormap. "

    if len(args) > 0:   # matrix entered
        matrix = numpy.array(args[0])
    #print 'showIm range %f %f' % (matrix.min(), matrix.max())

    if len(args) > 1:   # range entered
        if isinstance(args[1], basestring):
            if args[1] is "auto":
                imRange = ( numpy.amin(matrix), numpy.amax(matrix) )
            elif args[1] is "auto2":
                imRange = ( matrix.mean()-2*matrix.std(), 
                            matrix.mean()+2*matrix.std() )
            elif args[1] is "auto3":
                #p1 = numpy.percentile(matrix, 10)  not in python 2.6.6?!
                #p2 = numpy.percentile(matrix, 90)
                p1 = scipy.stats.scoreatpercentile(numpy.hstack(matrix), 10)
                p2 = scipy.stats.scoreatpercentile(numpy.hstack(matrix), 90)
                imRange = (p1-(p2-p1)/8.0, p2+(p2-p1)/8.0)
            else:
                print "Error: range of %s is not recognized." % args[1]
                print "       please use a two element tuple or "
                print "       'auto', 'auto2' or 'auto3'"
                print "       enter 'showIm' for more info about options"
                return
        else:
            imRange = args[1][0], args[1][1]
    else:
        imRange = ( numpy.amin(matrix), numpy.amax(matrix) )
    
    if len(args) > 2:   # zoom entered
        zoom = args[2]
    else:
        zoom = 1

    if len(args) > 3:   # label entered
        label = args[3]
    else:
        label = ''

    if len(args) > 4:   # colormap entered
        nshades = args[4]
    else:
        nshades = 256

    # create window
    app = QtGui.QApplication(sys.argv)
    window = QtGui.QMainWindow()
    window.setWindowTitle('showIm')

    # draw image in pixmap
    matrix = JBhelpers.rerange(matrix.astype(float), imRange[0], imRange[1])
    #matrix = numpy.require(matrix, numpy.uint8, 'C')
    # thanks to Johannes for the following two line fix!!
    data = numpy.empty( ( matrix.shape[ 0 ], 
                          ( matrix.shape[ 1 ] + 3 ) // 4 * 4 ), 
                        numpy.uint8 )
    data[ :, :matrix.shape[ 1 ] ] = matrix
    #(w, h) = matrix.shape
    (nRows, nCols) = matrix.shape
    matrix = data[:]
    #qim = QtGui.QImage(matrix, w, h, QtGui.QImage.Format_Indexed8)
    qim = QtGui.QImage(matrix, nCols, nRows, QtGui.QImage.Format_Indexed8)
    #qim.ndarray = matrix
    
    # make colormap
    incr = (256/nshades)+1
    colors = range(0,255,(256/nshades)+1)
    colors[-1] = 255
    colctr = -1
    for i in range(256):
        if i % incr == 0:
            colctr += 1
        qim.setColor(i, QtGui.QColor(colors[colctr], colors[colctr], 
                                     colors[colctr]).rgb())

    # zoom
    #dims = (matrix.shape[0]*zoom, matrix.shape[1]*zoom)
    dims = (nRows*zoom, nCols*zoom)
    print 'dims'
    print dims
    print 'nRows=%d nCols=%d' % (nRows, nCols)
    #qim = qim.scaled(dims[0], dims[1])
    qim = qim.scaled(nCols, nRows)
    #pixmap = QtGui.QPixmap()
    #pixmap = QtGui.QPixmap(dims[0], dims[1])
    #pixmap = QtGui.QPixmap(w,h)
    pixmap = QtGui.QPixmap(nCols, nRows)
    pixmap = QtGui.QPixmap.fromImage(qim)

    # set up widgets and layout
    mainWidget = QtGui.QWidget()
    layout = QtGui.QVBoxLayout(mainWidget)
    if len(label) > 0:
        tlab = QtGui.QLabel()
        tlab.setText(label)
        tlab.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(tlab)
    im = QtGui.QLabel()
    im.setPixmap(pixmap)
    layout.addWidget(im)
    rlab = QtGui.QLabel()
    rlab.setText('Range: [%.1f %.1f]' % (imRange[0], imRange[1]))
    rlab.setAlignment(QtCore.Qt.AlignCenter)
    layout.addWidget(rlab)
    dlab = QtGui.QLabel()
    #dlab.setText('Dims: [%d %d] * %.2f' % (w, h, zoom))
    dlab.setText('Dims: [%d %d] * %.2f' % (nRows, nCols, zoom))
    dlab.setAlignment(QtCore.Qt.AlignCenter)
    layout.addWidget(dlab)
    mainWidget.setLayout(layout)
    window.setCentralWidget(mainWidget)

    # display window and exit when requested
    window.show()
    #sys.exit(app.exec_())
    app.exec_()

def showIm(*args):
    # check and set input parameters
    if len(args) == 0:
        print "showIm( matrix, range, zoom, label, nshades )"
        print "  matrix is string. It should be the name of a 2D array."
        print "  range is a two element tuple.  It specifies the values that "
        print "    map to the min and max colormap values.  Passing a value "
        print "    of 'auto' (default) sets range=[min,max].  'auto2' sets "
        print "    range=[mean-2*stdev, mean+2*stdev].  'auto3' sets "
        print "    range=[p1-(p2-p1)/8, p2+(p2-p1)/8], where p1 is the 10th "
        print "    percientile value of the sorted matix samples, and p2 is "
        print "    the 90th percentile value."
        print "  zoom specifies the number of matrix samples per screen pixel."
        print "    It will be rounded to an integer, or 1 divided by an "
        print "    integer."
        #print "    A value of 'same' or 'auto' (default) causes the "
        #print "    zoom value to be chosen automatically to fit the image into"
        #print "    the current axes."
        #print "    A value of 'full' fills the axis region "
        #print "    (leaving no room for labels)."
        print "  label - A string that is used as a figure title."
        print "  NSHADES (optional) specifies the number of gray shades, "
        print "    and defaults to the size of the current colormap. "

    if len(args) > 0:   # matrix entered
        matrix = numpy.array(args[0])
    #print 'showIm range %f %f' % (matrix.min(), matrix.max())

    if len(args) > 1:   # range entered
        if isinstance(args[1], basestring):
            if args[1] is "auto":
                imRange = ( numpy.amin(matrix), numpy.amax(matrix) )
            elif args[1] is "auto2":
                imRange = ( matrix.mean()-2*matrix.std(), 
                            matrix.mean()+2*matrix.std() )
            elif args[1] is "auto3":
                #p1 = numpy.percentile(matrix, 10)  not in python 2.6.6?!
                #p2 = numpy.percentile(matrix, 90)
                p1 = scipy.stats.scoreatpercentile(numpy.hstack(matrix), 10)
                p2 = scipy.stats.scoreatpercentile(numpy.hstack(matrix), 90)
                imRange = (p1-(p2-p1)/8.0, p2+(p2-p1)/8.0)
            else:
                print "Error: range of %s is not recognized." % args[1]
                print "       please use a two element tuple or "
                print "       'auto', 'auto2' or 'auto3'"
                print "       enter 'showIm' for more info about options"
                return
        else:
            imRange = args[1][0], args[1][1]
    else:
        imRange = ( numpy.amin(matrix), numpy.amax(matrix) )
    
    if len(args) > 2:   # zoom entered
        zoom = args[2]
    else:
        zoom = 1

    if len(args) > 3:   # label entered
        label = args[3]
    else:
        label = ''

    if len(args) > 4:   # colormap entered
        nshades = args[4]
    else:
        nshades = 256

    # create window
    #master = Tkinter.Tk()
    master = Tkinter.Toplevel()
    master.title('showIm')
    canvas_width = matrix.shape[0] * zoom
    canvas_height = matrix.shape[1] * zoom
    master.geometry(str(canvas_width+20) + "x" + str(canvas_height+60) +
                    "+200+200")
    # put in top spacer
    spacer = Tkinter.Label(master, text='').pack()
    
    # create canvas
    canvas = Tkinter.Canvas(master, width=canvas_width, height=canvas_height)
    canvas.pack()
    #img = Image.fromarray(matrix)
    # FIX: shift matrix to 0.0-1.0 then to 0-255
    if (matrix < 0).any():
        matrix = matrix + math.fabs(matrix.min())
    matrix = (matrix / matrix.max()) * 255.0
    print matrix.astype('uint8')[0,:]
    img = PIL.Image.fromarray(matrix.astype('uint8'))

    # make colormap - works without range
    #colorTable = [0] * 256
    #incr = 256/(nshades-1)
    #colors = range(0, 255, incr)
    #colors += [255]
    #colctr = -1
    ## compute color transition indices
    #thresh = 255 / len(colors)
    #for i in range(256):
    #    # handle uneven color boundaries
    #    if thresh == 0 or (i % thresh == 0 and colctr < len(colors)-1):
    #        colctr += 1
    #    colorTable[i] = colors[colctr]
    #img = img.point(colorTable)

    # make colormap
    colorTable = [0] * 256
    #incr = int(numpy.ceil(float(imRange[1]-imRange[0]+1) / float(nshades)))
    incr = int(numpy.ceil(float(matrix.max()-matrix.min()+1) / float(nshades)))
    #colors = range(int(imRange[0]), int(imRange[1])+1, incr)
    colors = range(int(matrix.min()), int(matrix.max())+1, incr)
    colors[0] = 0
    colors[-1] = 255
    colctr = -1
    # compute color transition indices
    #thresh = round( (imRange[1]-imRange[0]) / len(colors) )
    thresh = round( (matrix.max() - matrix.min()) / len(colors) )
    for i in range(len(colorTable)):
        # handle uneven color boundaries
        if thresh == 0 or (i % thresh == 0 and colctr < len(colors)-1):
            colctr += 1
        colorTable[i] = colors[colctr]
    img = img.point(colorTable)

    # zoom
    if zoom != 1:
        img = img.resize((canvas_width, canvas_height), Image.NEAREST)

    # apply image to canvas
    imgPI = ImageTk.PhotoImage(img)    
    canvas.create_image(0,0, anchor=Tkinter.NW, image=imgPI)

    # add labels
    rangeStr = 'Range: [%.1f, %.1f]' % (imRange[0], imRange[1])
    rangeLabel = Tkinter.Label(master, text=rangeStr).pack()
    dimsStr = 'Dims: [%d, %d] / %d' % (matrix.shape[0], matrix.shape[1], zoom)
    dimsLabel = Tkinter.Label(master, text=dimsStr).pack()
    
    Tkinter.mainloop()

def corrDn(image = None, filt = None, edges = 'reflect1', step = (1,1), 
           start = (0,0), stop = None, result = None):

    if image == None or filt == None:
        print 'Error: image and filter are required input parameters!'
        return
    else:
        image = image.copy()
        filt = filt.copy()

    if len(filt.shape) == 1:
        filt = numpy.reshape(filt, (1,len(filt)))

    if stop == None:
        stop = (image.shape[0], image.shape[1])

    if result == None:
        rxsz = len(range(start[0], stop[0], step[0]))
        rysz = len(range(start[1], stop[1], step[1]))
        result = numpy.zeros((rxsz, rysz))
        
    if edges == 'circular':
        lib.internal_wrap_reduce(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                                 image.shape[1], image.shape[0], 
                                 filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                                 filt.shape[1], filt.shape[0], 
                                 start[1], step[1], stop[1], start[0], step[0], 
                                 stop[0], 
                                 result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    else:
        tmp = numpy.zeros((filt.shape[0], filt.shape[1]))
        lib.internal_reduce(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                            image.shape[1], image.shape[0], 
                            filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                            tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                            filt.shape[1], filt.shape[0], 
                            start[1], step[1], stop[1], start[0], step[0], 
                            stop[0], 
                            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                            edges)

    return result

def upConv(image = None, filt = None, edges = 'reflect1', step = (1,1), 
           start = (0,0), stop = None, result = None):

    if image == None or filt == None:
        print 'Error: image and filter are required input parameters!'
        return
    else:
        image = image.copy()
        filt = filt.copy()
        
    origShape = filt.shape
    if len(filt.shape) == 1:
        filt = numpy.reshape(filt, (1,len(filt)))

    if ( (edges != "reflect1" or edges != "extend" or edges != "repeat") and
         (filt.shape[0] % 2 == 0 or filt.shape[1] % 2 == 0) ):
        if filt.shape[1] == 1:
            filt = numpy.append(filt,0.0);
            filt = numpy.reshape(filt, (len(filt), 1))
        elif filt.shape[0] == 1:
            filt = numpy.append(filt,0.0);
            filt = numpy.reshape(filt, (1, len(filt)))
        else:
            print 'Even sized 2D filters not yet supported by upConv.'
            return

    if stop == None and result == None:
        stop = (image.shape[0]*step[0], image.shape[1]*step[1])
        stop = (stop[0], stop[1])
    elif stop == None:
        stop = (result.shape[0], result.shape[1])

    if result == None:
        result = numpy.zeros((stop[1], stop[0]))

    temp = numpy.zeros((filt.shape[1], filt.shape[0]))

    if edges == 'circular':
        lib.internal_wrap_expand(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 filt.shape[1], filt.shape[0], start[1], 
                                 step[1], stop[1], start[0], step[0], stop[0],
                                 result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 stop[1], stop[0])
        result = result.T
    else:
        lib.internal_expand(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            temp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            filt.shape[1], filt.shape[0], start[1], step[1],
                            stop[1], start[0], step[0], stop[0],
                            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            stop[1], stop[0], edges)
        result = numpy.reshape(result, stop)

    return result

def pointOp(image, lut, origin, increment, warnings):
    result = numpy.zeros((image.shape[0], image.shape[1]))

    lib.internal_pointop(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                         result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         image.shape[0] * image.shape[1], 
                         lut.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         lut.shape[0], 
                         ctypes.c_double(origin), 
                         ctypes.c_double(increment), warnings)

    return result

def cconv2(*args):
    # RES = CCONV2(MTX1, MTX2, CTR)
    #
    # Circular convolution of two matrices.  Result will be of size of
    # LARGER vector.
    # 
    # The origin of the smaller matrix is assumed to be its center.
    # For even dimensions, the origin is determined by the CTR (optional) 
    # argument:
    #      CTR   origin
    #       0     DIM/2      (default)
    #       1     (DIM/2)+1  
    #
    # Eero Simoncelli, 6/96.  Modified 2/97.  Python port by Rob Young, 8/15
    
    if len(args) < 2:
        print 'Error: cconv2 requires two input matrices!'
        print 'Usage: cconv2(matrix1, matrix2, center)'
        print 'where center parameter is optional'
        return
    else:
        a = numpy.array(args[0])
        b = numpy.array(args[1])

    if len(args) == 3:
        ctr = args[2]
    else:
        ctr = 0

    if a.shape[0] >= b.shape[0] and a.shape[1] >= b.shape[1]:
        large = a
        small = b
    elif a.shape[0] <= b.shape[0] and a.shape[1] <= b.shape[1]:
        large = b
        small = a
    else:
        print 'Error: one matrix must be larger than the other in both dimensions!'
        return
    
    ly = large.shape[0]
    lx = large.shape[1]
    sy = small.shape[0]
    sx = small.shape[1]

    ## These values are the index of the small mtx that falls on the
    ## border pixel of the large matrix when computing the first
    ## convolution response sample:
    sy2 = numpy.floor((sy+ctr+1)/2.0)
    sx2 = numpy.floor((sx+ctr+1)/2.0)

    # pad
    nw = large[ly-sy+sy2:ly, lx-sx+sx2:lx]
    n = large[ly-sy+sy2:ly, :]
    ne = large[ly-sy+sy2:ly, :sx2-1]
    w = large[:, lx-sx+sx2:lx]
    c = large
    e = large[:, :sx2-1]
    sw = large[:sy2-1, lx-sx+sx2:lx]
    s = large[:sy2-1, :]
    se = large[:sy2-1, :sx2-1]

    n = numpy.column_stack((nw, n, ne))
    c = numpy.column_stack((w,large,e))
    s = numpy.column_stack((sw, s, se))

    clarge = numpy.concatenate((n, c), axis=0)
    clarge = numpy.concatenate((clarge, s), axis=0)

    c = scipy.signal.convolve(clarge, small, 'valid')

    return c

def clip(*args):
    # [RES] = clip(IM, MINVALorRANGE, MAXVAL)
    #
    # Clip values of matrix IM to lie between minVal and maxVal:
    #      RES = max(min(IM,MAXVAL),MINVAL)
    # The first argument can also specify both min and max, as a 2-vector.
    # If only one argument is passed, the range defaults to [0,1].
    # ported to Python by Rob Young, 8/15
    
    if len(args) == 0 or len(args) > 3:
        print 'Usage: clip(im, minVal or Range, maxVal)'
        print 'first input parameter is required'
        return
        
    im = numpy.array(args[0])

    if len(args) == 1:
        minVal = 0;
        maxVal = 1;
    elif len(args) == 2:
        if isinstance(args[1], (int, long, float)):
            minVal = args[1]
            maxVal = args[1]+1
        else:
            minVal = args[1][0]
            maxVal = args[1][1]
    elif len(args) == 3:
        minVal = args[1]
        maxVal = args[2]
        
    if maxVal < minVal:
        print 'Error: maxVal cannot be less than minVal!'
        return

    im[numpy.where(im < minVal)] = minVal
    im[numpy.where(im > maxVal)] = maxVal

    return im

# round equivalent to matlab function
# used in histo so we can unit test against matlab code
# numpy version rounds to closest even number to remove bias
def round(arr):
    if isinstance(arr, (int, float, long)):
        arr = roundVal(arr)
    else:
        for i in range(len(arr)):
            arr[i] = roundVal(arr[i])
    return arr

def roundVal(val):
    (fracPart, intPart) = math.modf(val)
    if numpy.abs(fracPart) >= 0.5:
        if intPart >= 0:
            intPart += 1
        else:
            intPart -= 1
    return intPart

def histo(*args):
    # [N,X] = histo(MTX, nbinsOrBinsize, binCenter);
    #
    # Compute a histogram of (all) elements of MTX.  N contains the histogram
    # counts, X is a vector containg the centers of the histogram bins.
    #
    # nbinsOrBinsize (optional, default = 101) specifies either
    # the number of histogram bins, or the negative of the binsize.
    #
    # binCenter (optional, default = mean2(MTX)) specifies a center position
    # for (any one of) the histogram bins.
    #
    # How does this differ from MatLab's HIST function?  This function:
    #   - allows uniformly spaced bins only.
    #   +/- operates on all elements of MTX, instead of columnwise.
    #   + is much faster (approximately a factor of 80 on my machine).
    #   + allows specification of number of bins OR binsize.  Default=101 bins.
    #   + allows (optional) specification of binCenter.
    #
    # Eero Simoncelli, 3/97.  ported to Python by Rob Young, 8/15.
    #
    # NOTE: a C version of this code in available in this directory 
    #       called histo.c

    if len(args) == 0 or len(args) > 3:
        print 'Usage: histo(mtx, nbins, binCtr)'
        print 'first argument is required'
        return
    else:
        mtx = args[0]
    mtx = numpy.array(mtx)

    (mn, mx) = range2(mtx)

    if len(args) > 2:
        binCtr = args[2]
    else:
        binCtr = mtx.mean()

    if len(args) > 1:
        if args[1] < 0:
            binSize = -args[1]
        else:
            binSize = ( float(mx-mn) / float(args[1]) )
            tmpNbins = ( round(float(mx-binCtr) / float(binSize)) - 
                         round(float(mn-binCtr) / float(binSize)) )
            if tmpNbins != args[1]:
                print 'Warning: Using %d bins instead of requested number (%d)' % (tmpNbins, args[1])
    else:
        binSize = float(mx-mn) / 101.0

    firstBin = binCtr + binSize * round( (mn-binCtr)/float(binSize) )
    firstEdge = firstBin - (binSize / 2.0) + (binSize * 0.01)

    tmpNbins = int( round( (mx-binCtr) / binSize ) -
                    round( (mn-binCtr) / binSize ) )
    
    # numpy.histogram uses bin edges, not centers like Matlab's hist
    #bins = firstBin + binSize * numpy.array(range(tmpNbins+1))
    # compute bin edges
    binsE = firstEdge + binSize * numpy.array(range(tmpNbins+1))
    
    [N, X] = numpy.histogram(mtx, binsE)

    # matlab version return column vectors, so we will too.
    N = N.reshape(1, N.shape[0])
    X = X.reshape(1, X.shape[0])

    return (N, X)

def entropy2(*args):
    # E = ENTROPY2(MTX,BINSIZE) 
    # 
    # Compute the first-order sample entropy of MTX.  Samples of VEC are
    # first discretized.  Optional argument BINSIZE controls the
    # discretization, and defaults to 256/(max(VEC)-min(VEC)).
    #
    # NOTE: This is a heavily  biased estimate of entropy when you
    # don't have much data.
    #
    # Eero Simoncelli, 6/96. Ported to Python by Rob Young, 10/15.
    
    vec = numpy.array(args[0])
    # if 2D flatten to a vector
    if len(vec.shape) != 1 and (vec.shape[0] != 1 or vec.shape[1] != 1):
        vec = vec.flatten()

    (mn, mx) = range2(vec)

    if len(args) > 1:
        binsize = args[1]
        # why is this max in the Matlab code; it's just a float?
        # we insure that vec isn't 2D above, so this shouldn't be needed
        #nbins = max( float(mx-mn)/float(binsize) )
        nbins = float(mx-mn) / float(binsize)
    else:
        nbins = 256

    [bincount, bins] = histo(vec, nbins)

    ## Collect non-zero bins:
    H = bincount[ numpy.where(bincount > 0) ]
    H = H / float(sum(H))

    return -sum(H * numpy.log2(H))
    
def factorial(*args):
    # RES = factorial(NUM)
    #
    # Factorial function that works on matrices (matlab's does not).
    #
    # EPS, 11/02, Python port by Rob Young, 10/15

    # if scalar input make it a single element array
    if isinstance(args[0], (int, long, float)):
        num = numpy.array([args[0]])
    else:
        num = numpy.array(args[0])

    res = numpy.ones(num.shape)

    ind = numpy.where(num > 0)
        
    if num.shape[0] != 0:
        subNum = num[ numpy.where(num > 0) ]
        res[ind] = subNum * factorial(subNum-1)

    # if scalar input, return scalar
    if len(res.shape) == 1 and res.shape[0] == 1:
        return res[0]
    else:
        return res

def histoMatch(*args):
    # RES = histoMatch(MTX, N, X, mode)
    #
    # Modify elements of MTX so that normalized histogram matches that
    # specified by vectors X and N, where N contains the histogram counts
    # and X the histogram bin positions (see histo).
    #
    # new input parameter 'mode' can be either 'centers' or 'edges' that tells
    # the function if the input X values are bin centers or edges.
    #
    # Eero Simoncelli, 7/96. Ported to Python by Rob Young, 10/15.
    
    mode = str(args[3])
    mtx = numpy.array(args[0])
    N = numpy.array(args[1])
    X = numpy.array(args[2])
    if mode == 'edges':         # convert to centers
        correction = (X[0][1] - X[0][0]) / 2.0
        X = (X[0][:-1] + correction).reshape(1, X.shape[1]-1)
        
    [oN, oX] = histo(mtx.flatten(), X.flatten().shape[0])
    if mode == 'edges':        # convert to centers
        correction = (oX[0][1] - oX[0][0]) / 2.0
        oX = (oX[0][:-1] + correction).reshape(1, oX.shape[1]-1)

    # remember: histo returns a column vector, so the indexing is thus
    oStep = oX[0][1] - oX[0][0]
    oC = numpy.concatenate((numpy.array([0]), 
                            numpy.array(numpy.cumsum(oN) / 
                                        float(sum(sum(oN))))))
    oX = numpy.concatenate((numpy.array([oX[0][0]-oStep/2.0]), 
                            numpy.array(oX[0]+oStep/2.0)))
    
    N = N.flatten()
    X = X.flatten()
    N = N + N.mean() / 1e8  # HACK: no empty bins ensures nC strictly monotonic
    
    nStep = X[1] - X[0]
    nC = numpy.concatenate((numpy.array([0]), 
                            numpy.array(numpy.cumsum(N) / sum(N))))
    nX = numpy.concatenate((numpy.array([X[0] - nStep / 2.0]),
                            numpy.array(X+nStep / 2.0)))
    
    # unlike in matlab, interp1d returns a function
    func = interpolate.interp1d(nC, nX, 'linear')
    nnX = func(oC)

    return pointOp(mtx, nnX, oX[0], oStep, 0)

def imGradient(*args):
    # [dx, dy] = imGradient(im, edges) 
    #
    # Compute the gradient of the image using smooth derivative filters
    # optimized for accurate direction estimation.  Coordinate system
    # corresponds to standard pixel indexing: X axis points rightward.  Y
    # axis points downward.  EDGES specify boundary handling (see corrDn
    # for options).
    #
    # EPS, 1997.
    # original filters from Int'l Conf Image Processing, 1994.
    # updated filters 10/2003: see Farid & Simoncelli, IEEE Trans Image Processing, 13(4):496-508, April 2004.
    # Incorporated into matlabPyrTools 10/2004.
    # Python port by Rob Young, 10/15
    
    if len(args) == 0 or len(args) > 2:
        print 'Usage: imGradient(image, edges)'
        print "'edges' argument is optional"
    elif len(args) == 1:
        edges = "dont-compute"
    elif len(args) == 2:
        edges = str(args[1])
        
    im = numpy.array(args[0])

    # kernels from Farid & Simoncelli, IEEE Trans Image Processing, 
    #   13(4):496-508, April 2004.
    gp = numpy.array([0.037659, 0.249153, 0.426375, 0.249153, 0.037659]).reshape(5,1)
    gd = numpy.array([-0.109604, -0.276691, 0.000000, 0.276691, 0.109604]).reshape(5,1)

    dx = corrDn(corrDn(im, gp, edges), gd.T, edges)
    dy = corrDn(corrDn(im, gd, edges), gp.T, edges)

    return (dx,dy)

def skew2(*args):
    # Sample skew (third moment divided by variance^3/2) of a matrix.
    #  MEAN (optional) and VAR (optional) make the computation faster.

    if len(args) == 0:
        print 'Usage: skew2(matrix, mean, variance)'
        print 'mean and variance arguments are optional'
    else:
        mtx = numpy.array(args[0])

    if len(args) > 1:
        mn = args[1]
    else:
        mn = mtx.mean()

    if len(args) > 2:
        v = args[2]
    else:
        v = var2(mtx, mn)

    if isinstance(mtx, complex):
        res = ( ( ((mtx.real - mn.real)**3).mean() / (v.real**(3.0/2.0)) ) +
                ( (1j * (mtx.imag-mn.image)**3) / (v.imag**(3.0/2.0))))
    else:
        res = ((mtx.real - mn.real)**3).mean() / (v.real**(3.0/2.0))

    return res
    
def upBlur(*args):
    # RES = upBlur(IM, LEVELS, FILT)
    #
    # Upsample and blur an image.  The blurring is done with filter
    # kernel specified by FILT (default = 'binom5'), which can be a string
    # (to be passed to namedFilter), a vector (applied separably as a 1D
    # convolution kernel in X and Y), or a matrix (applied as a 2D
    # convolution kernel).  The downsampling is always by 2 in each
    # direction.
    #
    # The procedure is applied recursively LEVELS times (default=1).
    #
    # Eero Simoncelli, 4/97. Python port by Rob Young, 10/15.
    
    #---------------------------------------------------------------
    # REQUIRED ARGS
    
    if len(args) == 0:
        print 'Usage: upBlur(image, levels, filter)'
        print 'first argument is required'
    else:
        im = numpy.array(args[0])

    #---------------------------------------------------------------
    # OPTIONAL ARGS
    
    if len(args) > 1:
        nlevs = args[1]
    else:
        nlevs = 1

    if len(args) > 2:
        filt = args[2]
    else:
        filt = 'binom5'

    #------------------------------------------------------------------

    if isinstance(filt, basestring):
        filt = namedFilter(filt)

    if nlevs > 1:
        im = upBlur(im, nlevs-1, filt)

    if nlevs >= 1:
        if im.shape[0] == 1 or im.shape[1] == 1:
            if im.shape[0] == 1:
                filt = filt.reshape(filt.shape[1], filt.shape[0])
                start = (1,2)
            else:
                start = (2,1)
            res = upConv(im, filt, 'reflect1', start)
        elif filt.shape[0] == 1 or filt.shape[1] == 1:
            if filt.shape[0] == 1:
                filt = filt.reshape(filt.shape[1], 1)
            res = upConv(im, filt, 'reflect1', [2,1])
            res = upConv(res, filt.T, 'reflect1', [1,2])
        else:
            res = upConv(im, filt, 'reflect1', [2,2])
    else:
        res = im

    return res

def zconv2(*args):
    # RES = ZCONV2(MTX1, MTX2, CTR)
    #
    # Convolution of two matrices, with boundaries handled as if the larger mtx
    # lies in a sea of zeros. Result will be of size of LARGER vector.
    # 
    # The origin of the smaller matrix is assumed to be its center.
    # For even dimensions, the origin is determined by the CTR (optional) 
    # argument:
    #      CTR   origin
    #       0     DIM/2      (default)
    #       1     (DIM/2)+1  (behaves like conv2(mtx1,mtx2,'same'))
    #
    # Eero Simoncelli, 2/97.  Python port by Rob Young, 10/15.

    # REQUIRED ARGUMENTS
    #----------------------------------------------------------------
    
    if len(args) < 2 or len(args) > 3:
        print 'Usage: zconv2(matrix1, matrix2, center)'
        print 'first two input parameters are required'
    else:
        a = numpy.array(args[0])
        b = numpy.array(args[1])

    # OPTIONAL ARGUMENT
    #----------------------------------------------------------------

    if len(args) == 3:
        ctr = args[2]
    else:
        ctr = 0

    #----------------------------------------------------------------

    if (a.shape[0] >= b.shape[0]) and (a.shape[1] >= b.shape[1]):
        large = a
        small = b
    elif (a.shape[0] <= b.shape[0]) and (a.shape[1] <= b.shape[1]):
        large = b
        small = a
    else:
        print 'Error: one arg must be larger than the other in both dimensions!'
        return
        
    ly = large.shape[0]
    lx = large.shape[1]
    sy = small.shape[0]
    sx = small.shape[1]
    #print '%d %d %d %d' % (ly, lx, sy, sx)

    ## These values are the index of the small matrix that falls on the 
    ## border pixel of the large matrix when computing the first
    ## convolution response sample:
    sy2 = numpy.floor((sy+ctr+1)/2.0)-1
    sx2 = numpy.floor((sx+ctr+1)/2.0)-1

    clarge = scipy.signal.convolve(large, small, 'full')
    
    c = clarge[sy2:ly+sy2, sx2:lx+sx2]

    return c
