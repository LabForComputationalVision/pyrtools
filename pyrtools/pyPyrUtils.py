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
from . import JBhelpers
import PIL
from PIL import ImageTk
import tkinter
import ctypes
import os
from .convolutions import corrDn, upConv, pointOp

# we want to support both PyQt5 and PyQt4. In order to do that, we use this little work around. Any
# other functions that need to make use of the PyQt functionality will import QtGui or QtCore from
# this module.
# try:
#     __import__('PyQt5')
#     use_pyqt5 = True
# except ImportError:
#     use_pyqt5 = False

# if use_pyqt5:
#     from PyQt5 import QtGui
#     from PyQt5 import QtCore
# else:
#     from PyQt4 import QtGui
#     from PyQt4 import QtCore


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
            print("Error: cannot have a 1D 'image' and 2D filter")
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

from .namedFilter import namedFilter

def compareRecon(recon1, recon2):
    prec = -11
    if recon1.shape != recon2.shape:
        print('shape is different!')
        print(recon1.shape)
        print(recon2.shape)
        return 0

    for i in range(recon1.shape[0]):
        for j in range(recon2.shape[1]):
            if numpy.absolute(recon1[i,j].real - recon2[i,j].real) > math.pow(10,-11):
                print("real: i=%d j=%d %.15f %.15f diff=%.15f" % (i, j, recon1[i,j].real, recon2[i,j].real, numpy.absolute(recon1[i,j].real-recon2[i,j].real)))
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
                print(prec)
                if numpy.absolute(recon1[i,j].imag - recon2[i,j].imag) > math.pow(10, prec):
                    print("imag: i=%d j=%d %.15f %.15f diff=%.15f" % (i, j, recon1[i,j].imag, recon2[i,j].imag, numpy.absolute(recon1[i,j].imag-recon2[i,j].imag)))
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
        print("mkAngularSine(SIZE, HARMONIC, AMPL, PHASE, ORIGIN)")
        print("first argument is required")
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print("first argument must be a two element tuple or an integer")
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
        print("mkRamp(SIZE, COVARIANCE, MEAN, AMPLITUDE)")
        print("first argument is required")
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print("first argument must be a two element tuple or an integer")
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

    (xramp, yramp) = numpy.meshgrid(numpy.array(list(range(1,sz[1]+1)))-mn[1],
                                    numpy.array(list(range(1,sz[0]+1)))-mn[0])

    if isinstance(cov, (int, float)):
        if 'norm' == ampl:
            ampl = 1.0 / (2.0 * numpy.pi * cov)
        e = ( (xramp**2) + (yramp**2) ) / ( -2.0 * cov )
    elif len(cov) == 2 and isinstance(cov[0], (int, float)):
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
        print("mkDisc(SIZE, RADIUS, ORIGIN, TWIDTH, VALS)")
        print("first argument is required")
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print("first argument must be a two element tuple or an integer")
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
        print("mkSine(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN)")
        print("       or")
        print("mkSine(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN)")
        print("first two arguments are required")
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print("first argument must be a two element tuple or an integer")
            exit(1)

    if isinstance(args[1], (int, float)):
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
        print("mkZonePlate(SIZE, AMPL, PHASE)")
        print("first argument is required")
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print("first argument must be a two element tuple or an integer")
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
        print("mkSquare(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN, TWIDTH)")
        print("       or")
        print("mkSquare(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN, TWIDTH)")
        print("first two arguments are required")
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print("first argument must be a two element tuple or an integer")
            exit(1)

    if isinstance(args[1], (int, float)):
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
        print("mkRamp(SIZE, DIRECTION, SLOPE, INTERCEPT, ORIGIN)")
        print("first argument is required")
        exit(1)
    else:
        sz = args[0]
        if isinstance(sz, (int)):
            sz = (sz, sz)
        elif not isinstance(sz, (tuple)):
            print("first argument must be a two element tuple or an integer")
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

    [xramp, yramp] = numpy.meshgrid( xinc * (numpy.array(list(range(sz[1])))-origin[1]),
                                  yinc * (numpy.array(list(range(sz[0])))-origin[0]) )

    res = intercept + xramp + yramp

    return res

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
        print("mkImpulse(size, origin, amplitude)")
        print("first input parameter is required")
        return

    if(isinstance(args[0], int)):
        sz = (args[0], args[0])
    elif(isinstance(args[0], tuple)):
        sz = args[0]
    else:
        print("size parameter must be either an integer or a tuple")
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
        print("Error: first parameter 'harmonics' is required.")
        return

    if len(args) > 0:
        harmonics = numpy.array(args[0])

    # optional parameters
    numh = (2*harmonics.shape[0]) - (harmonics == 0).sum()
    if len(args) > 1:
        angles = args[1]
    else:
        angles = numpy.pi * numpy.array(list(range(numh))) / numh

    if len(args) > 2:
        if isinstance(args[2], str):
            if args[2] == 'even' or args[2] == 'EVEN':
                evenorodd = 0
            elif args[2] == 'odd' or args[2] == 'ODD':
                evenorodd = 1
            else:
                print("Error: only 'even' and 'odd' are valid entries for the third input parameter.")
                return
        else:
            print("Error: third input parameter must be a string (even/odd).")
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
        print("Warning: matrix is not full rank")

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

    X = numpy.pi * numpy.array(list(range(-sz-1,2))) / (2*sz)

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
        print("Error: first input parameter 'size' is required!")
        print("makeAngle(size, phase, origin)")
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

    (xramp, yramp) = numpy.meshgrid(numpy.array(list(range(1,sz[1]+1)))-origin[1],
                                 (numpy.array(list(range(1,sz[0]+1))))-origin[0])
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
        print("Error: filter input parameter required.")
        return

    lfilt = args[0]
    # reshape lfilt to column vector
    if len(lfilt.shape) == 1:
        lfilt = lfilt.reshape(len(lfilt), 1)
    elif lfilt.shape[0] == 1:
        lfilt = lfilt.reshape(lfilt.shape[1], 1)
    elif len(lfilt.shape) > 2 or lfilt.shape[1] != 1:
        print('Error: only 1D input supported.')
        return

    sz = len(lfilt)
    sz2 = numpy.ceil(sz/2.0);

    ind = numpy.array(list(range(sz-1,-1,-1)))

    hfilt = lfilt[ind].T * (-1)**((ind+1)-sz2)

    # matlab version always returns a column vector
    if len(hfilt.shape) == 1:
        hfilt = hfilt.reshape(len(hfilt), 1)
    elif hfilt.shape[0] == 1:
        hfilt = hfilt.reshape(hfilt.shape[1], 1)

    return hfilt


# compute minimum and maximum values of input matrix, returning them as tuple
from .imStats import imStats, range2, var2, skew2, kurt2

# makes image the same as read in by matlab
def correctImage(img):
    #tmpcol = img[:,0]
    #for i in range(img.shape[1]-1):
    #    img[:,i] = img[:,i+1]
    #img[:, img.shape[1]-1] = tmpcol
    #return img
    return numpy.roll(img, -1)

# Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
# containing samples of a radial ramp function, raised to power EXPT
# (default = 1), with given ORIGIN (default = (size+1)/2, [1 1] =
# upper left).  All but the first argument are optional.
# Eero Simoncelli, 6/96.  Ported to Python by Rob Young, 5/14.
def mkR(*args):
    if len(args) == 0:
        print('Error: first input parameter is required!')
        return
    else:
        sz = args[0]

    if isinstance(sz, int) or len(sz) == 1:
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

    (xramp2, yramp2) = numpy.meshgrid(numpy.array(list(range(1,sz[1]+1)))-origin[1],
                                   numpy.array(list(range(1,sz[0]+1)))-origin[0])

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
        print('Error: input parameter dims required')
    else:
        if isinstance(args[0], int) or len(args[0]) == 1:
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
        print('Symmetry error in creating fractal')
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
        print('Error: input parameters basis and angle are required!')
        return

    basis = args[0]

    num = basis.shape[1]

    #if ( any(size(angle) ~= [size(basis,1) 1]) & any(size(angle) ~= [1 1]) )
    angle = args[1]
    if isinstance(angle, (int, float)):
        angle = numpy.array([angle])
    else:
        if angle.shape[0] != basis.shape[0] or angle.shape[1] != 1:
            print('ANGLE must be a scalar, or a column vector the size of the basis elements')
            return

    # If HARMONICS are not passed, assume derivatives.
    if len(args) < 3:
        if num%2 == 0:
            harmonics = numpy.array(list(range(num/2)))*2+1
        else:
            harmonics = numpy.array(list(range((15+1)/2)))*2
    else:
        harmonics = args[2]
    if len(harmonics.shape) == 1 or harmonics.shape[0] == 1:
        harmonics = harmonics.reshape(harmonics.shape[0], 1)
    elif harmonics.shape[0] != 1 and harmonics.shape[1] != 1:
        print('Error: input parameter HARMONICS must be 1D!')
        return

    if 2*harmonics.shape[0] - (harmonics == 0).sum() != num:
        print('harmonics list is incompatible with basis size!')
        return

    # If STEERMTX not passed, assume evenly distributed cosine-phase filters:
    if len(args) < 4:
        steermtx = steer2HarmMtx(harmonics, numpy.pi*numpy.array(list(range(num)))/num,
                                 'even')
    else:
        steermtx = args[3]

    steervect = numpy.zeros((angle.shape[0], num))
    arg = angle * harmonics[numpy.nonzero(harmonics)[0]].T
    if all(harmonics):
        steervect[:, list(range(0,num,2))] = numpy.cos(arg)
        steervect[:, list(range(1,num,2))] = numpy.sin(arg)
    else:
        steervect[:, 1] = numpy.ones((arg.shape[0],1))
        steervect[:, list(range(0,num,2))] = numpy.cos(arg)
        steervect[:, list(range(1,num,2))] = numpy.sin(arg)

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
        print("showIm( matrix, range, zoom, label, nshades )")
        print("  matrix is string. It should be the name of a 2D array.")
        print("  range is a two element tuple.  It specifies the values that ")
        print("    map to the min and max colormap values.  Passing a value ")
        print("    of 'auto' (default) sets range=[min,max].  'auto2' sets ")
        print("    range=[mean-2*stdev, mean+2*stdev].  'auto3' sets ")
        print("    range=[p1-(p2-p1)/8, p2+(p2-p1)/8], where p1 is the 10th ")
        print("    percientile value of the sorted matix samples, and p2 is ")
        print("    the 90th percentile value.")
        print("  zoom specifies the number of matrix samples per screen pixel.")
        print("    It will be rounded to an integer, or 1 divided by an ")
        print("    integer.")
        #print "    A value of 'same' or 'auto' (default) causes the "
        #print "    zoom value to be chosen automatically to fit the image into"
        #print "    the current axes."
        #print "    A value of 'full' fills the axis region "
        #print "    (leaving no room for labels)."
        print("  label - A string that is used as a figure title.")
        print("  NSHADES (optional) specifies the number of gray shades, ")
        print("    and defaults to the size of the current colormap. ")

    if len(args) > 0:   # matrix entered
        matrix = numpy.array(args[0])
    #print 'showIm range %f %f' % (matrix.min(), matrix.max())

    if len(args) > 1:   # range entered
        if isinstance(args[1], str):
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
                print("Error: range of %s is not recognized." % args[1])
                print("       please use a two element tuple or ")
                print("       'auto', 'auto2' or 'auto3'")
                print("       enter 'showIm' for more info about options")
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
    colors = list(range(0,255,(256/nshades)+1))
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
    print('dims')
    print(dims)
    print('nRows=%d nCols=%d' % (nRows, nCols))
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
        print("showIm( matrix, range, zoom, label, nshades )")
        print("  matrix is string. It should be the name of a 2D array.")
        print("  range is a two element tuple.  It specifies the values that ")
        print("    map to the min and max colormap values.  Passing a value ")
        print("    of 'auto' (default) sets range=[min,max].  'auto2' sets ")
        print("    range=[mean-2*stdev, mean+2*stdev].  'auto3' sets ")
        print("    range=[p1-(p2-p1)/8, p2+(p2-p1)/8], where p1 is the 10th ")
        print("    percientile value of the sorted matix samples, and p2 is ")
        print("    the 90th percentile value.")
        print("  zoom specifies the number of matrix samples per screen pixel.")
        print("    It will be rounded to an integer, or 1 divided by an ")
        print("    integer.")
        #print "    A value of 'same' or 'auto' (default) causes the "
        #print "    zoom value to be chosen automatically to fit the image into"
        #print "    the current axes."
        #print "    A value of 'full' fills the axis region "
        #print "    (leaving no room for labels)."
        print("  label - A string that is used as a figure title.")
        print("  NSHADES (optional) specifies the number of gray shades, ")
        print("    and defaults to the size of the current colormap. ")

    if len(args) > 0:   # matrix entered
        matrix = numpy.array(args[0])
    #print 'showIm range %f %f' % (matrix.min(), matrix.max())

    if len(args) > 1:   # range entered
        if isinstance(args[1], str):
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
                print("Error: range of %s is not recognized." % args[1])
                print("       please use a two element tuple or ")
                print("       'auto', 'auto2' or 'auto3'")
                print("       enter 'showIm' for more info about options")
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
    master = tkinter.Toplevel()
    master.title('showIm')
    canvas_width = matrix.shape[0] * zoom
    canvas_height = matrix.shape[1] * zoom
    master.geometry(str(canvas_width+20) + "x" + str(canvas_height+60) +
                    "+200+200")
    # put in top spacer
    spacer = tkinter.Label(master, text='').pack()

    # create canvas
    canvas = tkinter.Canvas(master, width=canvas_width, height=canvas_height)
    canvas.pack()
    #img = Image.fromarray(matrix)
    # FIX: shift matrix to 0.0-1.0 then to 0-255
    if (matrix < 0).any():
        matrix = matrix + math.fabs(matrix.min())
    matrix = (matrix / matrix.max()) * 255.0
    print(matrix.astype('uint8')[0,:])
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
    colors = list(range(int(matrix.min()), int(matrix.max())+1, incr))
    colors[0] = 0
    colors[-1] = 255
    colctr = -1
    # compute color transition indices
    #thresh = matlab_round( (imRange[1]-imRange[0]) / len(colors) )
    thresh = matlab_round( (matrix.max() - matrix.min()) / len(colors) )
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
    canvas.create_image(0,0, anchor=tkinter.NW, image=imgPI)

    # add labels
    rangeStr = 'Range: [%.1f, %.1f]' % (imRange[0], imRange[1])
    rangeLabel = tkinter.Label(master, text=rangeStr).pack()
    dimsStr = 'Dims: [%d, %d] / %d' % (matrix.shape[0], matrix.shape[1], zoom)
    dimsLabel = tkinter.Label(master, text=dimsStr).pack()

    tkinter.mainloop()

# round equivalent to matlab function
from .utils import matlab_round, shift, strictly_decreasing, clip
from .imStats import matlab_histo, entropy2

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

    [oN, oX] = matlab_histo(mtx.flatten(), X.flatten().shape[0])
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

from .image_tools import imGradient
