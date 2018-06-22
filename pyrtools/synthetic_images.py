import numpy as np
import sys

# TODO - update these modules to new names / locations
from .rcosFn import rcosFn
from .convolutions import pointOp

def mkRamp(size, direction=0, slope=1, intercept=0, origin=None):
    ''' make a ramp matrix

    Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
    containing samples of a ramp function, with given gradient DIRECTION
    (radians, CW from X-axis, default = 0), SLOPE (per pixel, default =
    1), and a value of INTERCEPT (default = 0) at the ORIGIN (default =
    (size+1)/2, (0, 0) = upper left)
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        # TODO understand why minus one (not plus)
        origin = ( (size[0] - 1)/2., (size[1] - 1)/2. )
        # origin = ( (size[0] + 1)/2., (size[1] + 1)/2. )
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    xinc = slope * np.cos(direction)
    yinc = slope * np.sin(direction)

    [xramp, yramp] = np.meshgrid( xinc * (np.array(range(size[1]))-origin[1]),
                                  yinc * (np.array(range(size[0]))-origin[0]) )

    res = intercept + xramp + yramp

    return res

def mkImpulse(size, origin=None, amplitude=1):
    '''make an impulse matrix

    create an image that is all zeros except for an impulse
    of default amplitude 1 at default position origin

    NOTE: the origin is rounded to the nearest int
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ( (size[0] + 1)//2, (size[1] + 1)//2 )
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    res = np.zeros(size)
    res[origin[0], origin[1]] = amplitude

    return res

def mkR(size, exponent=1, origin=None):
    '''make distance-from-origin (r) matrix

    Compute a matrix of dimension SIZE (a [Y X] list/tuple, or a scalar) containing samples of a
    radial ramp function, raised to power EXPONENT (default = 1), with given ORIGIN (default =
    (size+1)//2, (0, 0) = upper left).

    NOTE: the origin is not rounded to the nearest int
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    xramp, yramp = np.meshgrid(np.array(list(range(1, size[1]+1)))-origin[1],
                               np.array(list(range(1, size[0]+1)))-origin[0])

    res = (xramp**2 + yramp**2)**(exponent/2.0)

    return res

def mkAngle(size, phase=0, origin=None):
    '''make polar angle matrix (in radians)

    Compute a matrix of dimension SIZE (a [Y X] list/tuple, or a scalar) containing samples of the
    polar angle (in radians, CW from the X-axis, ranging from -pi to pi), relative to angle PHASE
    (default = 0), about ORIGIN pixel (default = (size+1)/2).

    NOTE: the origin is not rounded to the nearest int
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    xramp, yramp = np.meshgrid(np.array(list(range(1, size[1]+1)))-origin[1],
                                  np.array(list(range(1, size[0]+1)))-origin[0])
    xramp = np.array(xramp)
    yramp = np.array(yramp)

    res = np.arctan2(yramp, xramp)

    res = ((res+(np.pi-phase)) % (2*np.pi)) - np.pi

    return res

def mkDisc(size, radius=None, origin=None, twidth=2, vals=(1,0)):
    '''make a "disk" image

    SIZE (a [Y X] list/tuple, or a scalar) specifies the matrix size
    RADIUS (default = min(size)/4) specifies the radius of the disk
    ORIGIN (default = (size+1)/2) specifies the location of the disk center
    TWIDTH (in pixels, default = 2) specifies the width over which a soft threshold transition is made.
    VALS (default = [0,1]) should be a 2-vector containing the intensity value inside and outside the disk.
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if radius is None:
        radius = min(size) / 4.0

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    #--------------------------------------------------------------

    res = mkR(size, 1, origin)

    if abs(twidth) < sys.float_info.min:
        res = vals[1] + (vals[0] - vals[1]) * (res <= rad)
    else:
        [Xtbl, Ytbl] = rcosFn(twidth, rad, [vals[0], vals[1]])
        res = pointOp(res, Ytbl, Xtbl[0], Xtbl[1]-Xtbl[0], 0)

    return np.array(res)

def mkGaussian(size, covariance=None, origin=None, amplitude='norm'):
    '''make a two dimensional Gaussian

    A two dimensional Gaussian function of SIZE (a [Y X] list/tuple, or a scalar),
    centered at pixel position specified by ORIGIN (default = (size+1)/2),
    with given COVARIANCE (can be a scalar, 2-vector, or 2x2 matrix -  Default = (min(size)/6)^2 )
    AMPLITUDE='norm' (default) will produce a probability-normalized function

    TODO - use built in scipy function
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if covariance is None:
        covariance = (min([size[0], size[1]]) / 6.0) ** 2

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    #---------------------------------------------------------------

    (xramp, yramp) = np.meshgrid(np.array(list(range(1,size[1]+1)))-origin[1],
                                 np.array(list(range(1,size[0]+1)))-origin[0])

    if isinstance(cov, (int, float)):
        if amplitude == 'norm':
            amplitude = 1.0 / (2.0 * np.pi * cov)
        e = ( (xramp ** 2) + (yramp ** 2) ) / ( -2.0 * cov )

    elif len(cov) == 2 and isinstance(cov[0], (int, float)):
        if amplitude == 'norm':
            if cov[0] * cov[1] < 0:
                amplitude = 1.0 / (2.0 * np.pi *
                              np.sqrt(complex(cov[0] * cov[1])))
            else:
                amplitude = 1.0 / (2.0 * np.pi * np.sqrt(cov[0] * cov[1]))
        e = ( (xramp ** 2) / (-2 * cov[1]) ) + ( (yramp ** 2) / (-2 * cov[0]) )

    elif cov.shape == (2,2):
        if amplitude == 'norm':
            detCov = np.linalg.det(cov)
            if (detCov < 0).any():
                detCovComplex = np.empty(detCov.shape, dtype=complex)
                detCovComplex.real = detCov
                detCovComplex.imag = np.zeros(detCov.shape)
                amplitude = 1.0 / ( 2.0 * np.pi * np.sqrt( detCovComplex ) )
            else:
                amplitude = 1.0 / (2.0 * np.pi * np.sqrt( np.linalg.det(cov) ) )
        cov = - np.linalg.inv(cov) / 2.0
        e = (cov[1,1] * xramp**2) + (
            (cov[0,1]+cov[1,0])*(xramp*yramp) ) + ( cov[0,0] * yramp**2)

    res = amplitude * np.exp(e)

    return res

def mkZonePlate(size, amplitude = 1, phase=0):
    '''make a "zone plate" image

    SIZE specifies the matrix size
    AMPL * cos( r^2 + PHASE) (default = 1)
    PHASE (default = 0) are optional
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    res = amplitude * np.cos( (np.pi / max(size)) * mkR(size, 2) + phase )

    return res

def mkAngularSine(size, harmonic=1, amplitude=1, phase=0, origin=None):
    '''make an angular sinusoidal image:

    AMPL * sin( HARMONIC*theta + PHASE),
    where theta is the angle about the origin.
    SIZE specifies the matrix size (a [Y X] list/tuple, or a scalar)
    AMPL (default = 1) and PHASE (default = 0) are optional.
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    res = amplitude * np.sin( harmonic * mkAngle(size, phase, origin) + phase )

    return res

def mkSine(size, period=None, direction=None, frequency=None, amplitude=1, phase=0, origin=None):
    ''' make a two dimensional sinusoid

    IM = mkSine(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN)
              or
    IM = mkSine(SIZE,      FREQ,         AMPLITUDE, PHASE, ORIGIN)

    Compute a matrix of dimension SIZE (a [Y X] list/tuple, or a scalar)
    containing samples of a 2D sinusoid, with given PERIOD (in pixels),
    DIRECTION (radians, ClockWise from X-axis, default = 0), AMPLITUDE (default
    = 1), and PHASE (radians, relative to ORIGIN, default = 0).  ORIGIN
    defaults to the center of the image.

    In the second form, FREQ is a 2-vector of frequencies (radians/pixel).
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        # TODO make sure this is handled correctly
        origin = ((size[0]-1)/2., (size[1]-1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    # first form
    if isinstance(period, (int, float)):
        frequency = (2.0 * np.pi) / period
        if direction is None:
            direction = 0

    # second form
    elif frequency is not None:
        frequency = np.linalg.norm(frequency)
        direction = np.atan2(frequency[0], frequency[1])

    #----------------------------------------------------------------

    # if origin == None:
    #     res = amplitude * np.sin(mkRamp(size, direction, frequency, phase))
    # else:

    res = amplitude * np.sin(mkRamp(size, direction, frequency, phase,
                                           [origin[0]-1, origin[1]-1]))

    return res

def mkSquare(size, period=None, direction=None, frequency=None, amplitude=1, phase=0, origin=None, twidth=None):
    '''make a two dimensional square wave

    IM = mkSquare(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN, TWIDTH)
            or
    IM = mkSquare(SIZE,      FREQ,         AMPLITUDE, PHASE, ORIGIN, TWIDTH)

    Compute a matrix of dimension SIZE (a [Y X] list/tuple, or a scalar)
    containing samples of a 2D square wave, with given PERIOD (in
    pixels), DIRECTION (radians, CW from X-axis, default = 0), AMPLITUDE
    (default = 1), and PHASE (radians, relative to ORIGIN, default = 0).
    ORIGIN defaults to the center of the image.  TWIDTH specifies width
    of raised-cosine edges on the bars of the grating (default =
    min(2,period/3)).

    In the second form, FREQ is a 2-vector of frequencies (radians/pixel).
    TODO: Add duty cycle
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        # TODO make sure this is handled correctly
        origin = ((size[0]-1)/2., (size[1]-1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    # first form
    if isinstance(period, (int, float)):
        frequency = (2.0 * np.pi) / period
        if direction is None:
            direction = 0

    # second form
    elif frequency is not None:
        frequency = np.linalg.norm(frequency)
        direction = np.atan2(frequency[0], frequency[1])

    if twidth is None:
        twidth = min(2, 2.0 * np.pi / (3.0*frequency))

    #------------------------------------------------------------

    # if origin != 'not set':
    #     res = mkRamp(size, direction, frequency, phase,
    #                  (origin[0]-1, origin[1]-1)) - np.pi/2.0
    # else:
    #
    res = mkRamp(size, direction, frequency, phase) - np.pi/2.0

    [Xtbl, Ytbl] = rcosFn(transition * frequency, np.pi/2.0,
                          [-amplitude, amplitude])

    res = pointOp(abs(((res+np.pi) % (2.0*np.pi))-np.pi), Ytbl,
                  Xtbl[0], Xtbl[1]-Xtbl[0], 0)

    return res

def mkFract(size, fract_dim=1):
    '''make pink noise

    Make a matrix of dimensions SIZE (a [Y X] list/tuple, or a scalar)
    containing fractal (pink) noise with power spectral density of the
    form: 1/f^(5-2*FRACT_DIM).  Image variance is normalized to 1.0.
    FRACT_DIM defaults to 1.0

    TODO: Verify that this  matches Mandelbrot defn of fractal dimension.
          Make this more efficient!
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    res = np.random.randn(size[0], size[1])
    fres = np.fft.fft2(res)

    # TODO this should not change
    size = res.shape

    # ctr = (int(np.ceil((size[0]+1)/2.0)), int(np.ceil((size[1]+1)/2.0)))
    origin = ((size[0]-1)/2., (size[1]-1)/2.)

    sh = np.fft.ifftshift(mkR(size, -(2.5-fract_dim), origin))
    sh[0,0] = 1  #DC term

    fres = sh * fres
    fres = np.fft.ifft2(fres)

    if abs(fres.imag).max() > 1e-10:
        print('Symmetry error in creating fractal')
    else:
        res = np.real(fres)
        res = res / np.sqrt(var2(res))

    return res


if __name__ == "__main__":

    # TODO - update this module to new names / locations
    from display import showIm

    # pick some parameters
    size      = 256
    direction = 2 * np.pi * np.random.rand(1)
    slope     = 10 * np.random.rand(1) - 5
    intercept = 10 * np.random.rand(1) - 5
    origin    = round(1 + (size - 1) * np.random.rand(2,1))
    exponent  = 0.8 + np.random.rand(1)
    amplitude = 1 + 5 * np.random.rand(1)
    phase     = 2 * np.pi * np.random.rand(1)
    period    = 20
    twidth    = 7

    showIm(mkRamp(size, direction, slope, intercept, origin))
    showIm(mkImpulse(size, origin, amplitude))
    showIm(mkR(size, exponent, origin))
    showIm(mkAngle(size, direction))
    showIm(mkDisc(size, size/4, origin, twidth))
    showIm(mkGaussian(size, (size/6)^2, origin, amplitude)) # try various covariances
    showIm(mkZonePlate(size, amplitude, phase))
    showIm(mkAngularSine(size, 3, amplitude, phase, origin))
    showIm(mkSine(size, period, direction, amplitude, phase, origin))
    showIm(mkSquare(size, period, direction, amplitude, phase, origin, twidth))
    showIm(mkFract(size, exponent))
