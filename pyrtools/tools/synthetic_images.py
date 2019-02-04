import numpy as np
from ..pyramids.c.wrapper import pointOp
from .utils import rcosFn
from .image_stats import var


def ramp(size, direction=0, slope=1, intercept=0, origin=None):
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
        origin = ((size[0] - 1)/2., (size[1] - 1)/2.)
        # origin = ( (size[0] + 1)/2., (size[1] + 1)/2. )
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    xinc = slope * np.cos(direction)
    yinc = slope * np.sin(direction)

    [xramp, yramp] = np.meshgrid(xinc * (np.arange(size[1])-origin[1]),
                                 yinc * (np.arange(size[0])-origin[0]))

    res = intercept + xramp + yramp

    return res


def impulse(size, origin=None, amplitude=1):
    '''make an impulse matrix

    create an image that is all zeros except for an impulse
    of default amplitude 1 at default position origin

    NOTE: the origin is rounded to the nearest int
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0] + 1)//2, (size[1] + 1)//2)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    res = np.zeros(size)
    res[origin[0], origin[1]] = amplitude

    return res


def polar_radius(size, exponent=1, origin=None):
    '''make distance-from-origin (r) matrix

    Compute a matrix of dimension SIZE (a [Y X] list/tuple, or a scalar)
    containing samples of a radial ramp function, raised to power EXPONENT
    (default = 1), with given ORIGIN (default = (size+1)//2, (0, 0) = upper left).

    NOTE: the origin is not rounded to the nearest int
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    xramp, yramp = np.meshgrid(np.arange(1, size[1]+1)-origin[1],
                               np.arange(1, size[0]+1)-origin[0])

    if exponent <= 0:
        # zero to a negative exponent raises:
        # ZeroDivisionError: 0.0 cannot be raised to a negative power
        r = xramp ** 2 + yramp ** 2
        res = np.power(r, exponent / 2.0, where=(r != 0))
    else:
        res = (xramp ** 2 + yramp ** 2) ** (exponent / 2.0)
    return res


def polar_angle(size, phase=0, origin=None):
    '''make polar angle matrix (in radians)

    Compute a matrix of dimension SIZE (a [Y X] list/tuple, or a scalar)
    containing samples of the polar angle (in radians, CW from the X-axis,
    ranging from -pi to pi), relative to angle PHASE (default = 0), about ORIGIN
    pixel (default = (size+1)/2).

    NOTE: the origin is not rounded to the nearest int
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    xramp, yramp = np.meshgrid(np.arange(1, size[1]+1)-origin[1],
                               np.arange(1, size[0]+1)-origin[0])
    xramp = np.array(xramp)
    yramp = np.array(yramp)

    res = np.arctan2(yramp, xramp)

    res = ((res+(np.pi-phase)) % (2*np.pi)) - np.pi

    return res


def disk(size, radius=None, origin=None, twidth=2, vals=(1, 0)):
    '''make a "disk" image

    SIZE (a [Y X] list/tuple, or a scalar) specifies the matrix size
    RADIUS (default = min(size)/4) specifies the radius of the disk
    ORIGIN (default = (size+1)/2) specifies the location of the disk center
    TWIDTH (in pixels, default = 2) specifies the width over which a soft
    threshold transition is made. VALS (default = [0,1]) should be a 2-vector
    containing the intensity value inside and outside the disk.
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if radius is None:
        radius = min(size) / 4.0

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    res = polar_radius(size, exponent=1, origin=origin)

    if abs(twidth) < np.finfo(np.double).tiny:
        res = vals[1] + (vals[0] - vals[1]) * (res <= radius)
    else:
        [Xtbl, Ytbl] = rcosFn(twidth, radius, [vals[0], vals[1]])
        res = pointOp(res, Ytbl, Xtbl[0], Xtbl[1]-Xtbl[0], 0)

    return np.array(res)


def gaussian(size, covariance=None, origin=None, amplitude='norm'):
    '''make a two dimensional Gaussian

    make a two dimensional Gaussian function of SIZE (a [Y X] list/tuple,
    or a scalar), centered at pixel position specified by ORIGIN
    (default = (size+1)/2), with given COVARIANCE (can be a scalar, 2-vector,
    or 2x2 matrix -  Default = (min(size)/6)^2 ) AMPLITUDE='norm' (default)
    will produce a probability-normalized function

    TODO - use built in scipy function
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if covariance is None:
        covariance = (min([size[0], size[1]]) / 6.0) ** 2
    covariance = np.array(covariance)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    (xramp, yramp) = np.meshgrid(np.arange(1, size[1]+1)-origin[1],
                                 np.arange(1, size[0]+1)-origin[0])

    if len(covariance.shape) == 0:
        if isinstance(amplitude, str) and amplitude == 'norm':
            amplitude = 1.0 / (2.0 * np.pi * covariance)
        e = ((xramp ** 2) + (yramp ** 2)) / (-2.0 * covariance)

    elif len(covariance.shape) == 1:
        if isinstance(amplitude, str) and amplitude == 'norm':
            if covariance[0] * covariance[1] < 0:
                amplitude = 1.0 / (2.0 * np.pi * np.sqrt(complex(covariance[0] * covariance[1])))
            else:
                amplitude = 1.0 / (2.0 * np.pi * np.sqrt(covariance[0] * covariance[1]))
        e = (((xramp ** 2) / (-2 * covariance[1])) +
             ((yramp ** 2) / (-2 * covariance[0])))

    elif covariance.shape == (2, 2):
        # square matrix
        if isinstance(amplitude, str) and amplitude == 'norm':
            detCov = np.linalg.det(covariance)
            if (detCov < 0).any():
                detCovComplex = np.empty(detCov.shape, dtype=complex)
                detCovComplex.real = detCov
                detCovComplex.imag = np.zeros(detCov.shape)
                amplitude = 1.0 / (2.0 * np.pi * np.sqrt(detCovComplex))
            else:
                amplitude = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(covariance)))
        covariance = - np.linalg.inv(covariance) / 2.0
        e = ((covariance[1, 1] * xramp**2) +
             ((covariance[0, 1] + covariance[1, 0]) * (xramp*yramp)) +
             (covariance[0, 0] * yramp**2))
    else:
        raise Exception("ERROR: invalid covariance shape")

    res = amplitude * np.exp(e)

    return res


def zone_plate(size, amplitude=1, phase=0):
    '''make a "zone plate" image

    SIZE specifies the matrix size
    AMPL * cos( r^2 + PHASE) (default = 1)
    PHASE (default = 0) are optional
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    res = amplitude * np.cos((np.pi / max(size)) * polar_radius(size, 2) + phase)

    return res


def angular_sine(size, harmonic=1, amplitude=1, phase=0, origin=None):
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

    res = amplitude * np.sin(harmonic * polar_angle(size, phase, origin) + phase)

    return res


def sine(size, period=None, direction=None, frequency=None, amplitude=1,
         phase=0, origin=None):
    ''' make a two dimensional sinusoid

    IM = sine(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN)
              or
    IM = sine(SIZE,      FREQ,         AMPLITUDE, PHASE, ORIGIN)

    Compute a matrix of dimension SIZE (a [Y X] list/tuple, or a scalar)
    containing samples of a 2D sinusoid, with given PERIOD (in pixels),
    DIRECTION (radians, ClockWise from X-axis, default = 0), AMPLITUDE (default
    = 1), and PHASE (radians, relative to ORIGIN, default = 0).  ORIGIN
    defaults to the center of the image.

    In the second form, FREQ is a 2-vector of frequencies (radians/pixel).
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    # first form
    if isinstance(period, (int, float)):
        frequency = (2.0 * np.pi) / period
        if direction is None:
            direction = 0

    # second form
    elif frequency is not None:
        direction = np.arctan2(frequency[0], frequency[1])
        frequency = np.linalg.norm(frequency)

    elif period is None and direction is None and frequency is None:
        frequency = (2.0 * np.pi) / np.log2(size[0])
        direction = 0

    if origin is None:
        res = amplitude * np.sin(ramp(size=size, direction=direction,
                                      slope=frequency, intercept=phase))

    else:
        if not hasattr(origin, '__iter__'):
            origin = (origin, origin)
        res = amplitude * np.sin(ramp(size=size, direction=direction,
                                      slope=frequency, intercept=phase,
                                      origin=[origin[0]-1, origin[1]-1]))

    return res


def square_wave(size, period=None, direction=None, frequency=None, amplitude=1,
                phase=0, origin=None, twidth=None):
    '''make a two dimensional square wave

    IM = square_wave(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN, TWIDTH)
            or
    IM = square_wave(SIZE,      FREQ,         AMPLITUDE, PHASE, ORIGIN, TWIDTH)

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

    # first form
    if isinstance(period, (int, float)):
        frequency = (2.0 * np.pi) / period
        if direction is None:
            direction = 0.0

    # second form
    elif frequency is not None:
        direction = np.arctan2(frequency[0], frequency[1])
        frequency = np.linalg.norm(frequency)

    elif period is None and direction is None and frequency is None:
        frequency = (2.0 * np.pi) / np.log2(size[0])
        direction = 0

    if twidth is None:
        twidth = min(2, 2.0 * np.pi / (3.0*frequency))

    if origin is None:
        res = ramp(size, direction=direction, slope=frequency,
                   intercept=phase) - np.pi/2.0

    else:
        if not hasattr(origin, '__iter__'):
            origin = (origin, origin)
        res = ramp(size, direction=direction, slope=frequency,
                   intercept=phase, origin=[origin[0]-1, origin[1]-1]) - np.pi/2.0

    [Xtbl, Ytbl] = rcosFn(twidth * frequency, np.pi/2.0,
                          [-amplitude, amplitude])

    res = pointOp(abs(((res+np.pi) % (2.0*np.pi))-np.pi), Ytbl,
                  Xtbl[0], Xtbl[1]-Xtbl[0], 0)

    return res


def pink_noise(size, fract_dim=1):
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

    exp = -(2.5-fract_dim)
    ctr = np.ceil((res.shape + np.ones(2))/2.)
    sh = np.fft.ifftshift(polar_radius(size, exp, ctr))
    sh[0, 0] = 1  # DC term

    fres = sh * fres
    fres = np.fft.ifft2(fres)

    if abs(fres.imag).max() > 1e-10:
        print('Symmetry error in creating fractal')
    else:
        res = np.real(fres)
        res = res / np.sqrt(var(res))

    return res
