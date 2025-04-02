import numpy as np
from ..pyramids.c.wrapper import pointOp
from .utils import rcosFn
from .image_stats import var


def ramp(size, direction=0, slope=1, intercept=0, origin=None):
    '''make a ramp matrix

    Compute a matrix containing samples of a ramp function in a given direction.

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the ramp should be of dimensions `(size, size)`. if a tuple, must be a
        2-tuple of ints specifying the dimensions
    direction : `float`
        the direction of the ramp's gradient direction, in radians, clockwise from the X-axis
    slope : `float`
        the slope of the ramp (per pixel)
    intercept : `intercept`
        the value of the ramp at the origin
    origin : `int`, `tuple`, or None
        the origin of the matrix. if an int, we assume the origin is at `(origin, origin)`. if a
        tuple, must be a 2-tuple of ints specifying the origin (where `(0, 0)` is the upper left).
        if None, we assume the origin lies at the center of the matrix, `(size-1)/2`.

    Returns
    -------
    res : `np.array`
        the ramp matrix
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

    create an image that is all zeros except for an impulse of a given amplitude at the origin

    NOTE: the origin is rounded to the nearest int

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size, size)`. if a tuple, must
        be a 2-tuple of ints specifying the dimensions
    origin : `int`, `tuple`, or None
        the location of the impulse. if an int, we assume the origin is at `(origin, origin)`. if a
        tuple, must be a 2-tuple of ints specifying the origin (where `(0, 0)` is the upper left).
        if None, we assume the origin lies at the center of the matrix, `(size+1)//2` (note: this
        is rounded to the nearest int)
    amplitude : `float`
        the amplitude of the impulse

    Returns
    -------
    res : `np.array`
        the impulse matrix
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

    Compute a matrix of given size containing samples of a radial ramp function, raised to given
    exponent, centered at given origin.

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size, size)`. if a tuple, must be
        a 2-tuple of ints specifying the dimensions
    exponent : `float`
        the exponent of the radial ramp function.
    origin : `int`, `tuple`, or None
        the center of the image. if an int, we assume the origin is at `(origin, origin)`. if a
        tuple, must be a 2-tuple of ints specifying the origin (where `(0, 0)` is the upper left).
        if None, we assume the origin lies at the center of the matrix, `(size+1)/2`.

    Returns
    -------
    res : `np.array`
        the polar radius matrix

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


def polar_angle(size, phase=0, origin=None, direction='clockwise'):
    '''make polar angle matrix (in radians)

    Compute a matrix of given size containing samples of the polar angle (in radians,
    increasing in user-defined direction from the X-axis, ranging from -pi to pi), relative to
    given phase, about the given origin pixel.

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size, size)`. if a tuple, must be
        a 2-tuple of ints specifying the dimensions
    phase : `float`
        the phase of the polar angle function (in radians, clockwise from the X-axis)
    origin : `int`, `tuple`, or None
        the center of the image. if an int, we assume the origin is at `(origin, origin)`. if a
        tuple, must be a 2-tuple of ints specifying the origin (where `(0, 0)` is the upper left).
        if None, we assume the origin lies at the center of the matrix, `(size+1)/2`.
    direction : {'clockwise', 'counter-clockwise'}
        Whether the angle increases in a clockwise or counter-clockwise direction from
        the x-axis. The standard mathematical convention is to increase
        counter-clockwise, so that 90 degrees corresponds to the positive y-axis.

    Returns
    -------
    res : `np.array`
        the polar angle matrix

    '''
    if direction not in ['clockwise', 'counter-clockwise']:
        raise ValueError("direction must be one of {'clockwise', 'counter-clockwise'}, "
                         f"but received {direction}!")
    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    xramp, yramp = np.meshgrid(np.arange(1, size[1]+1)-origin[1],
                               np.arange(1, size[0]+1)-origin[0])
    if direction == 'counter-clockwise':
        yramp = np.flip(yramp, 0)

    res = np.arctan2(yramp, xramp)

    res = ((res+(np.pi-phase)) % (2*np.pi)) - np.pi

    return res


def disk(size, radius=None, origin=None, twidth=2, vals=(1, 0)):
    '''make a "disk" image

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size, size)`. if a tuple, must be
        a 2-tuple of ints specifying the dimensions
    radius : `float` or None
        the radius of the disk (in pixels). If None, defaults to `min(size)/4`.
    origin : `int`, `tuple`, or None
        the center of the image. if an int, we assume the origin is at `(origin, origin)`. if a
        tuple, must be a 2-tuple of ints specifying the origin (where `(0, 0)` is the upper left).
        if None, we assume the origin lies at the center of the matrix, `(size+1)/2`.
    twidth : `float`
        the width (in pixels) over which a soft threshold transition is made.
    vals : `tuple`
        2-tuple of floats containing the intensity value inside and outside the disk.

    Returns
    -------
    res : `np.array`
        the disk image matrix

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
        res = pointOp(res, Ytbl, Xtbl[0], Xtbl[1]-Xtbl[0])

    return np.asarray(res)


def gaussian(size, covariance=None, origin=None, amplitude='norm'):
    '''make a two dimensional Gaussian

    make a two dimensional Gaussian function with specified `size`, centered at a given pixel
    position, with given covariance and amplitude

    TODO - use built in scipy function

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size, size)`. if a tuple, must be
        a 2-tuple of ints specifying the dimensions
    covariance : `float`, `np.array`, or None
        the covariance of the Gaussian. If a `float`, the covariance is [[covar, 0], [0, covar]].
        If an array, must either be of shape (2,) (e.g., [1,2]) or (2,2) (e.g., [[1,0],[0,1]]). If
        it's of shape (2,), we use [[covar[0], 0], [0, covar[1]]]. If it's of shape (2,2), we use
        it as is. If None, defaults to `(min(size)/6)^2`
    origin : `int`, `tuple`, or None
        the center of the Gaussian. if an int, we assume the origin is at `(origin, origin)`. if a
        tuple, must be a 2-tuple of ints specifying the origin (where `(0, 0)` is the upper left).
        if None, we assume the origin lies at the center of the matrix, `(size+1)/2`.
    amplitude : `float` or 'norm'
        the amplitude of the Gaussian. If 'norm', will return the probability-normalized Gaussian

    Returns
    -------
    res : `np.array`
        the 2d Gaussian

    '''
    if not hasattr(size, '__iter__'):
        size = (size, size)

    if covariance is None:
        covariance = (min([size[0], size[1]]) / 6.0) ** 2
    covariance = np.asarray(covariance)

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

    zone plate is `amplitude` * cos( r^2 + `phase`)

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size, size)`. if a tuple, must be a
        2-tuple of ints specifying the dimensions
    amplitude : `float`
        the amplitude of the zone plate
    phase : `float`
        the phase of the zone plate (in radians, clockwise from the X-axis).

    Returns
    -------
    res : `np.array`
        the zone plate
    '''
    if not hasattr(size, '__iter__'):
        size = (size, size)

    res = amplitude * np.cos((np.pi / max(size)) * polar_radius(size, 2) + phase)

    return res


def angular_sine(size, harmonic=1, amplitude=1, phase=0, origin=None):
    '''make an angular sinusoidal image:

    the angular sinusoid is `amplitude` * sin(`harmonic`*theta + `phase`), where theta is the angle
    about the origin (clockwise from the X-axis).

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size, size)`. if a tuple, must be
        a 2-tuple of ints specifying the dimensions
    harmonic : `float`
        the frequency of the angular sinusoid.
    amplitude : `float`
        the amplitude of the angular sinusoid.
    phase : `float`
        the phase of the angular sinusoid. (in radians, clockwise from the X-axis).
    origin : `int`, `tuple`, or None
        the center of the image. if an int, we assume the origin is at `(origin, origin)`. if a
        tuple, must be a 2-tuple of ints specifying the origin (where `(0, 0)` is the upper left).
        if None, we assume the origin lies at the center of the matrix, `(size+1)/2`.

    Returns
    -------
    res : `np.array`
        the angular sinusoid

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
    '''make a two dimensional sinusoid

    this uses either the period and direction or a 2-tuple of frequencies. So either frequency or
    period and direction should be None, and the other should be set. If period is set, it takes
    precedence over frequency. If neither are set, we default to (2*pi) / (log2(size[0])).

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size, size)`. if a tuple, must be
        a 2-tuple of ints specifying the dimensions
    period : `float` or None
        the period of the two-dimensional sinusoid in pixels. If both `period` and `frequency` are
        None, we set `frequency` to (2*pi) / (log2(size[0])). If `period` is set, we ignore
        `frequency`
    direction : `float` or None
        the direction of the two-dimensional sinusoid, in radians, clockwise from the X-axis. If
        `period` is set and this is None, set to 0.
    frequency : `tuple` or None
        (f_x, f_y), the x and y frequency of the sinusoid in cycles per pixel. If both `period`
        and `frequency` are None, we set `frequency` to (2*pi) / (log2(size[0])).
    amplitude : `float`
        the amplitude of the sinusoid.
    phase : `float`
        the phase of the sinusoid (in radians, clockwise from the X-axis).
    origin : `int`, `tuple`, or None
        the center of the image. if an int, we assume the origin is at `(origin, origin)`. if a
        tuple, must be a 2-tuple of ints specifying the origin (where `(0, 0)` is the upper left).
        if None, we assume the origin lies at the center of the matrix, `(size+1)/2`.

    Returns
    -------
    res : `np.array`
        the two-dimensional sinusoid

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

    elif period is None and frequency is None:
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

    this uses either the period and direction or a 2-tuple of frequencies. So either frequency or
    period and direction should be None, and the other should be set. If period is set, it takes
    precedence over frequency. If neither are set, we default to (2*pi) / (log2(size[0])).

    TODO: Add duty cycle

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size, size)`. if a tuple, must be
        a 2-tuple of ints specifying the dimensions
    period : `float` or None
        the period of the square wave in pixels. If both `period` and `frequency` are None, we set
        `frequency` to (2*pi) / (log2(size[0])). If `period` is set, we ignore
        `frequency`
    direction : `float` or None
        the direction of the square wave, in radians, clockwise from the X-axis. If `period` is set
        and this is None, set to 0.
    frequency : `tuple` or None
        (f_x, f_y), the x and y frequency of the square wave in cycles per pixel. If both `period`
        and `frequency` are None, we set `frequency` to (2*pi) / (log2(size[0])).
    amplitude : `float`
        the amplitude of the sinusoid.
    phase : `float`
        the phase of the square wave (in radians, clockwise from the X-axis).
    origin : `int`, `tuple`, or None
        the center of the image. if an int, we assume the origin is at `(origin, origin)`. if a
        tuple, must be a 2-tuple of ints specifying the origin (where `(0, 0)` is the upper left).
        if None, we assume the origin lies at the center of the matrix, `(size+1)/2`.
    twidth : `float` or None
        the width of the raised-cosine edges on the bars of the grating. If None, default to
        min(2, period/3)

    Returns
    -------
    res : `np.array`
        the two-dimensional square wave
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

    elif period is None and frequency is None:
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
                  Xtbl[0], Xtbl[1]-Xtbl[0])

    return res


def pink_noise(size, fract_dim=1):
    '''make pink noise

    Make a matrix of specified size containing fractal (pink) noise with power spectral density of
    the form: 1/f^(5-2*`fract_dim`).  Image variance is normalized to 1.0.

    TODO: Verify that this  matches Mandelbrot defn of fractal dimension.
          Make this more efficient!

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size, size)`. if a tuple, must be
        a 2-tuple of ints specifying the dimensions
    fract_dim : `float`
        the fractal dimension of the pink noise

    Returns
    -------
    res : `np.array`
        the pink noise

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


def blue_noise(size, fract_dim=1):
    '''make blue noise

    Make a matrix of specified size containing blue noise with power
    spectral density of the form: f^(5-2*`fract_dim`).  Image variance
    is normalized to 1.0.

    Note that the power spectrum here is the reciprocal of the pink
    noises's power spectrum

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size,
        size)`. if a tuple, must be a 2-tuple of ints specifying the
        dimensions
    fract_dim : `float`
        the fractal dimension of the blue noise

    Returns
    -------
    res : `np.array`
        the blue noise

    '''
    if not hasattr(size, '__iter__'):
        size = (size, size)

    res = np.random.randn(size[0], size[1])
    fres = np.fft.fft2(res)

    exp = 2.5-fract_dim
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
