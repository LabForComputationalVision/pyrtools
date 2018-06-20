import numpy
from .mkR import mkR
from .var2 import var2

def mkFract(*args):
    ''' Make a matrix of dimensions SIZE (a [Y X] 2-vector, or a scalar)
        containing fractal (pink) noise with power spectral density of the
        form: 1/f^(5-2*FRACT_DIM).  Image variance is normalized to 1.0.
        FRACT_DIM defaults to 1.0
        Eero Simoncelli, 6/96. Ported to Python by Rob Young, 5/14.

        TODO: Verify that this  matches Mandelbrot defn of fractal dimension.
              Make this more efficient!   '''

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

    if abs(fres.imag).max() > 1e-10:
        print('Symmetry error in creating fractal')
    else:
        res = numpy.real(fres)
        res = res / numpy.sqrt(var2(res))

    return res
