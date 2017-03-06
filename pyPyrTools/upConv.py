import numpy
import ctypes
import os

# load the C library
lib = ctypes.cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) +
                              '/wrapConv.so')

def upConv(image = None, filt = None, edges = 'reflect1', step = (1,1), 
           start = (0,0), stop = None, result = None):

    if image is None or filt is None:
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

    if stop is None and result is None:
        stop = (image.shape[0]*step[0], image.shape[1]*step[1])
        stop = (stop[0], stop[1])
    elif stop is None:
        stop = (result.shape[0], result.shape[1])

    if result is None:
        result = numpy.zeros((stop[1], stop[0]))
    else:
        result = numpy.array(result.copy())

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
