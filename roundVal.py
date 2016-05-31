import math
import numpy

def roundVal(val):
    ''' for use with round() - returns the rounded integer part of val  '''

    (fracPart, intPart) = math.modf(val)
    if numpy.abs(fracPart) >= 0.5:
        if intPart >= 0:
            intPart += 1
        else:
            intPart -= 1
    return intPart
