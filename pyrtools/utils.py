import numpy as np

# not really necessary as a new function
def strictly_decreasing(np_array):
    ''' are all elements of list strictly decreasing '''
    return np.all(np.diff(np_array) < 0)

# not really necessary as a new function
def shift(np_array, offset):
    ''' Circular shift 2D matrix samples by OFFSET (a [Y,X] 2-tuple),
        such that  RES(POS) = MTX(POS-OFFSET).  '''
    return np.roll(np_array, offset)

def matlab_round(np_array):
    ''' round equivalent to matlab function, which rounds .5 away from zero
        used in matlab_histo so we can unit test against matlab code.
        But numpy.round() would rounds .5 to nearest even number
        e.g. numpy.round(0.5) = 0, matlab_round(0.5) = 1
        e.g. numpy.round(2.5) = 2, matlab_round(2.5) = 3
        '''
    (fracPart, intPart) = np.modf(np_array)
    return intPart + (np.abs(fracPart) >= 0.5) * np.sign(fracPart)

def clip(np_array, mini_or_range = 0.0, maxi = 1.0):
    ''' [RES] = clip(np_array, mini_or_range = 0.0, maxi = 1.0):

        A wrapper of numpy.np that handles multiple ways to pass parameters
        and default values [mini=0.0, maxi=1.0]'''

    if isinstance(mini_or_range, (int, float)):
        mini = mini_or_range
    else: # a range is provided
        mini = mini_or_range[0]
        maxi = mini_or_range[1]

    if maxi < mini:
        raise Exception('Error: maxVal cannot be less than minVal!')

    return np.clip(np_array, mini, maxi)
