import numpy
from .roundVal import roundVal

def round(arr):
    ''' round equivalent to matlab function
        used in histo so we can unit test against matlab code
        numpy version rounds to closest even number to remove bias  '''

    if isinstance(arr, (int, float)):
        arr = roundVal(arr)
    else:
        for i in range(len(arr)):
            arr[i] = roundVal(arr[i])

    return arr
