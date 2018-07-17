import numpy as np
from .namedFilter import namedFilter

class Pyramid:  # Pyramid base class

    def __init__(self, image, pyrType, edgeType):
        ''' - `edgeType` - specifies edge-handling.  Options are:
            * `'circular'` - circular convolution
            * `'reflect1'` - reflect about the edge pixels
            * `'reflect2'` - reflect, doubling the edge pixels
            * `'repeat'` - repeat the edge pixels
            * `'zero'` - assume values of zero outside image boundary
            * `'extend'` - reflect and invert
            * `'dont-compute'` - zero output when filter overhangs imput boundaries.
            '''
        self.image = image
        self.pyrType = pyrType
        self.edgeType = edgeType
        self.pyr = []
        self.pyrSize = []

    # methods
    def nbands(self):
        return len(self.pyr)

    def band(self, bandNum):
        if bandNum < len(self.pyr):
            return np.array(self.pyr[bandNum])
        raise IndexError('band number is out of range')

# maxPyrHt
# showPyr
