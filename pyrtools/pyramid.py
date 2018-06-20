import numpy

class pyramid:  # pyramid
    # properties
    pyr = []
    pyrSize = []
    pyrType = ''
    image = ''

    # constructor
    def __init__(self):
        print("please specify type of pyramid to create (Gpry, Lpyr, etc.)")
        return

    # methods
    def nbands(self):
        return len(self.pyr)

    def band(self, bandNum):
        return numpy.array(self.pyr[bandNum])
