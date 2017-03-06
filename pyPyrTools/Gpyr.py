from Lpyr import Lpyr
from namedFilter import namedFilter
from maxPyrHt import maxPyrHt
import numpy
from corrDn import corrDn

class Gpyr(Lpyr):
    filt = ''
    edges = ''
    height = ''

    # constructor
    def __init__(self, *args):    # (image, height, filter, edges)
        self.pyrType = 'Gaussian'
        if len(args) < 1:
            print "pyr = Gpyr(image, height, filter, edges)"
            print "First argument (image) is required"
            return
        else:
            self.image = args[0]

        if len(args) > 2:
            filt = args[2]
            if not (filt.shape == 1).any():
                print "Error: filt should be a 1D filter (i.e., a vector)"
                return
        else:
            print "no filter set, so filter is binom5"
            filt = namedFilter('binom5')
            if self.image.shape[0] == 1:
                filt = filt.reshape(1,5)
            else:
                filt = filt.reshape(5,1)

        maxHeight = 1 + maxPyrHt(self.image.shape, filt.shape)

        if len(args) > 1:
            if args[1] is "auto":
                self.height = maxHeight
            else:
                self.height = args[1]
                if self.height > maxHeight:
                    print ( "Error: cannot build pyramid higher than %d levels"
                            % (maxHeight) )
                    return
        else:
            self.height = maxHeight

        if len(args) > 3:
            edges = args[3]
        else:
            edges = "reflect1"

        # make pyramid
        self.pyr = []
        self.pyrSize = []
        pyrCtr = 0
        im = numpy.array(self.image).astype(float)

        if len(im.shape) == 1:
            im = im.reshape(im.shape[0], 1)

        self.pyr.append(im.copy())
        self.pyrSize.append(im.shape)
        pyrCtr += 1

        for ht in range(self.height-1,0,-1):
            im_sz = im.shape
            filt_sz = filt.shape
            if im_sz[0] == 1:
                lo2 = corrDn(image = im, filt = filt, step = (1,2))
                #lo2 = numpy.array(lo2)
            elif len(im_sz) == 1 or im_sz[1] == 1:
                lo2 = corrDn(image = im, filt = filt, step = (2,1))
                #lo2 = numpy.array(lo2)
            else:
                lo = corrDn(image = im, filt = filt.T, step = (1,2),
                            start = (0,0))
                #lo = numpy.array(lo)
                lo2 = corrDn(image = lo, filt = filt, step = (2,1),
                             start = (0,0))
                #lo2 = numpy.array(lo2)                

            self.pyr.append(lo2.copy())
            self.pyrSize.append(lo2.shape)

            pyrCtr += 1

            im = lo2
