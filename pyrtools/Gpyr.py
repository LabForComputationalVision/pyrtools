from .Lpyr import Lpyr
from .namedFilter import namedFilter
from .maxPyrHt import maxPyrHt
import numpy
from .corrDn import corrDn

class Gpyr(Lpyr):
    filt = ''
    edges = ''
    height = ''

    # constructor
    def __init__(self, image, height='auto', filt='binom5', edges='reflect1'):
        self.pyrType = 'Gaussian'
        self.image = image

        if isinstance(filt, str):
            self.filt = namedFilter(filt)
        else:
            self.filt = filt
        if not (numpy.array(self.filt.shape) == 1).any():
            raise Exception("filt should be a 1D filter (i.e., a vector)")

        # when the first dimension of the image is 1, we need the filter to have shape (1, x)
        # instead of the normal (x, 1) or we get a segfault during corrDn / upConv. That's because
        # we need to match the filter to the image dimensions
        if self.image.shape[0] == 1:
            self.filt = self.filt.reshape(1, max(self.filt.shape))

        maxHeight = 1 + maxPyrHt(self.image.shape, self.filt.shape)

        if height == "auto":
            self.height = maxHeight
        else:
            self.height = height
            if self.height > maxHeight:
                raise Exception("Cannot build pyramid higher than %d levels" % (maxHeight))

        self.edges = edges

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
            filt_sz = self.filt.shape
            if im_sz[0] == 1:
                lo2 = corrDn(image=im, filt=self.filt, step=(1, 2))
                #lo2 = numpy.array(lo2)
            elif len(im_sz) == 1 or im_sz[1] == 1:
                lo2 = corrDn(image=im, filt=self.filt, step=(2, 1))
                #lo2 = numpy.array(lo2)
            else:
                lo = corrDn(image=im, filt=self.filt.T, step=(1, 2),
                            start=(0, 0))
                #lo = numpy.array(lo)
                lo2 = corrDn(image=lo, filt=self.filt, step=(2, 1),
                             start=(0, 0))
                #lo2 = numpy.array(lo2)                

            self.pyr.append(lo2.copy())
            self.pyrSize.append(lo2.shape)

            pyrCtr += 1

            im = lo2
