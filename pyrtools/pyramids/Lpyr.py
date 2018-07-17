import numpy as np
from .pyramid import Pyramid
from .c.wrapper import corrDn, upConv
from .namedFilter import namedFilter
from ..tools.showIm import showIm
import matplotlib.pyplot as plt
from matplotlib import cm
# from ..tools import JBhelpers

class LaplacianPyramid(Pyramid):

    # constructor
    def __init__(self, image, pyrType='Laplacian', edgeType='reflect1',
                 height='auto', filter1='binom5', filter2=None):
        """Laplacian pyramid

            - `image` - a 2D numpy array
            - `height` - an integer denoting number of pyramid levels desired. Defaults to `maxPyrHt`
            - `filter1` - can be a string namimg a standard filter (from namedFilter()), or a
            numpy array which will be used for (separable) convolution. Default is 'binom5'.
            - `filter2` - specifies the "expansion" filter (default = filter1).
            - `edges` - see class Pyramid.__init__()
            """
        super().__init__(image=image, pyrType=pyrType, edgeType=edgeType)


        if filter2 is None:
            filter2 = filter1
        self.filter1 = self.parseFilter(filter1)
        self.filter2 = self.parseFilter(filter2)

        maxHeight = 1 + self.maxPyrHt(self.image.shape, self.filter1.shape)

        if isinstance(height, str) and height == "auto":
            self.height = maxHeight
        else:
            self.height = height
            if self.height > maxHeight:
                raise Exception("Error: cannot build pyramid higher than %d levels" % (maxHeight))

        # make pyramid
        pyrCtr = 0
        im = np.array(self.image).astype(float)
        if len(im.shape) == 1:
            im = im.reshape(im.shape[0], 1)
        los = {}
        los[self.height] = im
        # compute low bands
        for ht in range(self.height-1,0,-1):
            im_sz = im.shape
            if im_sz[0] == 1:
                lo2 = corrDn(image = im, filt = self.filter1, edges = self.edgeType,
                             step = (1,2))
                #lo2 = np.array(lo2)
            elif len(im_sz) == 1 or im_sz[1] == 1:
                lo2  = corrDn(image = im, filt = self.filter1, edges = self.edgeType,
                              step = (2,1))
                #lo2 = np.array(lo2)
            else:
                lo = corrDn(image = im, filt = self.filter1.T, edges = self.edgeType,
                            step = (1,2), start = (0,0))
                #lo = np.array(lo)
                lo2 = corrDn(image = lo, filt = self.filter1, edges = self.edgeType,
                             step = (2,1), start = (0,0))
                #lo2 = np.array(lo2)

            los[ht] = lo2

            im = lo2

        # adjust shape if 1D if needed
        self.pyr.append(lo2.copy())
        self.pyrSize.append(lo2.shape)

        # compute hi bands
        im = self.image
        for ht in range(self.height, 1, -1):
            im = los[ht-1]
            im_sz = los[ht-1].shape
            if len(im_sz) == 1 or im_sz[1] == 1:
                hi2 = upConv(image = im, filt = self.filter2.T, edges = self.edgeType,
                             step = (1,2), stop = (los[ht].shape[1],
                                                   los[ht].shape[0])).T
            elif im_sz[0] == 1:
                hi2 = upConv(image = im, filt = self.filter2.T, edges = self.edgeType,
                             step = (2,1), stop = (los[ht].shape[1],
                                                   los[ht].shape[0])).T
            else:
                hi = upConv(image = im, filt = self.filter2, edges = self.edgeType,
                            step = (2,1), stop = (los[ht].shape[0], im_sz[1]))
                hi2 = upConv(image = hi, filt = self.filter2.T, edges = self.edgeType,
                             step = (1,2), stop = (los[ht].shape[0],
                                                   los[ht].shape[1]))

            hi2 = los[ht] - hi2
            self.pyr.insert(pyrCtr, hi2.copy())
            self.pyrSize.insert(pyrCtr, hi2.shape)
            pyrCtr += 1


    # methods
    def parseFilter(self, filter):
        if isinstance(filter, str):
            filter = namedFilter(filter)
        filter = np.array(filter)

        if filter.size > max(filter.shape):
            raise Exception("Error: filter should be a 1D (i.e., a vector)")

        # when the first dimension of the image is 1, we need the filter to have shape (1, x)
        # instead of the normal (x, 1) or we get a segfault during corrDn / upConv. That's because
        # we need to match the filter to the image dimensions
        if filter.ndim == 1 or self.image.shape[0] == 1:
            filter = filter.reshape(1,-1)
        return filter

    # return concatenation of all levels of 1d pyramid
    def catBands(self, *args):
        outarray = np.array([]).reshape((1,0))
        for i in range(self.height):
            tmp = self.band(i).T
            outarray = np.concatenate((outarray, tmp), axis=1)
        return outarray

    def set(self, band, element, value):
        """set a pyramid value

        element must be a tuple, others are single numbers
        """
        self.pyr[band][element[0]][element[1]] = value

    def reconPyr(self, *args):
        if len(args) > 0:
            if not isinstance(args[0], str):
                levs = np.array(args[0])
            else:
                levs = args[0]
        else:
            levs = 'all'

        if len(args) > 1:
            filter2 = args[1]
        else:
            filter2 = 'binom5'

        if len(args) > 2:
            edges = args[2]
        else:
            edges = 'reflect1';

        maxLev = self.height

        if isinstance(levs, str) and levs == 'all':
            levs = list(range(0,maxLev))
        else:
            if (levs > maxLev-1).any():
                raise Exception("level numbers must be in the range [0, %d]." % (maxLev-1))

        if isinstance(filter2, str):
            filter2 = namedFilter(filter2)
        else:
            if len(filter2.shape) == 1:
                filter2 = filter2.reshape(1, len(filter2))

        res = []
        lastLev = -1
        for lev in range(maxLev-1, -1, -1):
            if lev in levs and len(res) == 0:
                res = self.band(lev)
            elif len(res) != 0:
                res_sz = res.shape
                new_sz = self.band(lev).shape
                filter2_sz = filter2.shape
                if res_sz[0] == 1:
                    hi2 = upConv(image = res, filt = filter2, edges = edges,
                                 step = (2,1), stop = (new_sz[1], new_sz[0])).T
                elif res_sz[1] == 1:
                    hi2 = upConv(image = res, filt = filter2.T, edges = edges,
                                 step = (1,2), stop = (new_sz[1], new_sz[0])).T
                else:
                    hi = upConv(image = res, filt = filter2, edges = edges,
                                step = (2,1), stop = (new_sz[0], res_sz[1]))
                    hi2 = upConv(image = hi, filt = filter2.T, edges = edges,
                                 step = (1,2), stop = (new_sz[0], new_sz[1]))
                if lev in levs:
                    bandIm = self.band(lev)
                    bandIm_sz = bandIm.shape
                    res = hi2 + bandIm
                else:
                    res = hi2
        return res

    def pyrLow(self):
        return np.array(self.band(self.height-1))

    # options for disp are 'qt' and 'nb'
    def showPyr(self, pRange = None, gap = 1, scale = None, disp = 'qt'):
        if ( len(self.band(0).shape) == 1 or self.band(0).shape[0] == 1 or
             self.band(0).shape[1] == 1 ):
            oned = 1
        else:
            oned = 0

        if pRange is None and oned == 1:
            pRange = 'auto1'
        elif pRange is None and oned == 0:
            pRange = 'auto2'

        if scale is None and oned == 1:
            scale = np.sqrt(2)
        elif scale is None and oned == 0:
            scale = 2

        nind = self.height

        # auto range calculations
        if pRange == 'auto1':
            pRange = np.zeros((nind,1))
            mn = 0.0
            mx = 0.0
            for bnum in range(nind):
                band = self.band(bnum)
                band /= np.power(scale, bnum-1)
                pRange[bnum] = np.power(scale, bnum-1)
                bmn = np.amin(band)
                bmx = np.amax(band)
                mn = np.amin([mn, bmn])
                mx = np.amax([mx, bmx])
            if oned == 1:
                pad = (mx-mn)/12       # magic number
                mn -= pad
                mx += pad
            pRange = np.outer(pRange, np.array([mn, mx]))
            band = self.pyrLow()
            mn = np.amin(band)
            mx = np.amax(band)
            if oned == 1:
                pad = (mx-mn)/12
                mn -= pad
                mx += pad
            pRange[nind-1,:] = [mn, mx];
        elif pRange == 'indep1':
            pRange = np.zeros((nind,1))
            for bnum in range(nind):
                band = self.band(bnum)
                mn = np.amin(band)
                mx = np.amax(band)
                if oned == 1:
                    pad = (mx-mn)/12;
                    mn -= pad
                    mx += pad
                pRange[bnum,:] = np.array([mn, mx])
        elif pRange == 'auto2':
            pRange = np.zeros((nind,1))
            sqsum = 0
            numpixels = 0
            for bnum in range(0, nind-1):
                band = self.band(bnum)
                band /= np.power(scale, bnum)
                sqsum += np.sum( np.power(band, 2) )
                numpixels += np.prod(band.shape)
                pRange[bnum,:] = np.power(scale, bnum)
            stdev = np.sqrt( sqsum / (numpixels-1) )
            pRange = np.outer( pRange, np.array([-3*stdev, 3*stdev]) )
            band = self.pyrLow()
            av = np.mean(band)
            stdev = np.std(band)
            pRange[nind-1,:] = np.array([av-2*stdev, av+2*stdev]);
        elif pRange == 'indep2':
            pRange = np.zeros(nind,2)
            for bnum in range(0,nind-1):
                band = self.band(bnum)
                stdev = np.std(band)
                pRange[bnum,:] = np.array([-3*stdev, 3*stdev])
            band = self.pyrLow()
            av = np.mean(band)
            stdev = np.std(band)
            pRange[nind,:] = np.array([av-2*stdev, av+2*stdev])
        elif isinstance(pRange, str):
            print("Error: band range argument: %s" % (pRange))
            return
        elif pRange.shape[0] == 1 and pRange.shape[1] == 2:
            scales = np.power( np.array( list(range(0,nind)) ), scale)
            pRange = np.outer( scales, pRange )
            band = self.pyrLow()
            pRange[nind,:] = ( pRange[nind,:] + np.mean(band) -
                               np.mean(pRange[nind,:]) )

        # draw
        if oned == 1:   # 1D signal
            fig = plt.figure()
            for bnum in range(nind):
                band = self.band(bnum)
                plt.subplot(len(self.pyrSize), 1, bnum+1)
                plt.plot(band.T)
            plt.tight_layout()
            plt.show()

        else:
            colormap = cm.Greys_r
            # skipping background calculation. needed?

            # compute positions of subbands:
            llpos = np.ones((nind, 2)).astype(float)
            dirr = np.array([-1.0, -1.0])
            ctr = np.array([self.band(0).shape[0]+1+gap, 1]).astype(float)
            sz = np.array([0.0, 0.0])
            for bnum in range(nind):
                prevsz = sz
                sz = self.band(bnum).shape

                # determine center position of new band:
                ctr = ( ctr + gap*dirr/2.0 + dirr *
                        np.floor( (prevsz+(dirr<0).astype(int))/2.0 ) )
                dirr = np.dot(dirr,np.array([ [0, -1], [1, 0] ])) # ccw rotation
                ctr = ( ctr + gap*dirr/2 + dirr *
                        np.floor( (sz+(dirr<0).astype(int)) / 2.0) )
                llpos[bnum,:] = ctr - np.floor(np.array(sz))/2.0
            # make position list positive, and allocate appropriate image
            llpos = llpos - np.ones((nind,1))*np.min(llpos)
            # we want to cast it as ints, since we'll be using these as indices
            llpos = llpos.astype(int)
            pind = list(range(self.height))
            for i in pind:
                pind[i] = self.band(i).shape
            urpos = llpos + pind
            d_im = np.ones((np.max(urpos), np.max(urpos))) * 255

            # paste bands into image, (im-r1)*(nshades-1)/(r2-r1) + 1.5
            nshades = 256
            for bnum in range(nind):
                mult = (nshades-1) / (pRange[bnum,1]-pRange[bnum,0])
                d_im[llpos[bnum,0]:urpos[bnum,0], llpos[bnum,1]:urpos[bnum,1]]=(
                    mult*self.band(bnum) + (1.5-mult*pRange[bnum,0]) )

            # FIX: need a mode to switch between above and below display
            if disp == 'qt':
                showIm(d_im[:self.band(0).shape[0]][:])
            # elif disp == 'nb':
            #     JBhelpers.showIm(d_im[:self.band(0).shape[0]][:])
