import numpy as np
from .pyramid import Pyramid
from .c.wrapper import corrDn
from ..tools.showIm import showIm
import matplotlib.pyplot as plt
from matplotlib import cm

class GaussianPyramid(Pyramid):

    # constructor
    def __init__(self, image, height='auto', filter='binom5', filter2=None,
                 edgeType='reflect1'):
        """Gaussian pyramid
            - `image` - a 2D numpy array
            - `height` - an integer denoting number of pyramid levels desired. Defaults to `maxPyrHt`
            - `filter` - can be a string namimg a standard filter (from namedFilter()), or a
            numpy array which will be used for (separable) convolution. Default is 'binom5'.
            - `edgeType` - see class Pyramid.__init__()
            """
        super().__init__(image=image, edgeType=edgeType)
        self.initFilters(filter1=filter, filter2=filter2)
        self.initHeight(height=height)
        self.buildPyr()
        self.pyrType = 'Gaussian'

    # methods

    def initFilters(self, filter1, filter2):
        if filter2 is None: filter2 = filter1
        self.filter1 = self.parseFilter(filter1)
        self.filter2 = self.parseFilter(filter2)

    def initHeight(self, height):
        self.height = 1 + self.maxPyrHt(self.image.shape, self.filter1.shape)
        if isinstance(height, int):
            assert height <= self.height, "Error: cannot build pyramid higher than %d levels" % (self.height)
            self.height = height

    def buildNext(self, image, filt=None, edges=None):
        if filt is None: filt = self.filter1
        if edges is None: edges = self.edgeType
        imsz = image.shape
        if imsz[0] == 1:
            res = corrDn(image=image, filt=filt, edges=edges, step=(1,2))
        elif len(imsz) == 1 or imsz[1] == 1:
            res = corrDn(image=image, filt=filt, edges=edges, step=(2,1))
        else:
            tmp = corrDn(image=image, filt=filt.T, edges=edges, step=(1,2))
            res = corrDn(image=tmp, filt=filt, edges=edges, step=(2,1))
        return res

    def buildPyr(self):
        img = self.image
        if len(img.shape) == 1:
            img = img.reshape(-1, 1)

        self.pyr.append(img.copy())
        self.pyrSize.append(img.shape)
        for h in range(1,self.height):
            img = self.buildNext(img)
            self.pyr.append(img.copy())
            self.pyrSize.append(img.shape)

    def reconPrev(self, *args):
        raise Exception('Error: not necessary for Gaussian Pyramids...')

    def reconPyr(self, *args):
        raise Exception('Error: not necessary for Gaussian Pyramids...')

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
