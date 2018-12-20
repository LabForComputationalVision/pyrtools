
import numpy as np
from .pyramid import Pyramid
from .filters import steerable_filters
from .c.wrapper import corrDn, upConv

from matplotlib import cm
from .pyr_utils import LB2idx


class SteerablePyramidOld(Pyramid):
    def __init__(self, image, height='auto', filt='sp1Filters',
                 edgeType='reflect1'):
        """Steerable pyramid. image parameter is required, others are optional
        - `image` - a 2D numpy array
        - `height` - an integer denoting number of pyramid levels desired.  'auto' (default) uses
        maxPyrHt from pyPyrUtils.
        - `filter` - The name of one of the steerable pyramid filters in pyPyrUtils:
        `'sp0Filters'`, `'sp1Filters'`, `'sp3Filters'`, `'sp5Filters'`.  Default is `'sp1Filters'`.
        - `edgeType` - see class Pyramid.__init__()
        """
        super().__init__(image=image, edge_type=edgeType)

        self.filt = steerable_filters(filt)
        self.pyrType = 'Steerable'

        filters   = self.filt # temporary hack...
        harmonics = filters['harmonics']
        lo0filt   = filters['lo0filt']
        hi0filt   = filters['hi0filt']
        lofilt    = filters['lofilt']
        bfilts    = filters['bfilts']
        steermtx  = filters['mtx']

        max_ht = self.maxPyrHt(self.image.shape, lofilt.shape)
        if height == 'auto':
            ht = max_ht
        elif height > max_ht:
            raise Exception("cannot build pyramid higher than %d levels." % (max_ht))
        else:
            ht = int(height)

        nbands = bfilts.shape[1]

        self.pyr = []
        self.pyrSize = []
        for n in range((ht*nbands)+2):
            self.pyr.append([])
            self.pyrSize.append([])

        im     = self.image
        im_sz  = im.shape
        pyrCtr = 0

        hi0 = corrDn(image = im, filt = hi0filt, edges = self.edgeType);

        self.pyr[pyrCtr] = hi0
        self.pyrSize[pyrCtr] = hi0.shape

        pyrCtr += 1

        lo = corrDn(image = im, filt = lo0filt, edges = self.edgeType)
        for i in range(ht):
            lo_sz = lo.shape
            # assume square filters  -- start of buildSpyrLevs
            bfiltsz = int(np.floor(np.sqrt(bfilts.shape[0])))

            for b in range(bfilts.shape[1]):
                filt = bfilts[:,b].reshape(bfiltsz,bfiltsz).T
                band = corrDn(image = lo, filt = filt, edges = self.edgeType)
                self.pyr[pyrCtr] = np.array(band)
                self.pyrSize[pyrCtr] = (band.shape[0], band.shape[1])
                pyrCtr += 1

            lo = corrDn(image = lo, filt = lofilt, edges = self.edgeType, step = (2,2))

        self.pyr[pyrCtr] = np.array(lo)
        self.pyrSize[pyrCtr] = lo.shape


    def spyrLev(self, lev):
        if lev < 0 or lev > self.spyrHt()-1:
            raise Exception('level parameter must be between 0 and %d!' % (self.spyrHt()-1))

        levArray = []
        for n in range(self.numBands()):
            levArray.append(self.spyrBand(lev, n))
        levArray = np.array(levArray)

        return levArray

    def spyrBand(self, lev, band):
        if lev < 0 or lev > self.spyrHt()-1:
            raise Exception('level parameter must be between 0 and %d!' % (self.spyrHt()-1))
        if band < 0 or band > self.numBands()-1:
            raise Exception('band parameter must be between 0 and %d!' % (self.numBands()-1))

        return self.band( ((lev*self.numBands())+band)+1 )

    def spyrHt(self):
        if len(self.pyrSize) > 2:
            spHt = (len(self.pyrSize)-2) // self.numBands()
        else:
            spHt = 0
        return spHt

    def numBands(self):
        if len(self.pyrSize) == 2:
            return 0
        else:
            b = 2
            while ( b <= len(self.pyrSize) and
                    self.pyrSize[b] == self.pyrSize[1] ):
                b += 1
            return b-1

    def pyrLow(self):
        return np.array(self.band(len(self.pyrSize)-1))

    def pyrHigh(self):
        return np.array(self.band(0))

    def reconPyr(self, filt=None, edgeType=None, levs='all', bands='all'):
        # defaults

        if filt is None:
            filters = self.filt
        else:
            filters = steerable_filters(filt)

        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']
        # assume square filters  -- start of buildSpyrLevs
        bfiltsz = int(np.floor(np.sqrt(bfilts.shape[0])))

        if edgeType is None:
            edges = self.edgeType
        else:
            edges = edgeType

        maxLev = 2 + self.spyrHt()
        if isinstance(levs, str) and levs == 'all':
            levs = np.arange(maxLev)
        else:
            levs = np.array(levs)
            assert (levs >= 0).all(), "Error: level numbers must be larger than 0."
            assert (levs < maxLev).all(), "Error: level numbers must be in the range [0, %d]" % (maxLev - 1)
            levs = np.sort(levs) # we want smallest first

        if isinstance(bands, str) and bands == "all":
            bands = np.arange(self.numBands())
        else:
            bands = np.array(bands)
            assert (bands >= 0).all(), "Error: band numbers must be larger than 0."
            assert (bands < bfilts.shape[1]).all(), "Error: band numbers must be in the range [0, %d]" % (self.numBands() - 1)

        # make a list of all pyramid layers to be used in reconstruction
        Nlevs = self.spyrHt()
        Nbands = self.numBands()

        reconList = []  # pyr indices used in reconstruction

        for lev in levs:
            if lev == 0: # highpass residual
                reconList.append(0)
            elif lev == Nlevs+1: # lowpass residual
                # number of levels times number of bands + top and bottom
                #   minus 1 for 0 starting index
                reconList.append( (Nlevs*Nbands) + 2 - 1)
            else:
                for band in bands:
                    reconList.append( ((lev-1) * Nbands) + band + 1)

        reconList = np.sort(reconList)[::-1]  # deepest level first

        # print(reconList) # TODO recon should still work without the hipass

        # initialize reconstruction
        if len(self.pyr)-1 in reconList:
            recon = np.array(self.pyr[len(self.pyrSize)-1])
        else:
            recon = np.zeros(self.pyr[len(self.pyrSize)-1].shape)

        # recursive subsystem
        # we need to loop over recursive subsystem pairs
        for level in range(Nlevs):
            maxLevIdx = ((maxLev-2) * Nbands) + 1
            resSzIdx = maxLevIdx - (level * Nbands) - 1
            recon = upConv(image = recon, filt = lofilt, edges = edges,
                           step = (2,2), start = (0,0),
                           stop = self.pyrSize[resSzIdx])

            bandImageIdx = 1 + (((Nlevs-1)-level) * Nbands)
            for band in range(Nbands-1,-1,-1):
                if bandImageIdx in reconList:
                    filt = bfilts[:,(Nbands-1)-band].reshape(bfiltsz,
                                                             bfiltsz,
                                                             order='F')

                    recon += upConv(image = self.pyr[bandImageIdx],
                                    filt = filt, edges = edges,
                                    stop = (self.pyrSize[bandImageIdx][0],
                                            self.pyrSize[bandImageIdx][1]))
                    bandImageIdx += 1

        # apply lo0filt
        sz = recon.shape
        recon = upConv(image = recon, filt = lo0filt, edges = edges, stop = sz)

        # apply hi0filt if needed
        if 0 in reconList:
            recon += upConv(image = self.pyr[0], filt = hi0filt, edges = edges,
                            start = (0,0), step = (1,1), stop = recon.shape)

        return recon
