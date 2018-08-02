import numpy as np
import os
from .pyramid import Pyramid
from .pyr_utils import LB2idx
from .filters import steerable_filters
from .c.wrapper import corrDn, upConv
from ..tools.display_tools import imshow
from matplotlib import cm

# from ..tools import JBhelpers
# from . import pyPyrUtils
# from .convolutions import *

class Spyr(Pyramid):

    #constructor
    def __init__(self, image, height='auto', filter='sp1Filters',
                 edgeType='reflect1'):
        """Steerable pyramid. image parameter is required, others are optional
        - `image` - a 2D numpy array
        - `height` - an integer denoting number of pyramid levels desired.  'auto' (default) uses
        maxPyrHt from pyPyrUtils.
        - `filter` - The name of one of the steerable pyramid filters in pyPyrUtils:
        `'sp0Filters'`, `'sp1Filters'`, `'sp3Filters'`, `'sp5Filters'`.  Default is `'sp1Filters'`.
        - `edgeType` - see class Pyramid.__init__()
        """
        super().__init__(image=image, edgeType=edgeType)

        self.filt = steerable_filters(filter)
        self.pyrType = 'Steerable'

        filters = self.filt # temporary hack...
        harmonics = filters['harmonics']
        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']

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

        im = self.image
        im_sz = im.shape
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

    # methods

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

    def reconPyr(self, *args):
        # defaults

        if len(args) > 0:
            filters = steerable_filters(args[0])
        else:
            filters = steerable_filters('sp1Filters')

        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']
        # assume square filters  -- start of buildSpyrLevs
        bfiltsz = int(np.floor(np.sqrt(bfilts.shape[0])))

        if len(args) > 1:
            edges = args[1]
        else:
            edges = 'reflect1'

        if len(args) > 2:
            levs = args[2]
        else:
            levs = 'all'

        if len(args) > 3:
            bands = args[3]
        else:
            bands = 'all'

        #---------------------------------------------------------

        maxLev = 2 + self.spyrHt()
        if levs == 'all':
            levs = np.array(list(range(maxLev)))
        else:
            levs = np.array(levs)
            if (levs < 0).any() or (levs >= maxLev).any():
                raise Exception("level numbers must be in the range [0, %d]." % (maxLev-1))
            else:
                levs = np.array(levs)
                if len(levs) > 1 and levs[0] < levs[1]:
                    levs = levs[::-1]  # we want smallest first
        if bands == 'all':
            bands = np.array(list(range(self.numBands())))
        else:
            bands = np.array(bands)
            if (bands < 0).any() or (bands > bfilts.shape[1]).any():
                raise Exception("band numbers must be in the range [0, %d]." % (self.numBands()-1))
            else:
                bands = np.array(bands)

        # make a list of all pyramid layers to be used in reconstruction
        Nlevs = self.spyrHt()
        Nbands = self.numBands()

        reconList = []  # pyr indices used in reconstruction

        for lev in levs:
            if lev == 0:
                reconList.append(0)
            elif lev == Nlevs+1:
                # number of levels times number of bands + top and bottom
                #   minus 1 for 0 starting index
                reconList.append( (Nlevs*Nbands) + 2 - 1)
            else:
                for band in bands:
                    reconList.append( ((lev-1) * Nbands) + band + 1)

        reconList = np.sort(reconList)[::-1]  # deepest level first

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

    def showPyr(self, prange = 'auto2', gap = 1, scale = 2, disp = 'qt'):
        ht = int(self.spyrHt())
        nind = len(self.pyr)
        nbands = self.numBands()

        ## Auto range calculations:
        if prange == 'auto1':
            prange = np.ones((nind,1))
            band = self.pyrHigh()
            mn = np.amin(band)
            mx = np.amax(band)
            for lnum in range(1,ht+1):
                for bnum in range(nbands):
                    idx = pyPyrUtils.LB2idx(lnum, bnum, ht+2, nbands)
                    band = self.band(idx)/(np.power(scale,lnum))
                    prange[(lnum-1)*nbands+bnum+1] = np.power(scale,lnum-1)
                    bmn = np.amin(band)
                    bmx = np.amax(band)
                    mn = min([mn, bmn])
                    mx = max([mx, bmx])
            prange = np.outer(prange, np.array([mn, mx]))
            band = self.pyrLow()
            mn = np.amin(band)
            mx = np.amax(band)
            prange[nind-1,:] = np.array([mn, mx])
        elif prange == 'indep1':
            prange = np.zeros((nind,2))
            for bnum in range(nind):
                band = self.band(bnum)
                mn = band.min()
                mx = band.max()
                prange[bnum,:] = np.array([mn, mx])
        elif prange == 'auto2':
            prange = np.ones(nind)
            band = self.pyrHigh()
            sqsum = np.sum( np.power(band, 2) )
            numpixels = band.shape[0] * band.shape[1]
            for lnum in range(1,ht+1):
                for bnum in range(nbands):
                    band = self.band(LB2idx(lnum, bnum, ht+2, nbands))
                    band = band / np.power(scale,lnum-1)
                    sqsum += np.sum( np.power(band, 2) )
                    numpixels += band.shape[0] * band.shape[1]
                    prange[(lnum-1)*nbands+bnum+1] = np.power(scale, lnum-1)
            stdev = np.sqrt( sqsum / (numpixels-1) )
            prange = np.outer(prange, np.array([-3*stdev, 3*stdev]))
            band = self.pyrLow()
            av = np.mean(band)
            stdev = np.sqrt( np.var(band) )
            prange[nind-1,:] = np.array([av-2*stdev, av+2*stdev])
        elif prange == 'indep2':
            prange = np.zeros((nind,2))
            for bnum in range(nind-1):
                band = self.band(bnum)
                stdev = np.sqrt( np.var(band) )
                prange[bnum,:] = np.array([-3*stdev, 3*stdev])
            band = self.pyrLow()
            av = np.mean(band)
            stdev = np.sqrt( np.var(band) )
            prange[nind-1,:] = np.array([av-2*stdev, av+2*stdev])
        elif isinstance(prange, str):
            raise Exception("Bad RANGE argument: %s'" % (prange))
        elif prange.shape[0] == 1 and prange.shape[1] == 2:
            scales = np.power(scale, list(range(ht)))
            scales = np.outer( np.ones((nbands,1)), scales )
            scales = np.array([1, scales, np.power(scale, ht)])
            prange = np.outer(scales, prange)
            band = self.pyrLow()
            prange[nind,:] += np.mean(band) - np.mean(prange[nind,:])

        colormap = cm.Greys_r

        # compute positions of subbands
        llpos = np.ones((nind,2));

        if nbands == 2:
            ncols = 1
            nrows = 1 # pe: nrows was incorrectely set to 2, changed it to 1 2/18
        else:
            ncols = (nbands+1)//2
            nrows = nbands//2

        a = np.array(list(range(1-nrows, 1)))
        b = np.zeros((1,ncols))[0]
        ab = np.concatenate((a,b))
        c = np.zeros((1,nrows))[0]
        d = list(range(-1, -ncols-1, -1))
        cd = np.concatenate((c,d))
        relpos = np.vstack((ab,cd)).T

        if nbands > 1:
            mvpos = np.array([-1, -1]).reshape(1,2)
        else:
            mvpos = np.array([0, -1]).reshape(1,2)
        basepos = np.array([0, 0]).reshape(1,2)

        for lnum in range(1,ht+1):
            ind1 = (lnum-1)*nbands + 1
            sz = np.array(self.pyrSize[ind1]) + gap
            basepos = basepos + mvpos * sz
            if nbands < 5:         # to align edges
                sz += gap * (ht-lnum)
            # print(relpos)
            # print(sz)
            # print(nbands)
            # print(basepos)
            llpos[ind1:ind1+nbands, :] = np.dot(relpos, np.diag(sz)) + ( np.ones((nbands,1)) * basepos )

        # lowpass band
        sz = np.array(self.pyrSize[nind-1]) + gap
        basepos += mvpos * sz
        llpos[nind-1,:] = basepos

        # make position list positive, and allocate appropriate image:
        llpos = llpos - ((np.ones((nind,2)) * np.amin(llpos, axis=0)) + 1) + 1
        llpos[0,:] = np.array([1, 1])
        # we want to cast it as ints, since we'll be using these as indices
        llpos = llpos.astype(int)
        urpos = llpos + self.pyrSize
        d_im = np.zeros((np.amax(urpos), np.amax(urpos)))

        # paste bands into image, (im-r1)*(nshades-1)/(r2-r1) + 1.5
        nshades = 64;

        for bnum in range(1,nind):
            mult = (nshades-1) / (prange[bnum,1]-prange[bnum,0])
            d_im[llpos[bnum,0]:urpos[bnum,0],
                 llpos[bnum,1]:urpos[bnum,1]] = mult * self.band(bnum) + (1.5-mult*prange[bnum,0])

        if disp == 'qt':
            imshow(d_im[:self.pyrSize[0][0]*2,:])
        # elif disp == 'nb':
        #     JBhelpers.showIm(d_im[:self.pyrSize[0][0]*2,:])
