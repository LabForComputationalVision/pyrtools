from .pyramid import pyramid
import numpy
from .get_filter import get_filter
import os
from .maxPyrHt import maxPyrHt
from .corrDn import corrDn
import math
from .LB2idx import LB2idx
import matplotlib
from .showIm import showIm
from . import JBhelpers
from .upConv import upConv
from . import pyPyrUtils

class Spyr(pyramid):
    filt = ''
    edges = ''

    #constructor
    def __init__(self, image, height='auto', filter='sp1Filters', edges='reflect1'):
        """Steerable pyramid. image parameter is required, others are optional

        - `image` - a 2D numpy array

        - `height` - an integer denoting number of pyramid levels desired.  'auto' (default) uses
        maxPyrHt from pyPyrUtils.

        - `filter` - The name of one of the steerable pyramid filters in pyPyrUtils:
        `'sp0Filters'`, `'sp1Filters'`, `'sp3Filters'`, `'sp5Filters'`.  Default is `'sp1Filters'`.

        - `edges` - specifies edge-handling.  Options are:
           * `'circular'` - circular convolution
           * `'reflect1'` - reflect about the edge pixels
           * `'reflect2'` - reflect, doubling the edge pixels
           * `'repeat'` - repeat the edge pixels
           * `'zero'` - assume values of zero outside image boundary
           * `'extend'` - reflect and invert
           * `'dont-compute'` - zero output when filter overhangs input boundaries.
        """
        self.pyrType = 'steerable'
        self.image = numpy.array(image)
        self.filt = get_filter(filter)

        self.edges = edges

        harmonics = filters['harmonics']
        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']

        max_ht = maxPyrHt(self.image.shape, lofilt.shape)
        if height == 'auto':
            ht = max_ht
        elif height > max_ht:
            raise Exception("cannot build pyramid higher than %d levels." % (max_ht))
        else:
            ht = height

        nbands = bfilts.shape[1]

        self.pyr = []
        self.pyrSize = []
        for n in range((ht*nbands)+2):
            self.pyr.append([])
            self.pyrSize.append([])

        im = self.image
        im_sz = im.shape
        pyrCtr = 0

        hi0 = corrDn(image = im, filt = hi0filt, edges = edges);

        self.pyr[pyrCtr] = hi0
        self.pyrSize[pyrCtr] = hi0.shape

        pyrCtr += 1

        lo = corrDn(image = im, filt = lo0filt, edges = edges)
        for i in range(ht):
            lo_sz = lo.shape
            # assume square filters  -- start of buildSpyrLevs
            bfiltsz = int(math.floor(math.sqrt(bfilts.shape[0])))

            for b in range(bfilts.shape[1]):
                filt = bfilts[:,b].reshape(bfiltsz,bfiltsz).T
                band = corrDn(image = lo, filt = filt, edges = edges)
                self.pyr[pyrCtr] = numpy.array(band)
                self.pyrSize[pyrCtr] = (band.shape[0], band.shape[1])
                pyrCtr += 1

            lo = corrDn(image = lo, filt = lofilt, edges = edges, step = (2,2))

        self.pyr[pyrCtr] = numpy.array(lo)
        self.pyrSize[pyrCtr] = lo.shape

    # methods
    def set(self, band, location, value):
        """set value at specified band and location

        band and value must be integers, location can be an int or a tuple
        """
        if isinstance(location, int):
            self.pyr[band][0][location] = value
        elif isinstance(location, tuple):
            self.pyr[band][location[0]][location[1]] = value
        else:
            raise Exception('location parameter must be int or tuple!')

    def spyrLev(self, lev):
        if lev < 0 or lev > self.spyrHt()-1:
            raise Exception('level parameter must be between 0 and %d!' % (self.spyrHt()-1))

        levArray = []
        for n in range(self.numBands()):
            levArray.append(self.spyrBand(lev, n))
        levArray = numpy.array(levArray)

        return levArray

    def spyrBand(self, lev, band):
        if lev < 0 or lev > self.spyrHt()-1:
            raise Exception('level parameter must be between 0 and %d!' % (self.spyrHt()-1))
        if band < 0 or band > self.numBands()-1:
            raise Exception('band parameter must be between 0 and %d!' % (self.numBands()-1))

        return self.band( ((lev*self.numBands())+band)+1 )

    def spyrHt(self):
        if len(self.pyrSize) > 2:
            spHt = (len(self.pyrSize)-2)/self.numBands()
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
        return numpy.array(self.band(len(self.pyrSize)-1))

    def pyrHigh(self):
        return numpy.array(self.band(0))

    def reconPyr(self, *args):
        # defaults

        if len(args) > 0:
            filters = get_filter(args[0])
        else:
            filters = get_filter('sp1Filters')

        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']
        # assume square filters  -- start of buildSpyrLevs
        bfiltsz = int(math.floor(math.sqrt(bfilts.shape[0])))

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
            levs = numpy.array(list(range(maxLev)))
        else:
            levs = numpy.array(levs)
            if (levs < 0).any() or (levs >= maxLev).any():
                raise Exception("level numbers must be in the range [0, %d]." % (maxLev-1))
            else:
                levs = numpy.array(levs)
                if len(levs) > 1 and levs[0] < levs[1]:
                    levs = levs[::-1]  # we want smallest first
        if bands == 'all':
            bands = numpy.array(list(range(self.numBands())))
        else:
            bands = numpy.array(bands)
            if (bands < 0).any() or (bands > bfilts.shape[1]).any():
                raise Exception("band numbers must be in the range [0, %d]." % (self.numBands()-1))
            else:
                bands = numpy.array(bands)

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

        reconList = numpy.sort(reconList)[::-1]  # deepest level first

        # initialize reconstruction
        if len(self.pyr)-1 in reconList:
            recon = numpy.array(self.pyr[len(self.pyrSize)-1])
        else:
            recon = numpy.zeros(self.pyr[len(self.pyrSize)-1].shape)

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

                    recon = upConv(image = self.pyr[bandImageIdx],
                                   filt = filt, edges = edges,
                                   stop = (self.pyrSize[bandImageIdx][0],
                                           self.pyrSize[bandImageIdx][1]),
                                   result = recon)
                    bandImageIdx += 1

        # apply lo0filt
        sz = recon.shape
        recon = upConv(image = recon, filt = lo0filt, edges = edges, stop = sz)

        # apply hi0filt if needed
        if 0 in reconList:
            recon = upConv(image = self.pyr[0], filt = hi0filt, edges = edges,
                           start = (0,0), step = (1,1), stop = recon.shape,
                           result = recon)

        return recon

    def showPyr(self, prange = 'auto2', gap = 1, scale = 2, disp = 'qt'):
        ht = self.spyrHt()
        nind = len(self.pyr)
        nbands = self.numBands()

        ## Auto range calculations:
        if prange == 'auto1':
            prange = numpy.ones((nind,1))
            band = self.pyrHigh()
            mn = numpy.amin(band)
            mx = numpy.amax(band)
            for lnum in range(1,ht+1):
                for bnum in range(nbands):
                    idx = pyPyrUtils.LB2idx(lnum, bnum, ht+2, nbands)
                    band = self.band(idx)/(numpy.power(scale,lnum))
                    prange[(lnum-1)*nbands+bnum+1] = numpy.power(scale,lnum-1)
                    bmn = numpy.amin(band)
                    bmx = numpy.amax(band)
                    mn = min([mn, bmn])
                    mx = max([mx, bmx])
            prange = numpy.outer(prange, numpy.array([mn, mx]))
            band = self.pyrLow()
            mn = numpy.amin(band)
            mx = numpy.amax(band)
            prange[nind-1,:] = numpy.array([mn, mx])
        elif prange == 'indep1':
            prange = numpy.zeros((nind,2))
            for bnum in range(nind):
                band = self.band(bnum)
                mn = band.min()
                mx = band.max()
                prange[bnum,:] = numpy.array([mn, mx])
        elif prange == 'auto2':
            prange = numpy.ones(nind)
            band = self.pyrHigh()
            sqsum = numpy.sum( numpy.power(band, 2) )
            numpixels = band.shape[0] * band.shape[1]
            for lnum in range(1,ht+1):
                for bnum in range(nbands):
                    band = self.band(LB2idx(lnum, bnum, ht+2, nbands))
                    band = band / numpy.power(scale,lnum-1)
                    sqsum += numpy.sum( numpy.power(band, 2) )
                    numpixels += band.shape[0] * band.shape[1]
                    prange[(lnum-1)*nbands+bnum+1] = numpy.power(scale, lnum-1)
            stdev = numpy.sqrt( sqsum / (numpixels-1) )
            prange = numpy.outer(prange, numpy.array([-3*stdev, 3*stdev]))
            band = self.pyrLow()
            av = numpy.mean(band)
            stdev = numpy.sqrt( numpy.var(band) )
            prange[nind-1,:] = numpy.array([av-2*stdev, av+2*stdev])
        elif prange == 'indep2':
            prange = numpy.zeros((nind,2))
            for bnum in range(nind-1):
                band = self.band(bnum)
                stdev = numpy.sqrt( numpy.var(band) )
                prange[bnum,:] = numpy.array([-3*stdev, 3*stdev])
            band = self.pyrLow()
            av = numpy.mean(band)
            stdev = numpy.sqrt( numpy.var(band) )
            prange[nind-1,:] = numpy.array([av-2*stdev, av+2*stdev])
        elif isinstance(prange, str):
            raise Exception("Bad RANGE argument: %s'" % (prange))
        elif prange.shape[0] == 1 and prange.shape[1] == 2:
            scales = numpy.power(scale, list(range(ht)))
            scales = numpy.outer( numpy.ones((nbands,1)), scales )
            scales = numpy.array([1, scales, numpy.power(scale, ht)])
            prange = numpy.outer(scales, prange)
            band = self.pyrLow()
            prange[nind,:] += numpy.mean(band) - numpy.mean(prange[nind,:])

        colormap = matplotlib.cm.Greys_r

        # compute positions of subbands
        llpos = numpy.ones((nind,2));

        if nbands == 2:
            ncols = 1
            nrows = 1 # pe: nrows was incorrectely set to 2, changed it to 1 2/18
        else:
            ncols = int(numpy.ceil((nbands+1)/2))
            nrows = int(numpy.ceil(nbands/2))

        a = numpy.array(list(range(1-nrows, 1)))
        b = numpy.zeros((1,ncols))[0]
        ab = numpy.concatenate((a,b))
        c = numpy.zeros((1,nrows))[0]
        d = list(range(-1, -ncols-1, -1))
        cd = numpy.concatenate((c,d))
        relpos = numpy.vstack((ab,cd)).T

        if nbands > 1:
            mvpos = numpy.array([-1, -1]).reshape(1,2)
        else:
            mvpos = numpy.array([0, -1]).reshape(1,2)
        basepos = numpy.array([0, 0]).reshape(1,2)

        for lnum in range(1,ht+1):
            ind1 = (lnum-1)*nbands + 1
            sz = numpy.array(self.pyrSize[ind1]) + gap
            basepos = basepos + mvpos * sz
            if nbands < 5:         # to align edges
                sz += gap * (ht-lnum)
            llpos[ind1:ind1+nbands, :] = numpy.dot(relpos, numpy.diag(sz)) + ( numpy.ones((nbands,1)) * basepos )

        # lowpass band
        sz = numpy.array(self.pyrSize[nind-1]) + gap
        basepos += mvpos * sz
        llpos[nind-1,:] = basepos

        # make position list positive, and allocate appropriate image:
        llpos = llpos - ((numpy.ones((nind,2)) * numpy.amin(llpos, axis=0)) + 1) + 1
        llpos[0,:] = numpy.array([1, 1])
        # we want to cast it as ints, since we'll be using these as indices
        llpos = llpos.astype(int)
        urpos = llpos + self.pyrSize
        d_im = numpy.zeros((numpy.amax(urpos), numpy.amax(urpos)))

        # paste bands into image, (im-r1)*(nshades-1)/(r2-r1) + 1.5
        nshades = 64;

        for bnum in range(1,nind):
            mult = (nshades-1) / (prange[bnum,1]-prange[bnum,0])
            d_im[llpos[bnum,0]:urpos[bnum,0],
                 llpos[bnum,1]:urpos[bnum,1]] = mult * self.band(bnum) + (1.5-mult*prange[bnum,0])

        if disp == 'qt':
            showIm(d_im[:self.pyrSize[0][0]*2,:])
        elif disp == 'nb':
            JBhelpers.showIm(d_im[:self.pyrSize[0][0]*2,:])
