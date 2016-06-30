from pyramid import pyramid
import numpy
from sp0Filters import sp0Filters
from sp1Filters import sp1Filters
from sp3Filters import sp3Filters
from sp5Filters import sp5Filters
import os
from maxPyrHt import maxPyrHt
from corrDn import corrDn
import math
from LB2idx import LB2idx
import matplotlib
from showIm import showIm
import JBhelpers
from upConv import upConv

class Spyr(pyramid):
    filt = ''
    edges = ''
    
    #constructor
    def __init__(self, *args):    # (image height, filter file, edges)
        self.pyrType = 'steerable'
        if len(args) > 0:
            self.image = numpy.array(args[0])
        else:
            print "First argument (image) is required."
            return

        #------------------------------------------------
        # defaults:

        if len(args) > 2:
            if args[2] == 'sp0Filters':
                filters = sp0Filters()
            elif args[2] == 'sp1Filters':
                filters = sp1Filters()
            elif args[2] == 'sp3Filters':
                filters = sp3Filters()
            elif args[2] == 'sp5Filters':
                filters = sp5Filters()
            elif os.path.isfile(args[2]):
                print "Filter files not supported yet"
                return
            else:
                print "filter parameters value %s not supported" % (args[2])
                return
        else:
            filters = sp1Filters()

        harmonics = filters['harmonics']
        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']
        
        max_ht = maxPyrHt(self.image.shape, lofilt.shape)
        if len(args) > 1:
            if args[1] == 'auto':
                ht = max_ht
            elif args[1] > max_ht:
                print "Error: cannot build pyramid higher than %d levels." % (
                    max_ht)
                return
            else:
                ht = args[1]
        else:
            ht = max_ht

        if len(args) > 3:
            edges = args[3]
        else:
            edges = 'reflect1'

        #------------------------------------------------------

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
    def set(self, *args):
        if len(args) != 3:
            print 'Error: three input parameters required:'
            print '  set(band, location, value)'
            print '  where band and value are integer and location is a tuple'
        if isinstance(args[1], (int, long)):
            self.pyr[args[0]][0][args[1]] = args[2]
        elif isinstance(args[1], tuple):
            self.pyr[args[0]][args[1][0]][args[1][1]] = args[2] 
        else:
            print 'Error: location parameter must be int or tuple!'
            return

    def spyrLev(self, lev):
        if lev < 0 or lev > self.spyrHt()-1:
            print 'Error: level parameter must be between 0 and %d!' % (self.spyrHt()-1)
            return
        
        levArray = []
        for n in range(self.numBands()):
            levArray.append(self.spyrBand(lev, n))
        levArray = numpy.array(levArray)

        return levArray

    def spyrBand(self, lev, band):
        if lev < 0 or lev > self.spyrHt()-1:
            print 'Error: level parameter must be between 0 and %d!' % (self.spyrHt()-1)
            return
        if band < 0 or band > self.numBands()-1:
            print 'Error: band parameter must be between 0 and %d!' % (self.numBands()-1)

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
            if args[0] == 'sp0Filters':
                filters = sp0Filters()
            elif args[0] == 'sp1Filters':
                filters = sp1Filters()
            elif args[0] == 'sp3Filters':
                filters = sp3Filters()
            elif args[0] == 'sp5Filters':
                filters = sp5Filters()
            elif os.path.isfile(args[0]):
                print "Filter files not supported yet"
                return
            else:
                print "filter %s not supported" % (args[0])
                return
        else:
            filters = sp1Filters()

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
            levs = numpy.array(range(maxLev))
        else:
            levs = numpy.array(levs)
            if (levs < 0).any() or (levs >= maxLev).any():
                print "Error: level numbers must be in the range [0, %d]." % (maxLev-1)
                return
            else:
                levs = numpy.array(levs)
                if len(levs) > 1 and levs[0] < levs[1]:
                    levs = levs[::-1]  # we want smallest first
        if bands == 'all':
            bands = numpy.array(range(self.numBands()))
        else:
            bands = numpy.array(bands)
            if (bands < 0).any() or (bands > bfilts.shape[1]).any():
                print "Error: band numbers must be in the range [0, %d]." % (self.numBands()-1)
                return
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
    
    #def showPyr(self, *args):
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
        elif isinstance(prange, basestring):
            print "Error:Bad RANGE argument: %s'" % (prange)
        elif prange.shape[0] == 1 and prange.shape[1] == 2:
            scales = numpy.power(scale, range(ht))
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
            nrows = 2
        else:
            ncols = int(numpy.ceil((nbands+1)/2))
            nrows = int(numpy.ceil(nbands/2))

        a = numpy.array(range(1-nrows, 1))
        b = numpy.zeros((1,ncols))[0]
        ab = numpy.concatenate((a,b))
        c = numpy.zeros((1,nrows))[0]
        d = range(-1, -ncols-1, -1)
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
