import numpy
#import pyPyrUtils as ppu
import pyPyrUtils
import pyPyrCcode
import math
import matplotlib.cm
import os
import scipy.misc
import cmath
import JBhelpers
import pylab

class pyramid:  # pyramid
    # properties
    pyr = []
    pyrSize = []
    pyrType = ''
    image = ''

    # constructor
    def __init__(self):
        print "please specify type of pyramid to create (Gpry, Lpyr, etc.)"
        return

    # methods
    def nbands(self):
        return len(self.pyr)

    def band(self, bandNum):
        return numpy.array(self.pyr[bandNum])

    #def showPyr(self, *args):
    #    if self.pyrType == 'Gaussian':
    #        self.showLpyr(args)
    #    elif self.pyrType == 'Laplacian':
    #        self.showLpyr(args)
    #    elif self.pyrType == 'wavelet':
    #        self.showLpyr(args)
    #    else:
    #        print "pyramid type %s not currently supported" % (args[0])
    #        return

# works with all filters
'''
class Spyr_old(pyramid):
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
                filters = pyPyrUtils.sp0Filters()
            elif args[2] == 'sp1Filters':
                filters = pyPyrUtils.sp1Filters()
            elif args[2] == 'sp3Filters':
                filters = pyPyrUtils.sp3Filters()
            elif args[2] == 'sp5Filters':
                filters = pyPyrUtils.sp5Filters()
            elif os.path.isfile(args[2]):
                print "Filter files not supported yet"
                return
            else:
                print "filter parameters value %s not supported" % (args[2])
                return
        else:
            filters = pyPyrUtils.sp1Filters()

        harmonics = filters['harmonics']
        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']
            
        max_ht = pyPyrUtils.maxPyrHt(self.image.shape, lofilt.shape)  # just lofilt[1]?
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

        hi0 = pyPyrCcode.corrDn(im_sz[1], im_sz[0], im, hi0filt.shape[0], 
                                hi0filt.shape[1], hi0filt, edges);
        hi0 = numpy.array(hi0).reshape(im_sz[0], im_sz[1])
        print 'hi0'
        print hi0
        
        self.pyr[pyrCtr] = hi0.copy()
        self.pyrSize[pyrCtr] = hi0.shape
        #self.pyr.append(hi0.copy())
        #self.pyrSize.append(hi0.shape)
        pyrCtr += 1

        lo = pyPyrCcode.corrDn(im_sz[1], im_sz[0], im, lo0filt.shape[0], 
                               lo0filt.shape[1], lo0filt, edges);
        lo = numpy.array(lo).reshape(im_sz[0], im_sz[1])
        print 'lo'
        print lo
        for i in range(ht):
            lo_sz = lo.shape
            # assume square filters  -- start of buildSpyrLevs
            bfiltsz = int(math.floor(math.sqrt(bfilts.shape[0])))

            #pyrCtr += nbands-1
            #for b in range(bfilts.shape[1]-1,-1,-1):
            # for old version
            for b in range(bfilts.shape[1]):
                filt = bfilts[:,b].reshape(bfiltsz,bfiltsz).T
                print 'filt'
                print filt
                band = pyPyrCcode.corrDn(lo_sz[1], lo_sz[0], lo, bfiltsz, 
                                         bfiltsz, filt, edges)
             
                print 'band'
                print band
                self.pyr[pyrCtr] = numpy.array(band.copy()).reshape(lo_sz[0], 
                                                                 lo_sz[1])
                self.pyrSize[pyrCtr] = lo_sz
                pyrCtr += 1


            lo = pyPyrCcode.corrDn(lo_sz[1], lo_sz[0], lo, lofilt.shape[0], 
                                   lofilt.shape[1], lofilt, edges, 2, 2)
            lo = numpy.array(lo).reshape(math.ceil(lo_sz[0]/2.0), 
                                      math.ceil(lo_sz[1]/2.0))
            print 'lo'
            print lo

        self.pyr[pyrCtr] = numpy.array(lo).copy()
        self.pyrSize[pyrCtr] = lo.shape
        #self.pyr.append(numpy.array(lo).copy())
        #self.pyrSize.append(lo.shape)

    # methods
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

    #def reconSpyr(self, *args):
    def reconPyr(self, *args):
        # defaults

        if len(args) > 0:
            if args[0] == 'sp0Filters':
                filters = pyPyrUtils.sp0Filters()
            elif args[0] == 'sp1Filters':
                filters = pyPyrUtils.sp1Filters()
            elif args[0] == 'sp3Filters':
                filters = pyPyrUtils.sp3Filters()
            elif args[0] == 'sp5Filters':
                filters = pyPyrUtils.sp5Filters()
            elif os.path.isfile(args[0]):
                print "Filter files not supported yet"
                return
            else:
                print "supported filter parameters are 'sp0Filters' and 'sp1Filters'"
                return
        else:
            filters = pyPyrUtils.sp1Filters()

        #harmonics = filters['harmonics']
        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']
        #print harmonics.shape
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
            if (levs < 0).any() or (levs >= maxLev-1).any():
                print "Error: level numbers must be in the range [0, %d]." % (maxLev-1)
            else:
                levs = numpy.array(levs)
                if len(levs) > 1 and levs[0] < levs[1]:
                    levs = levs[::-1]  # we want smallest first
        if bands == 'all':
            bands = numpy.array(range(self.numBands()))
        else:
            bands = numpy.array(bands)
            #if (bands < 0).any() or (bands > self.numBands).any():
            if (bands < 0).any() or (bands > bfilts.shape[1]).any():
                print "Error: band numbers must be in the range [0, %d]." % (self.numBands())
            else:
                bands = numpy.array(bands)

        # make a list of all pyramid layers to be used in reconstruction
        # FIX: if not supplied by user
        Nlevs = self.spyrHt()+2
        Nbands = self.numBands()

        reconList = []  # pyr indices used in reconstruction
        for lev in levs:
            if lev == 0 or lev == Nlevs-1 :
                reconList.append( pyPyrUtils.LB2idx(lev, -1, Nlevs, Nbands) )
            else:
                for band in bands:
                    reconList.append( pyPyrUtils.LB2idx(lev, band, Nlevs, Nbands) )
        # reconstruct
        # FIX: shouldn't have to enter step, start and stop in upConv!
        band = -1
        for lev in range(Nlevs-1,-1,-1):
            if lev == Nlevs-1 and pyPyrUtils.LB2idx(lev,-1,Nlevs,Nbands) in reconList:
                idx = pyPyrUtils.LB2idx(lev, band, Nlevs, Nbands)
                recon = numpy.array(self.pyr[len(self.pyrSize)-1].copy())
            elif lev == Nlevs-1:
                idx = pyPyrUtils.LB2idx(lev, band, Nlevs, Nbands)
                recon = numpy.zeros(self.pyr[len(self.pyrSize)-1].shape)
            elif lev == 0 and 0 in reconList:
                # orig working code
                #idx = pyPyrUtils.LB2idx(lev, band, Nlevs, Nbands)
                #sz = recon.shape
                ##recon = upConv(sz[0], sz[1], recon, hi0filt.shape[0], 
                ##               hi0filt.shape[1], lo0filt, edges, 1, 1, 0, 0, 
                ##               sz[1], sz[0])
                #recon = pyPyrCcode.upConv(sz[0], sz[1], recon, lo0filt.shape[0],
                #lo0filt.shape[1], lo0filt, edges, 
                #                          1, 1, 0, 0, sz[1], sz[0])
                ##recon = numpy.array(recon).reshape(sz[0], sz[1])
                #recon = pyPyrCcode.upConv(self.pyrSize[idx][0], 
                #                          self.pyrSize[idx][1], self.pyr[idx], 
                #                          hi0filt.shape[0], hi0filt.shape[1], 
                #                          hi0filt, edges, 1, 1, 0, 0, 
                #                          self.pyrSize[idx][1], 
                #                          self.pyrSize[idx][0], recon)
                ##recon = numpy.array(recon).reshape(self.pyrSize[idx][0], self.pyrSize[idx][1], order='C')
                idx = pyPyrUtils.LB2idx(lev, band, Nlevs, Nbands)
                sz = recon.shape
                recon = lib.upConv(image = recon, filt = lo2filt, edges = edges)
                lib.upConv(image = self.pyr[idx], filt = hi0filt, edges = edges,
                           result = recon)

            elif lev == 0:
                # orig working code
                #sz = recon.shape
                #recon = pyPyrCcode.upConv(sz[0],sz[1], recon, lo0filt.shape[0],
                #                          lo0filt.shape[1], lo0filt, edges, 
                #                          1, 1, 0, 0, sz[0], sz[1])
                ##recon = numpy.array(recon).reshape(sz[0], sz[1])
                sz = recon.shape
                recon = lib.upConv(image = recon, filt = lo0filt, edges = edges)
            else:
                for band in range(Nbands-1,-1,-1):
                    idx = pyPyrUtils.LB2idx(lev, band, Nlevs, Nbands)
                    if idx in reconList:
                        # made filter band match matlab version
                        #filtBand = band
                        #if filtBand > Nbands-1:
                        #    filtBand = filtBand - Nbands
                        #print 'new filter band idx = %d' % (filtBand)
                        #filt = numpy.negative(bfilts[:,band].reshape(bfiltsz, 
                        #bfiltsz,
                        #                                           order='C'))
                        filt = bfilts[:,band].reshape(bfiltsz, 
                                                      bfiltsz,
                                                      order='F')
                        # orig working version
                        #recon = pyPyrCcode.upConv(self.pyrSize[idx][0], 
                        #                          self.pyrSize[idx][1], 
                        #                          self.pyr[idx], bfiltsz, 
                        #                          bfiltsz, filt, edges, 1, 1, 
                        #                          0, 0, self.pyrSize[idx][1], 
                        #                          self.pyrSize[idx][0], recon)
                        recon = lib.upConv(image = self.pyr[idx], filt = filt,
                                           edges = edges, result = recon)
                        #recon = numpy.array(recon).reshape(self.pyrSize[idx][0], 
                        #                                self.pyrSize[idx][1],
                        #                                order='C')

            # upsample
            #newSz = ppu.nextSz(recon.shape, self.pyrSize.values())
            newSz = pyPyrUtils.nextSz(recon.shape, self.pyrSize)
            mult = newSz[0] / recon.shape[0]
            if newSz[0] % recon.shape[0] > 0:
                mult += 1
            if mult > 1:
                # orig working version
                #recon = pyPyrCcode.upConv(recon.shape[1], recon.shape[0], 
                #                          recon.T, lofilt.shape[0], 
                #                          lofilt.shape[1], lofilt, edges, mult,
                #                          mult, 0, 0, newSz[0], newSz[1]).T
                recon = lib.upConv(image = recon.T, filt = lofilt, 
                                   edges = edges, step = (mult, mult), 
                                   stop = newSz).T
                #recon = numpy.array(recon).reshape(newSz[0], newSz[1], order='F')
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
                    band = self.band(pyPyrUtils.LB2idx(lnum, bnum, ht+2, nbands))
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

        #relpos = numpy.array([numpy.concatenate([ range(1-nrows,1), 
        #                                    numpy.zeros(ncols-1)]), 
        #                   numpy.concatenate([ numpy.zeros(nrows), 
        #                                    range(-1, 1-ncols, -1) ])]).T
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
            #ind1 = (lnum-1)*nbands + 2
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
        #urpos = llpos + self.pyrSize.values()
        urpos = llpos + self.pyrSize
        d_im = numpy.zeros((numpy.amax(urpos), numpy.amax(urpos)))
        
        # paste bands into image, (im-r1)*(nshades-1)/(r2-r1) + 1.5
        nshades = 64;

        for bnum in range(1,nind):
            mult = (nshades-1) / (prange[bnum,1]-prange[bnum,0])
            d_im[llpos[bnum,0]:urpos[bnum,0], 
                 llpos[bnum,1]:urpos[bnum,1]] = mult * self.band(bnum) + (1.5-mult*prange[bnum,0])

        if disp == 'qt':
            pyPyrUtils.showIm(d_im[:self.pyrSize[0][0]*2,:])
        elif disp == 'nb':
            JBhelpers.showIm(d_im[:self.pyrSize[0][0]*2,:])
'''
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
                filters = pyPyrUtils.sp0Filters()
            elif args[2] == 'sp1Filters':
                filters = pyPyrUtils.sp1Filters()
            elif args[2] == 'sp3Filters':
                filters = pyPyrUtils.sp3Filters()
            elif args[2] == 'sp5Filters':
                filters = pyPyrUtils.sp5Filters()
            elif os.path.isfile(args[2]):
                print "Filter files not supported yet"
                return
            else:
                print "filter parameters value %s not supported" % (args[2])
                return
        else:
            filters = pyPyrUtils.sp1Filters()

        harmonics = filters['harmonics']
        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']
            
        max_ht = pyPyrUtils.maxPyrHt(self.image.shape, lofilt.shape)  # just lofilt[1]?
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

        #hi0 = pyPyrCcode.corrDn(im_sz[1], im_sz[0], im, hi0filt.shape[0], 
        #                        hi0filt.shape[1], hi0filt, edges);
        #print hi0
        #hi0 = numpy.array(hi0).reshape(im_sz[0], im_sz[1])
        #print hi0

        hi0 = pyPyrUtils.corrDn(image = im, filt = hi0filt, edges = edges);
        print 'hi0'
        print hi0

        self.pyr[pyrCtr] = hi0.copy()
        self.pyrSize[pyrCtr] = hi0.shape
        print 'wrote idx %d' % pyrCtr
        #self.pyr.append(hi0.copy())
        #self.pyrSize.append(hi0.shape)
        pyrCtr += 1

        #lo = pyPyrCcode.corrDn(im_sz[1], im_sz[0], im, lo0filt.shape[0], 
        #                       lo0filt.shape[1], lo0filt, edges);
        #lo = numpy.array(lo).reshape(im_sz[0], im_sz[1])
        lo = pyPyrUtils.corrDn(image = im, filt = lo0filt, edges = edges)
        print 'lo'
        print lo
        for i in range(ht):
            lo_sz = lo.shape
            # assume square filters  -- start of buildSpyrLevs
            bfiltsz = int(math.floor(math.sqrt(bfilts.shape[0])))

            #pyrCtr += nbands-1
            #for b in range(bfilts.shape[1]-1,-1,-1):
            # for old version
            #for b in range(bfilts.shape[1]):
            #    filt = bfilts[:,b].reshape(bfiltsz,bfiltsz).T
            #    # this one works, needs b in range(bfilts.shape[1])
            #    print 'band - orig %d' % (b)
            #    band = pyPyrCcode.corrDn(lo_sz[1], lo_sz[0], lo, bfiltsz, 
            #                             bfiltsz, filt, edges)
            # 
            #    print band
            #    print 'band - new %d' % (b)
            #    band = pyPyrUtils.corrDn(image = lo.T, filt = filt.T, 
            #                             edges = edges)
            #    print band
            #    self.pyr[pyrCtr] = numpy.array(band.copy()).reshape(lo_sz[0], 
            #                                                     lo_sz[1])
            #    self.pyrSize[pyrCtr] = lo_sz
            #    pyrCtr += 1

            for b in range(bfilts.shape[1]-1,-1,-1):
                filt = numpy.negative(bfilts[:,b].reshape(bfiltsz,bfiltsz))
                print 'filt'
                print filt
                band = pyPyrUtils.corrDn(image = numpy.negative(lo),
                                         filt = filt, 
                                         edges = edges)
                print 'band'
                print band
                self.pyr[pyrCtr] = numpy.negative(numpy.array(band.copy()))
                self.pyrSize[pyrCtr] = (band.shape[0], band.shape[1])
                print 'wrote idx %d' % pyrCtr
                pyrCtr += 1


            #lo = pyPyrCcode.corrDn(lo_sz[1], lo_sz[0], lo, lofilt.shape[0], 
            #                       lofilt.shape[1], lofilt, edges, 2, 2)
            #lo = numpy.array(lo).reshape(math.ceil(lo_sz[0]/2.0), 
            #                          math.ceil(lo_sz[1]/2.0))
            lo = pyPyrUtils.corrDn(image = lo, filt = lofilt, edges = edges, 
                                   step = (2,2))
            print 'lo'
            print lo

        self.pyr[pyrCtr] = numpy.array(lo).copy()
        self.pyrSize[pyrCtr] = lo.shape
        print 'wrote idx %d' % pyrCtr
        #self.pyr.append(numpy.array(lo).copy())
        #self.pyrSize.append(lo.shape)

    # methods
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

    #def reconSpyr(self, *args):
    def reconPyr(self, *args):
        # defaults

        if len(args) > 0:
            if args[0] == 'sp0Filters':
                filters = pyPyrUtils.sp0Filters()
            elif args[0] == 'sp1Filters':
                filters = pyPyrUtils.sp1Filters()
            elif args[0] == 'sp3Filters':
                filters = pyPyrUtils.sp3Filters()
            elif args[0] == 'sp5Filters':
                filters = pyPyrUtils.sp5Filters()
            elif os.path.isfile(args[0]):
                print "Filter files not supported yet"
                return
            else:
                print "supported filter parameters are 'sp0Filters' and 'sp1Filters'"
                return
        else:
            filters = pyPyrUtils.sp1Filters()

        #harmonics = filters['harmonics']
        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']
        #print harmonics.shape
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
            if (levs < 0).any() or (levs >= maxLev-1).any():
                print "Error: level numbers must be in the range [0, %d]." % (maxLev-1)
            else:
                levs = numpy.array(levs)
                if len(levs) > 1 and levs[0] < levs[1]:
                    levs = levs[::-1]  # we want smallest first
        if bands == 'all':
            bands = numpy.array(range(self.numBands()))
        else:
            bands = numpy.array(bands)
            #if (bands < 0).any() or (bands > self.numBands).any():
            if (bands < 0).any() or (bands > bfilts.shape[1]).any():
                print "Error: band numbers must be in the range [0, %d]." % (self.numBands())
            else:
                bands = numpy.array(bands)

        # make a list of all pyramid layers to be used in reconstruction
        # FIX: if not supplied by user
        Nlevs = self.spyrHt()+2
        Nbands = self.numBands()

        reconList = []  # pyr indices used in reconstruction
        for lev in levs:
            if lev == 0 or lev == Nlevs-1 :
                reconList.append( pyPyrUtils.LB2idx(lev, -1, Nlevs, Nbands) )
            else:
                for band in bands:
                    reconList.append( pyPyrUtils.LB2idx(lev, band, Nlevs, Nbands) )
        # reconstruct
        # FIX: shouldn't have to enter step, start and stop in upConv!
        band = -1
        for lev in range(Nlevs-1,-1,-1):
            if lev == Nlevs-1 and pyPyrUtils.LB2idx(lev,-1,Nlevs,Nbands) in reconList:
                idx = pyPyrUtils.LB2idx(lev, band, Nlevs, Nbands)
                recon = numpy.array(self.pyr[len(self.pyrSize)-1].copy())
            elif lev == Nlevs-1:
                idx = pyPyrUtils.LB2idx(lev, band, Nlevs, Nbands)
                recon = numpy.zeros(self.pyr[len(self.pyrSize)-1].shape)
            elif lev == 0 and 0 in reconList:
                ### orig working code
                #idx = pyPyrUtils.LB2idx(lev, band, Nlevs, Nbands)
                #sz = recon.shape
                ##recon = upConv(sz[0], sz[1], recon, hi0filt.shape[0], 
                ##               hi0filt.shape[1], lo0filt, edges, 1, 1, 0, 0, 
                ##               sz[1], sz[0])
                #recon = pyPyrCcode.upConv(sz[0],sz[1], recon, lo0filt.shape[0],
                #                          lo0filt.shape[1], lo0filt, edges, 
                #                          1, 1, 0, 0, sz[1], sz[0])
                #print 'recon1'
                #print recon
                ##recon = numpy.array(recon).reshape(sz[0], sz[1])
                #recon = pyPyrCcode.upConv(self.pyrSize[idx][0], 
                #                          self.pyrSize[idx][1], self.pyr[idx], 
                #                          hi0filt.shape[0], hi0filt.shape[1], 
                #                          hi0filt, edges, 1, 1, 0, 0, 
                #                          self.pyrSize[idx][1], 
                #                          self.pyrSize[idx][0], recon)
                #print 'recon1'
                #print recon
                ##recon = numpy.array(recon).reshape(self.pyrSize[idx][0], self.pyrSize[idx][1], order='C')
                ### old code for testing
                #recon2 = recon.copy()
                #idx = pyPyrUtils.LB2idx(lev, band, Nlevs, Nbands)
                #sz = recon.shape
                #print 'recon1_1_start'
                #recon = pyPyrCcode.upConv(sz[0],sz[1], recon, lo0filt.shape[0],
                #                          lo0filt.shape[1], lo0filt, edges, 
                #                          1, 1, 0, 0, sz[1], sz[0])
                #print 'recon1_1_end'
                #print recon
                #print 'recon1_2_start'
                #recon = pyPyrCcode.upConv(self.pyrSize[idx][0], 
                #                          self.pyrSize[idx][1], self.pyr[idx], 
                #                          hi0filt.shape[0], hi0filt.shape[1], 
                #                          hi0filt, edges, 1, 1, 0, 0, 
                #                          self.pyrSize[idx][1], 
                #                          self.pyrSize[idx][0], recon)
                #print 'recon1_2_end'
                #print recon
                #### new code -- output from individual functions looks the
                #                same, but fails unit test 12. WHY?!
                sz = recon.shape
                print 'recon2_1_start'
                recon = pyPyrUtils.upConv(image = recon, filt = lo0filt, 
                                          edges = edges, step = (1,1), 
                                          start = (0,0), stop = (sz[1], sz[0]))
                #print 'recon2_1_end'
                #print recon2
                idx = pyPyrUtils.LB2idx(lev, band, Nlevs, Nbands)
                #print 'recon2_2_start'
                pyPyrUtils.upConv(image = self.pyr[idx],
                                  filt = hi0filt, edges = edges,
                                  stop = (self.pyrSize[idx][1], 
                                          self.pyrSize[idx][0]), 
                                  result = recon)
                #print 'recon2_2_end'
                #print recon2
            elif lev == 0:
                sz = recon.shape
                #recon = pyPyrCcode.upConv(sz[0],sz[1], recon, lo0filt.shape[0],
                #                          lo0filt.shape[1], lo0filt, edges, 
                #                          1, 1, 0, 0, sz[0], sz[1])
                recon = pyPyrUtils.upConv(image = recon, filt = lo0filt, 
                                          edges = edges, stop = sz)
            else:
                for band in range(Nbands-1,-1,-1):
                    idx = pyPyrUtils.LB2idx(lev, band, Nlevs, Nbands)
                    if idx in reconList:
                        filt = bfilts[:,band].reshape(bfiltsz, 
                                                      bfiltsz,
                                                      order='F')
                        #recon2 = recon.copy()
                        #recon = pyPyrCcode.upConv(self.pyrSize[idx][0], 
                        #                          self.pyrSize[idx][1], 
                        #                          self.pyr[idx], bfiltsz, 
                        #                          bfiltsz, filt, edges, 1, 1, 
                        #                          0, 0, self.pyrSize[idx][1], 
                        #                          self.pyrSize[idx][0], recon)
                        #print 'recon'
                        #print recon
                        recon = pyPyrUtils.upConv(image = self.pyr[idx], 
                                                  filt = filt, edges = edges,
                                                  stop = (self.pyrSize[idx][1],
                                                          self.pyrSize[idx][0]),
                                                  result = recon)
                        #print 'recon2'
                        #print recon2

            # upsample
            #newSz = ppu.nextSz(recon.shape, self.pyrSize.values())
            newSz = pyPyrUtils.nextSz(recon.shape, self.pyrSize)
            mult = newSz[0] / recon.shape[0]
            if newSz[0] % recon.shape[0] > 0:
                mult += 1
            if mult > 1:
                # old working code
                #recon = pyPyrCcode.upConv(recon.shape[1], recon.shape[0], 
                #                          recon.T, lofilt.shape[0], 
                #                          lofilt.shape[1], lofilt, edges, mult,
                #                          mult, 0, 0, newSz[0], newSz[1]).T
                # working copy of old code
                #recon2 = recon.copy()
                #recon = pyPyrCcode.upConv(recon.shape[1], recon.shape[0], 
                #                          recon.T, lofilt.shape[0], 
                #                          lofilt.shape[1], lofilt, edges, mult,
                #                          mult, 0, 0, newSz[0], newSz[1]).T
                #print 'upsample - recon'
                #print recon.shape
                #print recon
                #  works!!
                recon = pyPyrUtils.upConv(image = recon.T, filt = lofilt, 
                                          edges = edges, step = (mult, mult),
                                          stop = newSz).T
                #print 'upsample - recon2'
                #print recon2.shape
                #print recon2
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
                    band = self.band(pyPyrUtils.LB2idx(lnum, bnum, ht+2, nbands))
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

        #relpos = numpy.array([numpy.concatenate([ range(1-nrows,1), 
        #                                    numpy.zeros(ncols-1)]), 
        #                   numpy.concatenate([ numpy.zeros(nrows), 
        #                                    range(-1, 1-ncols, -1) ])]).T
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
            #ind1 = (lnum-1)*nbands + 2
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
        #urpos = llpos + self.pyrSize.values()
        urpos = llpos + self.pyrSize
        d_im = numpy.zeros((numpy.amax(urpos), numpy.amax(urpos)))
        
        # paste bands into image, (im-r1)*(nshades-1)/(r2-r1) + 1.5
        nshades = 64;

        for bnum in range(1,nind):
            mult = (nshades-1) / (prange[bnum,1]-prange[bnum,0])
            d_im[llpos[bnum,0]:urpos[bnum,0], 
                 llpos[bnum,1]:urpos[bnum,1]] = mult * self.band(bnum) + (1.5-mult*prange[bnum,0])

        if disp == 'qt':
            pyPyrUtils.showIm(d_im[:self.pyrSize[0][0]*2,:])
        elif disp == 'nb':
            JBhelpers.showIm(d_im[:self.pyrSize[0][0]*2,:])
 
class SFpyr(Spyr):
    filt = ''
    edges = ''
    
    #constructor
    def __init__(self, *args):    # (image, height, order, twidth)
        self.pyrType = 'steerableFrequency'

        if len(args) > 0:
            self.image = args[0]
        else:
            print "First argument (image) is required."
            return

        #------------------------------------------------
        # defaults:

        max_ht = numpy.floor( numpy.log2( min(self.image.shape) ) ) - 2
        if len(args) > 1:
            if(args[1] > max_ht):
                print "Error: cannot build pyramid higher than %d levels." % (max_ht)
            ht = args[1]
        else:
            ht = max_ht
        ht = int(ht)
            
        if len(args) > 2:
            if args[2] > 15 or args[2] < 0:
                print "Warning: order must be an integer in the range [0,15]. Truncating."
                order = min( max(args[2],0), 15 )
            else:
                order = args[2]
        else:
            order = 3

        nbands = order+1

        if len(args) > 3:
            if args[3] <= 0:
                print "Warning: twidth must be positive. Setting to 1."
                twidth = 1
            else:
                twidth = args[3]
        else:
            twidth = 1

        #------------------------------------------------------
        # steering stuff:

        if nbands % 2 == 0:
            harmonics = numpy.array(range(nbands/2)) * 2 + 1
        else:
            harmonics = numpy.array(range((nbands-1)/2)) * 2

        steermtx = pyPyrUtils.steer2HarmMtx(harmonics, 
                                            numpy.pi*numpy.array(range(nbands))/nbands,
                                            'even')
        #------------------------------------------------------
        
        dims = numpy.array(self.image.shape)
        ctr = numpy.ceil((numpy.array(dims)+0.5)/2)
        
        (xramp, yramp) = numpy.meshgrid((numpy.array(range(1,dims[1]+1))-ctr[1])/
                                     (dims[1]/2), 
                                     (numpy.array(range(1,dims[0]+1))-ctr[0])/
                                     (dims[0]/2))
        angle = numpy.arctan2(yramp, xramp)
        log_rad = numpy.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        log_rad = numpy.log2(log_rad);

        ## Radial transition function (a raised cosine in log-frequency):
        (Xrcos, Yrcos) = pyPyrUtils.rcosFn(twidth, (-twidth/2.0), numpy.array([0,1]))
        Yrcos = numpy.sqrt(Yrcos)

        YIrcos = numpy.sqrt(1.0 - Yrcos**2)
        lo0mask = numpy.array(pyPyrCcode.pointOp(log_rad.shape[0],
                                                 log_rad.shape[1], log_rad,
                                                 YIrcos.shape[0], YIrcos,
                                                 Xrcos[0], Xrcos[1]-Xrcos[0],
                                                 0))

        imdft = numpy.fft.fftshift(numpy.fft.fft2(self.image))

        self.pyr = []
        self.pyrSize = []

        hi0mask = numpy.array(pyPyrCcode.pointOp(log_rad.shape[0],
                                                 log_rad.shape[1], log_rad,
                                                 Yrcos.shape[0], Yrcos,
                                                 Xrcos[0], Xrcos[1]-Xrcos[0],
                                                 0))
        hi0dft = imdft * hi0mask.reshape(imdft.shape[0], imdft.shape[1])
        hi0 = numpy.fft.ifft2(numpy.fft.ifftshift(hi0dft))

        self.pyr.append(numpy.real(hi0.copy()))
        self.pyrSize.append(hi0.shape)

        lo0mask = lo0mask.reshape(imdft.shape[0], imdft.shape[1])
        lodft = imdft * lo0mask

        for i in range(ht):
            bands = numpy.zeros((lodft.shape[0]*lodft.shape[1], nbands))
            bind = numpy.zeros((nbands, 2))
        
            Xrcos -= numpy.log2(2)

            lutsize = 1024
            Xcosn = numpy.pi * numpy.array(range(-(2*lutsize+1), (lutsize+2))) / lutsize

            order = nbands -1
            const = (2**(2*order))*(scipy.misc.factorial(order, exact=True)**2)/float(nbands*scipy.misc.factorial(2*order, exact=True))
            Ycosn = numpy.sqrt(const) * (numpy.cos(Xcosn))**order
            himask = numpy.array(pyPyrCcode.pointOp(log_rad.shape[0],
                                                    log_rad.shape[1], log_rad,
                                                    Yrcos.shape[0], Yrcos,
                                                    Xrcos[0], Xrcos[1]-Xrcos[0],
                                                    0))
            himask = himask.reshape(lodft.shape[0], lodft.shape[1])

            for b in range(nbands):
                anglemask = numpy.array(pyPyrCcode.pointOp(angle.shape[0],
                                                           angle.shape[1], 
                                                           angle,
                                                           Ycosn.shape[0],
                                                           Ycosn, 
                                                           Xcosn[0]+numpy.pi*b/nbands,
                                                           Xcosn[1]-Xcosn[0],
                                                           0))
                anglemask = anglemask.reshape(lodft.shape[0], lodft.shape[1])
                banddft = ((-numpy.power(-1+0j,0.5))**order) * lodft * anglemask * himask
                band = numpy.fft.ifft2(numpy.fft.ifftshift(banddft))
                self.pyr.append(numpy.real(band.copy()))
                self.pyrSize.append(band.shape)

            dims = numpy.array(lodft.shape)
            ctr = numpy.ceil((dims+0.5)/2)
            lodims = numpy.ceil((dims-0.5)/2)
            #loctr = ceil((lodims+0.5)/2)
            loctr = numpy.ceil((lodims+0.5)/2)
            lostart = ctr - loctr
            loend = lostart + lodims

            log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = numpy.abs(numpy.sqrt(1.0 - Yrcos**2))
            lomask = numpy.array(pyPyrCcode.pointOp(log_rad.shape[0],
                                                    log_rad.shape[1], 
                                                    log_rad, YIrcos.shape[0],
                                                    YIrcos, Xrcos[0],
                                                    Xrcos[1]-Xrcos[0], 0))
            lodft = lodft * lomask.reshape(lodft.shape[0], lodft.shape[1])

        lodft = numpy.fft.ifft2(numpy.fft.ifftshift(lodft))
        self.pyr.append(numpy.real(numpy.array(lodft).copy()))
        self.pyrSize.append(lodft.shape)

    # methods
    def numBands(self):      # why isn't this inherited
        if len(self.pyrSize) == 2:
            return 0
        else:
            b = 2
            while ( b <= len(self.pyrSize) and 
                    self.pyrSize[b] == self.pyrSize[1] ):
                b += 1
            return b-1

    def spyrHt(self):
        if len(self.pyrSize) > 2:
            spHt = (len(self.pyrSize)-2)/self.numBands()
        else:
            spHt = 0
        return spHt

    def reconPyr(self, *args):
        res = self.reconSFpyr(self, *args)
        return res

    def reconSFpyr(self, *args):
    #def reconPyr(self, *args):
        if len(args) > 0:
            levs = args[0]
        else:
            levs = 'all'

        if len(args) > 1:
            bands = args[1]
        else:
            bands = 'all'

        if len(args) > 2:
            if args[2] <= 0:
                print "Warning: twidth must be positive. Setting to 1."
                twidth = 1
            else:
                twidth = args[2]
        else:
            twidth = 1

        #-----------------------------------------------------------------

        nbands = self.numBands()
        
        maxLev = 1 + self.spyrHt()
        if isinstance(levs, basestring) and levs == 'all':
            levs = numpy.array(range(maxLev+1))
        elif isinstance(levs, basestring):
            print "Error: %s not valid for levs parameter." % (levs)
            print "levs must be either a 1D numpy array or the string 'all'."
            return
        else:
            levs = numpy.array(levs)

        if isinstance(bands, basestring) and bands == 'all':
            bands = numpy.array(range(nbands))
        elif isinstance(bands, basestring):
            print "Error: %s not valid for bands parameter." % (bands)
            print "bands must be either a 1D numpy array or the string 'all'."
            return
        else:
            bands = numpy.array(bands)

        #-------------------------------------------------------------------
        # make list of dims and bounds
        boundList = []
        dimList = []
        for dimIdx in range(len(self.pyrSize)-1,-1,-1):
            dims = numpy.array(self.pyrSize[dimIdx])
            if (dims[0], dims[1]) not in dimList:
                dimList.append( (dims[0], dims[1]) )
            ctr = numpy.ceil((dims+0.5)/2)
            lodims = numpy.ceil((dims-0.5)/2)
            #loctr = ceil((lodims+0.5)/2)
            loctr = numpy.ceil((lodims+0.5)/2)
            lostart = ctr - loctr
            loend = lostart + lodims
            bounds = (lostart[0], lostart[1], loend[0], loend[1])
            if bounds not in boundList:
                boundList.append( bounds )
        boundList.append((0.0, 0.0, dimList[len(dimList)-1][0], 
                          dimList[len(dimList)-1][1]))
        dimList.append((dimList[len(dimList)-1][0], dimList[len(dimList)-1][1]))
        
        # matlab code starts here
        dims = numpy.array(self.pyrSize[0])
        ctr = numpy.ceil((dims+0.5)/2.0)

        (xramp, yramp) = numpy.meshgrid((numpy.array(range(1,dims[1]+1))-ctr[1])/
                                     (dims[1]/2), 
                                     (numpy.array(range(1,dims[0]+1))-ctr[0])/
                                     (dims[0]/2))
        angle = numpy.arctan2(yramp, xramp)
        log_rad = numpy.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        log_rad = numpy.log2(log_rad);

        ## Radial transition function (a raised cosine in log-frequency):
        (Xrcos, Yrcos) = pyPyrUtils.rcosFn(twidth, (-twidth/2.0), numpy.array([0,1]))
        Yrcos = numpy.sqrt(Yrcos)
        YIrcos = numpy.sqrt(1.0 - Yrcos**2)

        # from reconSFpyrLevs
        lutsize = 1024
        Xcosn = numpy.pi * numpy.array(range(-(2*lutsize+1), (lutsize+2))) / lutsize
        
        order = nbands -1
        const = (2**(2*order))*(scipy.misc.factorial(order, exact=True)**2)/float(nbands*scipy.misc.factorial(2*order, exact=True))
        Ycosn = numpy.sqrt(const) * (numpy.cos(Xcosn))**order

        # lowest band
        nres = self.pyr[len(self.pyr)-1]
        if self.spyrHt()+1 in levs:
            nresdft = numpy.fft.fftshift(numpy.fft.fft2(nres))
        else:
            nresdft = numpy.zeros(nres.shape)
        resdft = numpy.zeros(dimList[1]) + 0j

        bounds = (0, 0, 0, 0)
        for idx in range(len(boundList)-2, 0, -1):
            diff = (boundList[idx][2]-boundList[idx][0], 
                    boundList[idx][3]-boundList[idx][1])
            bounds = (bounds[0]+boundList[idx][0], bounds[1]+boundList[idx][1], 
                      bounds[0]+boundList[idx][0] + diff[0], 
                      bounds[1]+boundList[idx][1] + diff[1])
            Xrcos -= numpy.log2(2.0)
        nlog_rad = log_rad[bounds[0]:bounds[2], bounds[1]:bounds[3]]

        lomask = numpy.array(pyPyrCcode.pointOp(nlog_rad.shape[0],
                                                nlog_rad.shape[1], nlog_rad,
                                                YIrcos.shape[0], YIrcos,
                                                Xrcos[0], Xrcos[1]-Xrcos[0], 0))
        lomask = lomask.reshape(nres.shape[0], nres.shape[1])
        lomask = lomask + 0j
        resdft[boundList[1][0]:boundList[1][2], 
               boundList[1][1]:boundList[1][3]] = nresdft * lomask

        # middle bands
        bandIdx = (len(self.pyr)-1) + nbands
        for idx in range(1, len(boundList)-1):
            bounds1 = (0, 0, 0, 0)
            bounds2 = (0, 0, 0, 0)
            for boundIdx in range(len(boundList)-1,idx-1,-1):
                diff = (boundList[boundIdx][2]-boundList[boundIdx][0], 
                        boundList[boundIdx][3]-boundList[boundIdx][1])
                bound2tmp = bounds2
                bounds2 = (bounds2[0]+boundList[boundIdx][0], 
                           bounds2[1]+boundList[boundIdx][1],
                           bounds2[0]+boundList[boundIdx][0] + diff[0], 
                           bounds2[1]+boundList[boundIdx][1] + diff[1])
                bounds1 = bound2tmp
            nlog_rad1=log_rad[bounds1[0]:bounds1[2], bounds1[1]:bounds1[3]]
            nlog_rad2=log_rad[bounds2[0]:bounds2[2],bounds2[1]:bounds2[3]]
            dims = dimList[idx]
            nangle = angle[bounds1[0]:bounds1[2], bounds1[1]:bounds1[3]]
            YIrcos = numpy.abs(numpy.sqrt(1.0 - Yrcos**2))
            if idx > 1:
                Xrcos += numpy.log2(2.0)
                lomask = numpy.array(pyPyrCcode.pointOp(nlog_rad2.shape[0], 
                                                        nlog_rad2.shape[1], 
                                                        nlog_rad2,
                                                        YIrcos.shape[0], YIrcos,
                                                        Xrcos[0],
                                                        Xrcos[1]-Xrcos[0], 0))
                lomask = lomask.reshape(bounds2[2]-bounds2[0],
                                        bounds2[3]-bounds2[1])
                lomask = lomask + 0j
                nresdft = numpy.zeros(dimList[idx]) + 0j
                nresdft[boundList[idx][0]:boundList[idx][2], 
                        boundList[idx][1]:boundList[idx][3]] = resdft * lomask
                resdft = nresdft.copy()

            bandIdx -= 2 * nbands

            # reconSFpyrLevs
            if idx != 0 and idx != len(boundList)-1:
                for b in range(nbands):
                    if (bands == b).any():
                        himask = numpy.array(pyPyrCcode.pointOp(nlog_rad1.shape[0], nlog_rad1.shape[1], nlog_rad1, Yrcos.shape[0], Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0))
                        himask = himask.reshape(nlog_rad1.shape)
                        anglemask = numpy.array(pyPyrCcode.pointOp(nangle.shape[0], nangle.shape[1], nangle, Ycosn.shape[0], Ycosn, Xcosn[0]+numpy.pi*b/nbands, Xcosn[1]-Xcosn[0], 0))
                        anglemask = anglemask.reshape(nangle.shape)
                        band = self.pyr[bandIdx]
                        curLev = self.spyrHt() - (idx-1)
                        if curLev in levs and b in bands:
                            banddft = numpy.fft.fftshift(numpy.fft.fft2(band))
                        else:
                            banddft = numpy.zeros(band.shape)
                        resdft += ( (numpy.power(-1+0j,0.5))**(nbands-1) * 
                                    banddft * anglemask * himask )
                    bandIdx += 1

        # apply lo0mask
        Xrcos += numpy.log2(2.0)
        lo0mask = numpy.array(pyPyrCcode.pointOp(log_rad.shape[0],
                                                 log_rad.shape[1], log_rad,
                                                 YIrcos.shape[0], YIrcos,
                                                 Xrcos[0], Xrcos[1]-Xrcos[0],
                                                 0))
        lo0mask = lo0mask.reshape(dims[0], dims[1])
        resdft = resdft * lo0mask
        
        # residual highpass subband
        hi0mask = pyPyrCcode.pointOp(log_rad.shape[0], log_rad.shape[1],
                                     log_rad, Yrcos.shape[0], Yrcos, Xrcos[0],
                                     Xrcos[1]-Xrcos[0], 0)
        hi0mask = numpy.array(hi0mask)
        hi0mask = hi0mask.reshape(resdft.shape[0], resdft.shape[1])
        if 0 in levs:
            hidft = numpy.fft.fftshift(numpy.fft.fft2(self.pyr[0]))
        else:
            hidft = numpy.zeros(self.pyr[0].shape)
        resdft += hidft * hi0mask

        outresdft = numpy.real(numpy.fft.ifft2(numpy.fft.ifftshift(resdft)))

        return outresdft

class SCFpyr(SFpyr):
    filt = ''
    edges = ''
    
    #constructor
    def __init__(self, *args):    # (image, height, order, twidth)
        self.pyrType = 'steerableFrequency'

        if len(args) > 0:
            self.image = args[0]
        else:
            print "First argument (image) is required."
            return

        #------------------------------------------------
        # defaults:

        max_ht = numpy.floor( numpy.log2( min(self.image.shape) ) ) - 2
        if len(args) > 1:
            if(args[1] > max_ht):
                print "Error: cannot build pyramid higher than %d levels." % (max_ht)
            ht = args[1]
        else:
            ht = max_ht
        ht = int(ht)
            
        if len(args) > 2:
            if args[2] > 15 or args[2] < 0:
                print "Warning: order must be an integer in the range [0,15]. Truncating."
                order = min( max(args[2],0), 15 )
            else:
                order = args[2]
        else:
            order = 3

        nbands = order+1

        if len(args) > 3:
            if args[3] <= 0:
                print "Warning: twidth must be positive. Setting to 1."
                twidth = 1
            else:
                twidth = args[3]
        else:
            twidth = 1

        #------------------------------------------------------
        # steering stuff:

        if nbands % 2 == 0:
            harmonics = numpy.array(range(nbands/2)) * 2 + 1
        else:
            harmonics = numpy.array(range((nbands-1)/2)) * 2

        steermtx = pyPyrUtils.steer2HarmMtx(harmonics, 
                                     numpy.pi*numpy.array(range(nbands))/nbands,
                                     'even')
        #------------------------------------------------------
        
        dims = numpy.array(self.image.shape)
        ctr = numpy.ceil((numpy.array(dims)+0.5)/2)
        
        (xramp, yramp) = numpy.meshgrid((numpy.array(range(1,dims[1]+1))-ctr[1])/
                                     (dims[1]/2), 
                                     (numpy.array(range(1,dims[0]+1))-ctr[0])/
                                     (dims[0]/2))
        angle = numpy.arctan2(yramp, xramp)
        log_rad = numpy.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        log_rad = numpy.log2(log_rad);

        ## Radial transition function (a raised cosine in log-frequency):
        (Xrcos, Yrcos) = pyPyrUtils.rcosFn(twidth, (-twidth/2.0), numpy.array([0,1]))
        Yrcos = numpy.sqrt(Yrcos)

        YIrcos = numpy.sqrt(1.0 - Yrcos**2)
        lo0mask = numpy.array(pyPyrCcode.pointOp(log_rad.shape[0],
                                                 log_rad.shape[1], log_rad,
                                                 YIrcos.shape[0], YIrcos,
                                                 Xrcos[0], Xrcos[1]-Xrcos[0],
                                                 0))

        imdft = numpy.fft.fftshift(numpy.fft.fft2(self.image))

        self.pyr = []
        self.pyrSize = []

        hi0mask = numpy.array(pyPyrCcode.pointOp(log_rad.shape[0],
                                                 log_rad.shape[1], log_rad,
                                                 Yrcos.shape[0], Yrcos,
                                                 Xrcos[0], Xrcos[1]-Xrcos[0],
                                                 0))
        hi0dft = imdft * hi0mask.reshape(imdft.shape[0], imdft.shape[1])
        hi0 = numpy.fft.ifft2(numpy.fft.ifftshift(hi0dft))

        self.pyr.append(numpy.real(hi0.copy()))
        self.pyrSize.append(hi0.shape)

        lo0mask = lo0mask.reshape(imdft.shape[0], imdft.shape[1])
        lodft = imdft * lo0mask

        for i in range(ht):
            bands = numpy.zeros((lodft.shape[0]*lodft.shape[1], nbands))
            bind = numpy.zeros((nbands, 2))
        
            Xrcos -= numpy.log2(2)

            lutsize = 1024
            Xcosn = numpy.pi * numpy.array(range(-(2*lutsize+1), (lutsize+2))) / lutsize

            order = nbands -1
            const = (2**(2*order))*(scipy.misc.factorial(order, exact=True)**2)/float(nbands*scipy.misc.factorial(2*order, exact=True))

            alfa = ( (numpy.pi+Xcosn) % (2.0*numpy.pi) ) - numpy.pi
            Ycosn = ( 2.0*numpy.sqrt(const) * (numpy.cos(Xcosn)**order) * 
                      (numpy.abs(alfa)<numpy.pi/2.0).astype(int) )
            himask = numpy.array(pyPyrCcode.pointOp(log_rad.shape[0],
                                                    log_rad.shape[1], log_rad,
                                                    Yrcos.shape[0], Yrcos,
                                                    Xrcos[0], Xrcos[1]-Xrcos[0],
                                                    0))
            himask = himask.reshape(lodft.shape[0], lodft.shape[1])

            for b in range(nbands):
                anglemask = numpy.array(pyPyrCcode.pointOp(angle.shape[0],
                                                           angle.shape[1], 
                                                           angle,
                                                           Ycosn.shape[0],
                                                           Ycosn, 
                                                           Xcosn[0]+numpy.pi*b/nbands, 
                                                           Xcosn[1]-Xcosn[0],
                                                           0))
                anglemask = anglemask.reshape(lodft.shape[0], lodft.shape[1])
                banddft = (cmath.sqrt(-1)**order) * lodft * anglemask * himask
                band = numpy.negative(numpy.fft.ifft2(numpy.fft.ifftshift(banddft)))
                self.pyr.append(band.copy())
                self.pyrSize.append(band.shape)

            dims = numpy.array(lodft.shape)
            ctr = numpy.ceil((dims+0.5)/2)
            lodims = numpy.ceil((dims-0.5)/2)
            #loctr = ceil((lodims+0.5)/2)
            loctr = numpy.ceil((lodims+0.5)/2)
            lostart = ctr - loctr
            loend = lostart + lodims

            log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = numpy.abs(numpy.sqrt(1.0 - Yrcos**2))
            lomask = numpy.array(pyPyrCcode.pointOp(log_rad.shape[0],
                                                    log_rad.shape[1], log_rad,
                                                    YIrcos.shape[0], YIrcos, 
                                                    Xrcos[0], Xrcos[1]-Xrcos[0],
                                                    0))
            lodft = lodft * lomask.reshape(lodft.shape[0], lodft.shape[1])

        lodft = numpy.fft.ifft2(numpy.fft.ifftshift(lodft))
        self.pyr.append(numpy.real(numpy.array(lodft).copy()))
        self.pyrSize.append(lodft.shape)

    # methods
    #def reconSCFpyr(self, *args):
    def reconPyr(self, *args):
        if len(args) > 0:
            levs = args[0]
        else:
            levs = 'all'

        if len(args) > 1:
            bands = args[1]
        else:
            bands = 'all'

        if len(args) > 2:
            if args[2] <= 0:
                print "Warning: twidth must be positive. Setting to 1."
                twidth = 1
            else:
                twidth = args[2]
        else:
            twidth = 1

        #-----------------------------------------------------------------

        pind = self.pyrSize
        Nsc = int(numpy.log2(pind[0][0] / pind[-1][0]))
        Nor = (len(pind)-2) / Nsc

        pyrIdx = 1
        for nsc in range(Nsc):
            firstBnum = nsc * Nor+2
            #dims = pind[firstBnum.astype(int)][:]
            dims = pind[firstBnum][:]
            ctr = (numpy.ceil((dims[0]+0.5)/2.0), numpy.ceil((dims[1]+0.5)/2.0)) #-1?
            ang = pyPyrUtils.mkAngle(dims, 0, ctr)
            ang[ctr[0]-1, ctr[1]-1] = -numpy.pi/2.0
            for nor in range(Nor):
                nband = nsc * Nor + nor + 1
                #ch = self.pyr[nband.astype(int)]
                ch = self.pyr[nband]
                ang0 = numpy.pi * nor / Nor
                xang = ((ang-ang0+numpy.pi) % (2.0*numpy.pi)) - numpy.pi
                amask = 2 * (numpy.abs(xang) < (numpy.pi/2.0)).astype(int) + (numpy.abs(xang) == (numpy.pi/2.0)).astype(int)
                amask[ctr[0]-1, ctr[1]-1] = 1
                amask[:,0] = 1
                amask[0,:] = 1
                amask = numpy.fft.fftshift(amask)
                ch = numpy.fft.ifft2(amask * numpy.fft.fft2(ch))  # 'Analytic' version
                # f = 1.000008  # With this factor the reconstruction SNR
                                # goes up around 6 dB!
                f = 1
                ch = f*0.5*numpy.real(ch)   # real part
                self.pyr[pyrIdx] = ch
                pyrIdx += 1

        res = self.reconSFpyr(levs, bands, twidth);
        #res = self.reconPyr(levs, bands, twidth);

        return res


class Lpyr(pyramid):
    filt = ''
    edges = ''
    height = ''

    # constructor
    def __init__(self, *args):    # (image, height, filter1, filter2, edges)
        self.pyrType = 'Laplacian'
        if len(args) > 0:
            self.image = args[0]
        else:
            print "pyr = Lpyr(image, height, filter1, filter2, edges)"
            print "First argument (image) is required"
            return

        if len(args) > 2:
            filt1 = args[2]
            if isinstance(filt1, basestring):
                filt1 = pyPyrUtils.namedFilter(filt1)
            elif len(filt1.shape) != 1 and ( filt1.shape[0] != 1 and
                                             filt1.shape[1] != 1 ):
                print "Error: filter1 should be a 1D filter (i.e., a vector)"
                return
        else:
            filt1 = pyPyrUtils.namedFilter('binom5')
        #if len(filt1.shape) == 1 or self.image.shape[0] == 1
        #    filt1 = filt1.reshape(1,len(filt1))
        if len(filt1.shape) == 1:
            filt1 = filt1.reshape(1,len(filt1))
        elif self.image.shape[0] == 1:
            filt1 = filt1.reshape(filt1.shape[1], filt1.shape[0])

        if len(args) > 3:
            filt2 = args[3]
            if isinstance(filt2, basestring):
                filt2 = pyPyrUtils.namedFilter(filt2)
            elif len(filt2.shape) != 1 and ( filt2.shape[0] != 1 and
                                            filt2.shape[1] != 1 ):
                print "Error: filter2 should be a 1D filter (i.e., a vector)"
                return
        else:
            filt2 = filt1

        maxHeight = 1 + pyPyrUtils.maxPyrHt(self.image.shape, filt1.shape)

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

        if len(args) > 4:
            edges = args[4]
        else:
            edges = "reflect1"

        # make pyramid
        self.pyr = []
        self.pyrSize = []
        pyrCtr = 0
        im = numpy.array(self.image).astype(float)
        if len(im.shape) == 1:
            im = im.reshape(im.shape[0], 1)
        los = {}
        los[self.height] = im
        # compute low bands
        #im_test = im
        for ht in range(self.height-1,0,-1):
            im_sz = im.shape
            filt1_sz = filt1.shape
            if im_sz[0] == 1:
                print 'flag lo 1'
                #lo2 = numpy.array( pyPyrCcode.corrDn(1, im_sz[1], im, 
                #                                     filt1_sz[0], filt1_sz[1], 
                #                                     filt1, edges, 1, 2, 0, 0, 
                #                                     int(math.ceil(im_sz[1]/2.0)), 1) ).T
                lo2 = pyPyrUtils.corrDn(image = im, filt = filt1, edges = edges,
                                        step = (1,2))
                lo2 = numpy.array(lo2)
                print 'lo2'
                print lo2
            elif len(im_sz) == 1 or im_sz[1] == 1:
                print 'flag lo 2'
                #lo2 = numpy.array( pyPyrCcode.corrDn(im_sz[0], 1, im, 
                #                                     filt1_sz[0], filt1_sz[1], 
                #                                     filt1, edges, 2, 1, 0, 0, 
                #                                     int(math.ceil(im_sz[0]/2.0)), 1) ).T
                lo2  = pyPyrUtils.corrDn(image = im, filt = filt1, edges = edges,
                                         step = (2,1))
                lo2 = numpy.array(lo2)
                print 'lo2'
                print lo2
            else:
                # orig version
                #lo = numpy.array( pyPyrCcode.corrDn(im_sz[1], im_sz[0], im, 
                #                                    filt1_sz[0], filt1_sz[1], 
                #                                    filt1, edges, 2, 1, 0, 0, 
                #                                    im_sz[0], im_sz[1]) )
                ##lo = numpy.array(lo).reshape(math.ceil(im_sz[0]/1.0), 
                ##                          math.ceil(im_sz[1]/2.0), 
                ##                          order='C')
                #lo2 = numpy.array( pyPyrCcode.corrDn(int(math.ceil(im_sz[0]/1.0)), 
                #                                     int(math.ceil(im_sz[1]/2.0)), 
                #                                     lo.T, filt1_sz[0], 
                #                                     filt1_sz[1], filt1, edges, 
                #                                     2, 1, 0, 0, im_sz[0], 
                #                                     im_sz[1]) ).T
                ##lo2 = numpy.array(lo2).reshape(math.ceil(im_sz[0]/2.0), 
                ##                            math.ceil(im_sz[1]/2.0), 
                ##                            order='F')
                # new version
                lo = pyPyrUtils.corrDn(image = im, filt = filt1.T, edges = edges,
                                       step = (1,2), start = (0,0))
                lo = numpy.array(lo)
                lo2 = pyPyrUtils.corrDn(image = lo, filt = filt1, edges = edges,
                                        step = (2,1), start = (0,0))
                lo2 = numpy.array(lo2)

            los[ht] = lo2
                
            im = lo2

        #self.pyr[self.height-1] = lo2
        #self.pyrSize[self.height-1] = lo2.shape
        # adjust shape if 1D if needed
        self.pyr.append(lo2.copy())
        self.pyrSize.append(lo2.shape)

        # compute hi bands
        im = self.image
        for ht in range(self.height, 1, -1):
            im = los[ht-1]
            im_sz = los[ht-1].shape
            filt2_sz = filt2.shape
            if len(im_sz) == 1 or im_sz[1] == 1:
                print 'flag hi 1'
                #hi2 = pyPyrCcode.upConv(im_sz[0], im_sz[1], im, filt2_sz[0],
                #                        filt2_sz[1], filt2, edges, 2, 1, 0, 0,
                #                        los[ht].shape[0], los[ht].shape[1]).T
                hi2 = pyPyrUtils.upConv(image = im, filt = filt2.T, 
                                        edges = edges, step = (1,2),
                                        stop = (los[ht].shape[1],
                                                los[ht].shape[0]))
                print 'hi2'
                print hi2
            elif im_sz[0] == 1:
                print 'flag hi 2'
                #hi2 = pyPyrCcode.upConv(im_sz[0], im_sz[1], im, filt2_sz[1],
                #                        filt2_sz[0], filt2, edges, 1, 2, 0, 0,
                #                        los[ht].shape[0], los[ht].shape[1]).T
                # wrong for circualr edges?!
                #hi2 = pyPyrUtils.upConv(image = im, filt = filt2.T, 
                #                        edges = edges, step = (2,1), 
                #                        stop = (los[ht].shape[1],
                #                                los[ht].shape[0]))
                hi2 = pyPyrUtils.upConv(image = im, filt = filt2.T, 
                                        edges = edges, step = (2,1), 
                                        stop = (los[ht].shape[1],
                                                los[ht].shape[0]))
                print 'hi2'
                print hi2
            else:
                ## orig code
                #hi = pyPyrCcode.upConv(im_sz[0], im_sz[1], im.T, filt2_sz[0],
                #                       filt2_sz[1], filt2, edges, 2, 1, 0, 0,
                #                       los[ht].shape[0], im_sz[1]).T
                ##hi = numpy.array(hi).reshape(los[ht].shape[0], im_sz[1], order='F')
                #int_sz = hi.shape
                #hi2 = pyPyrCcode.upConv(los[ht].shape[0], im_sz[1], hi.T, 
                #                        filt2_sz[1], filt2_sz[0], filt2, edges,
                #                        1, 2, 0, 0, los[ht].shape[0],
                #                        los[ht].shape[1]).T
                ##hi2 = numpy.array(hi2).reshape(los[ht].shape[0], los[ht].shape[1],
                ##                            order='F')
                ## new code
                hi = pyPyrUtils.upConv(image = im.T, filt = filt2, edges = edges,
                                       step = (2,1), stop = (los[ht].shape[0], 
                                                             im_sz[1])).T
                hi2 = pyPyrUtils.upConv(image = hi.T, filt = filt2.T, 
                                        edges = edges, step = (1,2), 
                                        stop = (los[ht].shape[0], 
                                                los[ht].shape[1])).T
                                       

            hi2 = los[ht] - hi2
            #self.pyr[pyrCtr] = hi2
            #self.pyrSize[pyrCtr] = hi2.shape
            self.pyr.insert(pyrCtr, hi2.copy())
            self.pyrSize.insert(pyrCtr, hi2.shape)
            pyrCtr += 1

    # methods
    # return concatenation of all levels of 1d pyramid
    def catBands(self, *args):
        outarray = numpy.array([]).reshape((1,0))
        for i in range(self.height):
            tmp = self.band(i).T
            outarray = numpy.concatenate((outarray, tmp), axis=1)
        return outarray

    # set a pyramid value
    def set_old(self, *args):
        if len(args) != 3:
            print 'Error: three input parameters required:'
            print '  set(band, element, value)'
        print 'band=%d  element=%d  value=%d' % (args[0],args[1],args[2])
        print self.pyr[args[0]].shape
        self.pyr[args[0]][args[1]] = args[2] 

    def set(self, *args):
        if len(args) != 3:
            print 'Error: three input parameters required:'
            print '  set(band, element(tuple), value)'
        #print 'band=%d  element=%d  value=%d' % (args[0],args[1],args[2])
        print self.pyr[args[0]].shape
        self.pyr[args[0]][args[1][0]][args[1][1]] = args[2] 

    #def reconLpyr(self, *args):
    def reconPyr(self, *args):
        if len(args) > 0:
            levs = numpy.array(args[0])
        else:
            levs = 'all'

        if len(args) > 1:
            filt2 = args[1]
        else:
            filt2 = 'binom5'

        if len(args) > 2:
            edges = args[2]
        else:
            edges = 'reflect1';

        maxLev = self.height

        if levs == 'all':
            levs = range(0,maxLev)
        else:
            if (levs > maxLev-1).any():
                print ( "Error: level numbers must be in the range [0, %d]." % 
                        (maxLev-1) )
                return

        if isinstance(filt2, basestring):
            filt2 = pyPyrUtils.namedFilter(filt2)
        else:
            if len(filt2.shape) == 1:
                filt2 = filt2.reshape(1, len(filt2))

        res = []
        lastLev = -1
        for lev in range(maxLev-1, -1, -1):
            if lev in levs and len(res) == 0:
                res = self.band(lev)
            elif len(res) != 0:
                res_sz = res.shape
                new_sz = self.band(lev).shape
                filt2_sz = filt2.shape
                if res_sz[0] == 1:
                    #print 'recon flag 1'
                    #hi2 = pyPyrCcode.upConv(new_sz[0], res_sz[1], res.T,
                    #                        filt2_sz[1], filt2_sz[0], filt2,
                    #                        edges, 1, 2, 0, 0, new_sz[0],
                    #                        new_sz[1]).T
                    hi2 = pyPyrUtils.upConv(image = res, filt = filt2,
                                            edges = edges, step = (2,1), 
                                            stop = (new_sz[1], new_sz[0]))
                elif res_sz[1] == 1:
                    #print 'recon flag 2'
                    #hi2 = pyPyrCcode.upConv(new_sz[0], res_sz[1], res.T,
                    #                        filt2_sz[0], filt2_sz[1], filt2,
                    #                        edges, 2, 1, 0, 0, new_sz[0],
                    #                        new_sz[1]).T
                    hi2 = pyPyrUtils.upConv(image = res, filt = filt2.T,
                                            edges = edges, step = (1,2), 
                                            stop = (new_sz[1], new_sz[0]))
                else:
                    # orig code
                    #hi = pyPyrCcode.upConv(res_sz[0], res_sz[1], res.T,
                    #                       filt2_sz[0], filt2_sz[1], filt2,
                    #                       edges, 2, 1, 0, 0, new_sz[0],
                    #                       res_sz[1]).T
                    #hi2 = pyPyrCcode.upConv(new_sz[0], res_sz[1], hi.T,
                    #                        filt2_sz[1], filt2_sz[0], filt2,
                    #                        edges, 1, 2, 0, 0, new_sz[0],
                    #                        new_sz[1]).T
                    # new code
                    hi = pyPyrUtils.upConv(image = res, filt = filt2.T, 
                                           edges = edges, step = (1,2), 
                                           stop = (res_sz[1], new_sz[0]))
                    hi2 = pyPyrUtils.upConv(image = hi.T, filt = filt2.T, 
                                            edges = edges, step = (1,2),
                                            stop = new_sz).T
                if lev in levs:
                    bandIm = self.band(lev)
                    bandIm_sz = bandIm.shape
                    res = hi2 + bandIm
                else:
                    res = hi2
        return res                           
                
    def pyrLow(self):
        return numpy.array(self.band(self.height-1))

    #def showPyr(self, *args):
    # options for disp are 'qt' and 'nb'
    def showPyr(self, pRange = None, gap = 1, scale = None, disp = 'qt'):
        if ( len(self.band(0).shape) == 1 or self.band(0).shape[0] == 1 or
             self.band(0).shape[1] == 1 ):
            oned = 1
        else:
            oned = 0

        #if oned == 1:
        #    pRange = 'auto1'
        #else:
        #    pRange = 'auto2'
        
        #if len(args) > 1:
        #    gap = args[1]
        #else:
        #    gap = 1

        #if len(args) > 2:
        #    scale = args[2]
        #else:
        #    if oned == 1:
        #        scale = math.sqrt(2)
        #    else:
        #        scale = 2
        
        if pRange == None and oned == 1:
            pRange = 'auto1'
        elif pRange == None and oned == 0:
            pRange = 'auto2'

        if scale == None and oned == 1:
            scale = math.sqrt(2)
        elif scale == None and oned == 0:
            scale = 2

        #nind = self.height - 1
        nind = self.height
            
        # auto range calculations
        if pRange == 'auto1':
            pRange = numpy.zeros((nind,1))
            mn = 0.0
            mx = 0.0
            for bnum in range(nind):
                band = self.band(bnum)
                band /= numpy.power(scale, bnum-1)
                pRange[bnum] = numpy.power(scale, bnum-1)
                bmn = numpy.amin(band)
                bmx = numpy.amax(band)
                mn = numpy.amin([mn, bmn])
                mx = numpy.amax([mx, bmx])
            if oned == 1:
                pad = (mx-mn)/12       # magic number
                mn -= pad
                mx += pad
            pRange = numpy.outer(pRange, numpy.array([mn, mx]))
            band = self.pyrLow()
            mn = numpy.amin(band)
            mx = numpy.amax(band)
            if oned == 1:
                pad = (mx-mn)/12
                mn -= pad
                mx += pad
            pRange[nind-1,:] = [mn, mx];
        elif pRange == 'indep1':
            pRange = numpy.zeros((nind,1))
            for bnum in range(nind):
                band = self.band(bnum)
                mn = numpy.amin(band)
                mx = numpy.amax(band)
                if oned == 1:
                    pad = (mx-mn)/12;
                    mn -= pad
                    mx += pad
                pRange[bnum,:] = numpy.array([mn, mx])
        elif pRange == 'auto2':
            pRange = numpy.zeros((nind,1))
            sqsum = 0
            numpixels = 0
            for bnum in range(0, nind-1):
                band = self.band(bnum)
                band /= numpy.power(scale, bnum)
                sqsum += numpy.sum( numpy.power(band, 2) )
                numpixels += numpy.prod(band.shape)
                pRange[bnum,:] = numpy.power(scale, bnum)
            stdev = math.sqrt( sqsum / (numpixels-1) )
            pRange = numpy.outer( pRange, numpy.array([-3*stdev, 3*stdev]) )
            band = self.pyrLow()
            av = numpy.mean(band)
            stdev = numpy.std(band)
            pRange[nind-1,:] = numpy.array([av-2*stdev, av+2*stdev]);#by ref? safe?
        elif pRange == 'indep2':
            pRange = numpy.zeros(nind,2)
            for bnum in range(0,nind-1):
                band = self.band(bnum)
                stdev = numpy.std(band)
                pRange[bnum,:] = numpy.array([-3*stdev, 3*stdev])
            band = self.pyrLow()
            av = numpy.mean(band)
            stdev = numpy.std(band)
            pRange[nind,:] = numpy.array([av-2*stdev, av+2*stdev])
        elif isinstance(pRange, basestring):
            print "Error: band range argument: %s" % (pRange)
            return
        elif pRange.shape[0] == 1 and pRange.shape[1] == 2:
            scales = numpy.power( numpy.array( range(0,nind) ), scale)
            pRange = numpy.outer( scales, pRange )
            band = self.pyrLow()
            pRange[nind,:] = ( pRange[nind,:] + numpy.mean(band) - 
                               numpy.mean(pRange[nind,:]) )

        # draw
        if oned == 1:
            fig = matplotlib.pyplot.figure()
            ax0 = fig.add_subplot(nind, 1, 0)
            ax0.set_frame_on(False)
            ax0.get_xaxis().tick_bottom()
            ax0.get_xaxis().tick_top()
            ax0.get_yaxis().tick_right()
            ax0.get_yaxis().tick_left()
            ax0.get_yaxis().set_visible(False)
            for bnum in range(0,nind):
                pylab.subplot(nind, 1, bnum+1)
                pylab.plot(numpy.array(range(numpy.amax(self.band(bnum).shape))).T, 
                           self.band(bnum).T)
                ylim(pRange[bnum,:])
                xlim((0,self.band(bnum).shape[1]-1))
            matplotlib.pyplot.show()
        else:
            colormap = matplotlib.cm.Greys_r
            # skipping background calculation. needed?

            # compute positions of subbands:
            llpos = numpy.ones((nind, 2)).astype(float)
            dirr = numpy.array([-1.0, -1.0])
            ctr = numpy.array([self.band(0).shape[0]+1+gap, 1]).astype(float)
            sz = numpy.array([0.0, 0.0])
            for bnum in range(nind):
                prevsz = sz
                sz = self.band(bnum).shape
                
                # determine center position of new band:
                ctr = ( ctr + gap*dirr/2.0 + dirr * 
                        numpy.floor( (prevsz+(dirr<0).astype(int))/2.0 ) )
                dirr = numpy.dot(dirr,numpy.array([ [0, -1], [1, 0] ])) # ccw rotation
                ctr = ( ctr + gap*dirr/2 + dirr * 
                        numpy.floor( (sz+(dirr<0).astype(int)) / 2.0) )
                llpos[bnum,:] = ctr - numpy.floor(numpy.array(sz))/2.0 
            # make position list positive, and allocate appropriate image
            llpos = llpos - numpy.ones((nind,1))*numpy.min(llpos)
            pind = range(self.height)
            for i in pind:
                pind[i] = self.band(i).shape
            urpos = llpos + pind
            d_im = numpy.ones((numpy.max(urpos), numpy.max(urpos))) * 255
            
            # paste bands into image, (im-r1)*(nshades-1)/(r2-r1) + 1.5
            nshades = 256
            for bnum in range(nind):
                mult = (nshades-1) / (pRange[bnum,1]-pRange[bnum,0])
                d_im[llpos[bnum,0]:urpos[bnum,0], llpos[bnum,1]:urpos[bnum,1]]=(
                    mult*self.band(bnum) + (1.5-mult*pRange[bnum,0]) )
                # layout works
                #d_im[llpos[bnum,0]:urpos[bnum,0],llpos[bnum,1]:urpos[bnum,1]]=(
                    #(bnum+1)*10 )
            
            ##ppu.showIm(d_im)  # works
            ##plt.imshow(d_im, cm.Greys_r)
            #plt.imshow(d_im[:self.band(0).shape[0]][:], cm.Greys_r)
            #ax = plt.gca()
            #ax.set_yticks([])
            #ax.set_xticks([])
            # FIX: need a mode to switch between above and below display
            if disp == 'nb':
                JBhelpers.showIm(d_im[:self.band(0).shape[0]][:])
            elif disp == 'qt':
                pyPyrUtils.showIm(d_im[:self.band(0).shape[0]][:])

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
            filt = pyPyrUtils.namedFilter('binom5')
            if self.image.shape[0] == 1:
                filt = filt.reshape(1,5)
            else:
                filt = filt.reshape(5,1)

        maxHeight = 1 + pyPyrUtils.maxPyrHt(self.image.shape, filt.shape)

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

        #self.pyr[pyrCtr] = im
        #self.pyrSize[pyrCtr] = im.shape
        self.pyr.append(im.copy())
        self.pyrSize.append(im.shape)
        pyrCtr += 1

        for ht in range(self.height-1,0,-1):
            im_sz = im.shape
            filt_sz = filt.shape
            if im_sz[0] == 1:
                #lo2 = numpy.array( corrDn(1, im_sz[1], im, filt_sz[0], filt_sz[1],
                #                       filt, edges, 1, 2, 0, 0, 1, im_sz[1]) )
                filt = filt[0,:]
                ## orig code
                #lo2 = numpy.array( pyPyrCcode.corrDn(im_sz[0], 1, im, 
                #                                     filt_sz[0], filt_sz[1],
                #                                     filt, edges, 2, 1, 0, 0, 
                #                                     im_sz[0], 1) )
                ##lo2 = numpy.array(lo2).reshape(1, im_sz[1]/2, order='C')
                ## new code
                lo2 = pyPyrUitls.corrDn(image = im, filt = filt, step = (1,2))
                lo2 = numpy.array(lo2)
            elif len(im_sz) == 1 or im_sz[1] == 1:
                #lo2 = numpy.array( corrDn(im_sz[0], 1, im, filt_sz[0], filt_sz[1],
                #                       filt, edges, 2, 1, 0, 0, im_sz[0], 1) )
                ## orig code
                #lo2 = numpy.array( pyPyrCcode.corrDn(1, im_sz[1], im, 
                #                                 filt_sz[0], filt_sz[1],
                #                                    filt, edges, 1, 2, 0, 0, 1,
                #                                     im_sz[1]) )
                #print lo2
                ##lo2 = numpy.array(lo2).reshape(im_sz[0]/2, 1, order='C')
                ## new code
                lo2 = pyPyrUtils.corrDn(image = im, filt = filt1, step = (2,1))
                lo2 = numpy.array(lo2)
            else:
                ## orig version
                #lo = numpy.array( pyPyrCcode.corrDn(im_sz[1], im_sz[0], im, 
                #                                    filt_sz[0], filt_sz[1], 
                #                                    filt, edges, 2, 1, 0, 0, 
                #                                    im_sz[0], im_sz[1]) )
                #print lo
                ##lo = numpy.array(lo).reshape(math.ceil(im_sz[0]/1.0), 
                ##                          math.ceil(im_sz[1]/2.0), 
                ##                          order='C')
                #lo2 = numpy.array( pyPyrCcode.corrDn(int(math.ceil(im_sz[0]/1.0)), 
                #int(math.ceil(im_sz[1]/2.0)), 
                #                                     lo.T, filt_sz[0], 
                #                                     filt_sz[1], filt, edges, 
                #                                     2, 1, 0, 0, im_sz[0], 
                #                                     im_sz[1]) ).T
                #print lo2
                ##lo2 = numpy.array(lo2).reshape(math.ceil(im_sz[0]/2.0), 
                ##                            math.ceil(im_sz[1]/2.0), 
                ##                            order='F')
                lo = pyPyrUtils.corrDn(image = im, filt = filt.T, 
                                       step = (1,2), start = (0,0))
                lo = numpy.array(lo)
                lo2 = pyPyrUtils.corrDn(image = lo, filt = filt, 
                                        step = (2,1), start = (0,0))
                lo2 = numpy.array(lo2)                

            #self.pyr[pyrCtr] = lo2
            #self.pyrSize[pyrCtr] = lo2.shape
            self.pyr.append(lo2.copy())
            self.pyrSize.append(lo2.shape)
            pyrCtr += 1

            im = lo2
        
    # methods
'''
class Wpyr_new(Lpyr):
    filt = ''
    edges = ''
    height = ''
    
    #constructor
    def __init__(self, *args):    # (image, height, order, twidth)
        self.pyr = []
        self.pyrSize = []
        self.pyrType = 'wavelet'

        if len(args) > 0:
            im = args[0]
        else:
            print "First argument (image) is required."
            return

        #------------------------------------------------
        # defaults:

        if len(args) > 2:
            filt = args[1]
        else:
            filt = "qmf9"
        if isinstance(filt, basestring):
            filt = pyPyrUtils.namedFilter(filt)

        if len(filt.shape) != 1 and filt.shape[0] != 1 and filt.shape[1] != 1:
            print "Error: filter should be 1D (i.e., a vector)";
            return
        hfilt = pyPyrUtils.modulateFlip(filt).T

        if len(args) > 3:
            edges = args[2]
        else:
            edges = "reflect1"

        # Stagger sampling if filter is odd-length:
        if filt.shape[0] % 2 == 0:
            stag = 2
        else:
            stag = 1
        #print 'stag = %d' % (stag)

        im_sz = im.shape
        if len(im.shape) == 1 or im.shape[1] == 1:
            im = im.reshape(1, im.shape[0])

        if len(filt.shape) == 1 or filt.shape[1] == 1:
            filt = filt.reshape(1, filt.shape[0])
        filt_sz = filt.shape

        max_ht = pyPyrUtils.maxPyrHt(im_sz, filt_sz)

        if len(args) > 1:
            ht = args[1]
            if ht == 'auto':
                ht = max_ht
            elif(ht > max_ht):
                print "Error: cannot build pyramid higher than %d levels." % (max_ht)
        else:
            ht = max_ht
        ht = int(ht)
        self.height = ht + 1  # used with showPyr() method

        for lev in range(ht):
            #print "lev = %d" % (lev)
            im_sz = im.shape
            #if len(im.shape) == 1:
            #    im_sz = (1, im.shape[0])
            #elif im.shape[1] == 1:
            #    im_sz = (im.shape[1], im.shape[0])
            #print "im_sz"
            #print im_sz
            if len(im_sz) == 1 or im_sz[1] == 1:
                lolo = numpy.array( pyPyrCcode.corrDn(im_sz[0], im_sz[1], im.T,
                                                      filt_sz[0], filt_sz[1], 
                                                      filt, edges, 2, 1, stag-1,
                                                      0) )
                hihi = numpy.array( pyPyrCcode.corrDn(im_sz[0], im_sz[1], im.T,
                                                      filt_sz[0], filt_sz[1], 
                                                      hfilt, edges, 2, 1, 1, 0))
            elif im_sz[0] == 1:
                #lolo = numpy.array( corrDn(im_sz[0], im_sz[1], im, filt_sz[0], 
                #                        filt_sz[1], filt, edges, 1, 2, 0, 
                #                        stag-1) )
                lolo = numpy.array( pyPyrCcode.corrDn(im_sz[0], im_sz[1], im, 
                                                      filt_sz[0], filt_sz[1], 
                                                      filt, edges, 1, 2, 0, 1) ).T
                hihi = numpy.array( pyPyrCcode.corrDn(im_sz[0], im_sz[1], im, 
                                                      filt_sz[0], filt_sz[1], 
                                                      hfilt, edges, 1, 2, 0, 1) ).T
            else:
                lo = numpy.array( pyPyrCcode.corrDn(im.shape[0], im.shape[1], 
                                                    im.T, filt.shape[0], 1, 
                                                    filt, edges, 2, 1, stag-1, 
                                                    0) ).T
                #lo = lo.reshape(math.ceil(im.shape[0]/2.0), 
                #                math.ceil(im.shape[1]/stag), order='F')
                hi = numpy.array( pyPyrCcode.corrDn(im.shape[0], im.shape[1], 
                                                    im.T, hfilt.shape[0], 1, 
                                                    hfilt, edges, 2, 1, 1, 
                                                    0) ).T
                #hi = hi.reshape(math.floor(im.shape[0]/2.0), im.shape[1], 
                #                order='F')
                lolo = numpy.array( pyPyrCcode.corrDn(lo.shape[0], lo.shape[1],
                                                      lo.T, 1, filt.shape[0], 
                                                      filt, edges, 1, 2, 0, 
                                                      stag-1) ).T 
                #lolo = lolo.reshape(math.ceil(lo.shape[0]/float(stag)), 
                #                    math.ceil(lo.shape[1]/2.0), order='F')
                lohi = numpy.array( pyPyrCcode.corrDn(hi.shape[0], hi.shape[1],
                                                      hi.T, 1, filt.shape[0], 
                                                      filt, edges, 1, 2, 0,
                                                      stag-1) ).T
                #lohi = lohi.reshape(hi.shape[0], math.ceil(hi.shape[1]/2.0), 
                #                    order='F')
                hilo = numpy.array( pyPyrCcode.corrDn(lo.shape[0], lo.shape[1],
                                                      lo.T, 1, hfilt.shape[0], 
                                                      hfilt, edges, 1, 2, 0, 
                                                      1) ).T
                #hilo = hilo.reshape(lo.shape[0], math.floor(lo.shape[1]/2.0), 
                #                    order='F')
                hihi = numpy.array( pyPyrCcode.corrDn(hi.shape[0], hi.shape[1],
                                                      hi.T, 1, hfilt.shape[0], 
                                                      hfilt, edges, 1, 2, 0, 
                                                      1) ).T
                #hihi = hihi.reshape(hi.shape[0], math.floor(hi.shape[1]/2.0), 
                #                    order='F')
            if im_sz[0] == 1 or im_sz[1] == 1:
                self.pyr.append(hihi)
                self.pyrSize.append(hihi.shape)
            else:
                self.pyr.append(lohi)
                self.pyrSize.append(lohi.shape)
                self.pyr.append(hilo)
                self.pyrSize.append(hilo.shape)
                self.pyr.append(hihi)
                self.pyrSize.append(hihi.shape)
            im = lolo
        self.pyr.append(lolo)
        self.pyrSize.append(lolo.shape)

    # methods

    def wpyrHt(self):
        if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or 
             self.pyrSize[0][1] == 1 ): 
            nbands = 1
        else:
            nbands = 3

        ht = (len(self.pyrSize)-1)/float(nbands)

        return int(ht)


    #def reconWpyr(self, *args):
    def reconPyr(self, *args):
        # Optional args
        if len(args) > 0:
            filt = args[0]
        else:
            filt = 'qmf9'
            
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

        #------------------------------------------------------

        print self.pyrSize
        maxLev = self.wpyrHt() + 1
        print "maxLev = %d" % maxLev
        if levs == 'all':
            levs = numpy.array(range(maxLev))
        else:
            tmpLevs = []
            for l in levs:
                tmpLevs.append((maxLev-1)-l)
            levs = numpy.array(tmpLevs)
            if (levs > maxLev).any():
                print "Error: level numbers must be in the range [0, %d]" % (maxLev)
        allLevs = numpy.array(range(maxLev))

        print "levs:"
        print levs
        
        if bands == "all":
            bands = numpy.array(range(3))
        else:
            bands = numpy.array(bands)
            if (bands < 0).any() or (bands > 2).any():
                print "Error: band numbers must be in the range [0,2]."
        
        if isinstance(filt, basestring):
            filt = pyPyrUtils.namedFilter(filt)

        print "filt"
        print filt
        hfilt = pyPyrUtils.modulateFlip(filt)
        print "hfilt"
        print hfilt

        # for odd-length filters, stagger the sampling lattices:
        if len(filt) % 2 == 0:
            stag = 2
        else:
            stag = 1
        print "stag = %d" % (stag)

        #if 0 in levs:
        #    res = self.pyr[len(self.pyr)-1]
        #else:
        #    res = numpy.zeros(self.pyr[len(self.pyr)-1].shape)
        #print res

        print "pyrSize[0]:"
        print self.pyrSize[0]
        print "pyrSize[3]:"
        print self.pyrSize[3]

        print 'len pyrSize = %d' % (len(self.pyr))

        idx = len(self.pyrSize)-1

        print "levs:"
        print levs
        print "bands:"
        print bands

        #for lev in levs:
        for lev in allLevs:
            print "starting levs loop lev = %d" % lev

            if lev == 0:
                if 0 in levs:
                    res = self.pyr[len(self.pyr)-1]
                else:
                    res = numpy.zeros(self.pyr[len(self.pyr)-1].shape)
                print res
            elif lev > 0:
                # compute size of result image: assumes critical sampling
                resIdx = len(self.pyrSize)-(3*(lev-1))-3
                print "resIdx = %d" % resIdx
                res_sz = self.pyrSize[resIdx]
                print 'res_sz'
                print res_sz
                print 'self.pyrSize'
                print self.pyrSize
                if res_sz[0] == 1:
                    res_sz = (1, sum([i[1] for i in self.pyrSize]))
                elif res_sz[1] == 1:
                    res_sz = (sum([i[0] for i in self.pyrSize]), 1)
                else:
                #horizontal + vertical bands
                    res_sz = (self.pyrSize[resIdx][0]+self.pyrSize[resIdx-1][0],
                              self.pyrSize[resIdx][1]+self.pyrSize[resIdx-1][1])
                    lres_sz = numpy.array([self.pyrSize[resIdx][0], res_sz[1]])
                    hres_sz = numpy.array([self.pyrSize[resIdx-1][0], res_sz[1]])

                imageIn = res
                #fp = open('tmp.txt', 'a')
                #fp.write('%d ires\n' % lev)
                #fp.close()
                if res_sz[0] == 1:
                    #res = upConv(nres, filt', edges, [1 2], [1 stag], res_sz);
                    res = pyPyrCcode.upConv(res.shape[0], res.shape[1], res,
                                            filt.shape[1], filt.shape[0], filt,
                                            edges, 1, 2, 0, stag-1, res_sz[0],
                                            res_sz[1])
                elif res_sz[1] == 1:
                    #res = upConv(nres, filt, edges, [2 1], [stag 1], res_sz);
                    res = pyPyrCcode.upConv(res.shape[0], res.shape[1], res,
                                            filt.shape[0], filt.shape[1], filt,
                                            edges, 2, 1, stag-1, 0, res_sz[0],
                                            res_sz[1])
                else:
                    print "filt shape = %d %d\n" % (filt.shape[0], 
                                                    filt.shape[1])
                    ires = pyPyrCcode.upConv(imageIn.shape[1], imageIn.shape[0],
                                             imageIn.T, filt.shape[1],
                                             filt.shape[0], filt, edges, 1, 2,
                                             0, stag-1, lres_sz[0], lres_sz[1])
                    ires = numpy.array(ires).T
                    print "%d ires" % (lev)
                    print ires
                    print ires.shape
                    print "hfilt"
                    print hfilt
                    #fp = open('tmp.txt', 'a')
                    #fp.write("%d res\n" % (lev))
                    #fp.close()
                    res = pyPyrCcode.upConv(ires.shape[1], ires.shape[0],
                                            ires.T, filt.shape[0],
                                            filt.shape[1], filt, edges, 2, 1,
                                            stag-1, 0, res_sz[0], res_sz[1]).T
                    res = numpy.array(res)
                    print "%d res" % (lev)
                    print res

                idx = resIdx - 1
                print 'starting bands idx = %d' % (idx)
                #if res_sz[0] == 1:
                #    #upConv(pyrBand(pyr,ind,1), hfilt', edges, [1 2], [1 2], 
                #    #  res_sz, res);
                #    res = upConv(self.band(0).shape[0], self.band(0).shape[1], 
                #                 self.band(0), hfilt.shape[0], hfilt.shape[1], 
                #                 hfilt, edges, 1, 2, 0, 1, res_sz[0], 
                #                 res_sz[1], res)
                #elif res_sz[1] == 1:
                #    #upConv(pyrBand(pyr,ind,1), hfilt, edges, [2 1], [2 1], 
                #    #  res_sz, res);
                #    res = upConv(self.band(0).shape[0], self.band(0).shape[1],
                #                 self.band(0), hfilt.shape[1], hfilt.shape[0], 
                #                 hfilt, edges, 2, 2, 1, 0, res_sz[0], 
                #                 res_sz[1], res)
                #else:
                    #if 0 in bands:
                if 0 in bands and lev in levs:
                    print "0 band"
                    print "idx = %d" % idx
                    print "resIdx = %d" % resIdx
                    print "input band"
                    print self.band(idx)
                        #fp = open('tmp.txt', 'a')
                        #fp.write("ires 0 band\n")
                        #fp.close()
                        # broken for even filters
                        #ires = upConv(self.band(idx).shape[0], 
                        #              self.band(idx).shape[1],
                        #              self.band(idx).T, 
                        #              filt.shape[1], filt.shape[0], filt, 
                        #              edges, 1, 2, stag-1, 0, hres_sz[0], 
                        #              hres_sz[1])
                        # works with even filter
                    ires = pyPyrCcode.upConv(self.band(idx).shape[0], 
                                             self.band(idx).shape[1],
                                             self.band(idx).T, filt.shape[1],
                                             filt.shape[0], filt, edges, 1, 2,
                                             0, stag-1, hres_sz[0], hres_sz[1])
                    ires = numpy.array(ires).T
                        #ires = ires.reshape(hres_sz[1], hres_sz[0]).T
                    print "ires"
                    print ires
                    print ires.shape

                    print "pre upconv res"
                    print res
                        #fp = open('tmp.txt', 'a')
                        #fp.write("res 0 band\n")
                        #fp.close()
                    print "pre res size"
                    print res.shape
                    res = pyPyrCcode.upConv(ires.shape[0], ires.shape[1],
                                            ires.T, hfilt.shape[1],
                                            hfilt.shape[0], hfilt, edges, 2, 1,
                                            1, 0, res_sz[0], res_sz[1], res.T)
                    res = numpy.array(res).T
                    print "res size %d %d" % (res_sz[1], res_sz[0])
                    print "post res size"
                    print res.shape
                    print "post upconv res"
                    print res
                idx += 1
                    #if 1 in bands:
                if 1 in bands and lev in levs:
                    print "1 band"
                    print "idx = %d" % idx
                    print "lres_sz"
                    print lres_sz
                    print lres_sz[0]
                    print lres_sz[1]
                    print "self.band(idx)"
                    print self.band(idx)
                    print self.band(idx).shape
                    print hfilt
                        #fp = open('tmp.txt', 'a')
                        #fp.write("ires 1 band\n")
                        #fp.close()
                    ires = upConv(self.band(idx).shape[0], 
                                  self.band(idx).shape[1], self.band(idx).T, 
                                  hfilt.shape[0], hfilt.shape[1], hfilt, 
                                  edges, 1, 2, 0, 1, lres_sz[0], lres_sz[1])
                    ires = numpy.array(ires).T
                    print "ires"
                    print ires
                    print ires.shape
                    print filt.shape
                    print "pre res"
                    print res
                    print filt
                    print edges
                    print "stag = %d" % stag
                        #fp = open('tmp.txt', 'a')
                        #fp.write("res 1 band\n")
                        #fp.close()
                    res = pyPyrCcode.upConv(ires.shape[0], ires.shape[1],
                                            ires.T, filt.shape[0],
                                            filt.shape[1], filt, edges, 2, 1,
                                            stag-1, 0, res_sz[0], res_sz[1],
                                            res.T)
                    res = numpy.array(res).T
                        #res = res.reshape(res_sz[1], res_sz[0]).T
                    print "res"
                    print res
                idx += 1
                    #if 2 in bands:
                if 2 in bands and lev in levs:
                    print "2 band"
                    print "idx = %d" % idx
                    print "input image"
                    print self.band(idx)
                        #fp = open('tmp.txt', 'a')
                        #fp.write("ires 2 band\n")
                        #fp.close()
                    ires = pyPyrCcode.upConv(self.band(idx).shape[0],
                                             self.band(idx).shape[1],
                                             self.band(idx).T, hfilt.shape[0],
                                             hfilt.shape[1], hfilt, edges, 1, 2,
                                             0, 1, hres_sz[0], hres_sz[1])
                    ires = numpy.array(ires).T
                        #ires = ires.reshape(hres_sz[1], hres_sz[0]).T
                    print "ires"
                    print ires
                    print "pre res"
                    print res
                        #fp = open('tmp.txt', 'a')
                        #fp.write("res 2 band\n")
                        #fp.close()
                    res = pyPyrCcode.upConv(ires.shape[1], ires.shape[0],
                                            ires.T, hfilt.shape[1],
                                            hfilt.shape[0], hfilt, edges, 2, 1,
                                            1, 0, res_sz[0], res_sz[1], res.T)
                    res = numpy.array(res).T
                    print "res"
                    print res
                idx += 1
                    # need to jump back n bands in the idx each loop
                idx -= 2*len(bands)
        return res

class Wpyr_bak(pyramid):
    filt = ''
    edges = ''
    
    #constructor
    def __init__(self, *args):    # (image, height, order, twidth)
        self.pyr = []
        self.pyrSize = []
        self.pyrType = 'wavelet'

        if len(args) > 0:
            im = args[0]
        else:
            print "First argument (image) is required."
            return

        #------------------------------------------------
        # defaults:

        if len(args) > 2:
            filt = args[1]
        else:
            filt = "qmf9"
        if isinstance(filt, basestring):
            filt = pyPyrUtils.namedFilter(filt)

        if len(filt.shape) != 1 and filt.shape[0] != 1 and filt.shape[1] != 1:
            print "Error: filter should be 1D (i.e., a vector)";
            return
        hfilt = pyPyrUtils.modulateFlip(filt).T

        if len(args) > 3:
            edges = args[2]
        else:
            edges = "reflect1"

        # Stagger sampling if filter is odd-length:
        if filt.shape[0] % 2 == 0:
            stag = 2
        else:
            stag = 1

        im_sz = im.shape
        if len(im.shape) == 1:
            im_sz = (1, im.shape[0])
        elif im.shape[1] == 1:
            im_sz = (im.shape[1], im.shape[0])
        print "im_sz"
        print im_sz
        #if len(im.shape) == 1 or im.shape[1] == 1:
        #    im = im.reshape(1, im.shape[0])

        if len(filt.shape) == 1:
            filt_sz = (1, filt.shape[0])
        #elif filt.shape[1] == 1:  # FIX for 1D: breaks other tests
        #    filt_sz = (filt.shape[1], filt.shape[0])
        else:
            filt_sz = filt.shape
        print "filt_sz"
        print filt_sz
        max_ht = pyPyrUtils.maxPyrHt(im_sz, filt_sz)
        print "max_ht = %d" % (max_ht)
        if len(args) > 1:
            ht = args[1]
            if ht == 'auto':
                ht = max_ht
            elif(ht > max_ht):
                print "Error: cannot build pyramid higher than %d levels." % (max_ht)
        else:
            ht = max_ht
        ht = int(ht)

        for lev in range(ht):
            print "lev = %d" % (lev)
            im_sz = im.shape
            if len(im.shape) == 1:
                im_sz = (1, im.shape[0])
            elif im.shape[1] == 1:
                im_sz = (im.shape[1], im.shape[0])
                print "im_sz"
                print im_sz
            if len(im_sz) == 1 or im_sz[1] == 1:
                lolo = numpy.array( pyPyrCcode.corrDn(im_sz[0], im_sz[1], im.T,
                                                      filt_sz[0], filt_sz[1],
                                                      filt, edges, 2, 1, stag-1,
                                                      0) )
                hihi = numpy.array( pyPyrCcode.corrDn(im_sz[0], im_sz[1], im.T,
                                                      filt_sz[0], filt_sz[1], 
                                                      hfilt, edges, 2, 1, 1, 0))
            elif im_sz[0] == 1:
                lolo = numpy.array( pyPyrCcode.corrDn(im_sz[0], im_sz[1], im, 
                                                      filt_sz[0], filt_sz[1], 
                                                      filt, edges, 1, 2, 0, 
                                                      stag-1) )
                hihi = numpy.array( pyPyrCcode.corrDn(im_sz[0], im_sz[1], im, 
                                                      filt_sz[0], filt_sz[1], 
                                                      hfilt, edges, 1, 2, 0, 1))
            else:
                lo = numpy.array( pyPyrCcode.corrDn(im.shape[0], im.shape[1], 
                                                    im.T, filt.shape[0], 1, 
                                                    filt, edges, 2, 1, stag-1, 
                                                    0) ).T
                #lo = lo.reshape(math.ceil(im.shape[0]/2.0), 
                #                math.ceil(im.shape[1]/stag), order='F')
                hi = numpy.array( pyPyrCcode.corrDn(im.shape[0], im.shape[1], 
                                                    im.T, hfilt.shape[0], 1, 
                                                    hfilt, edges, 2, 1, 1,
                                                    0) ).T
                #hi = hi.reshape(math.floor(im.shape[0]/2.0), im.shape[1], 
                #                order='F')
                lolo = numpy.array( pyPyrCcode.corrDn(lo.shape[0], lo.shape[1],
                                                      lo.T, 1, filt.shape[0], 
                                                      filt, edges, 1, 2, 0,
                                                      stag-1) ).T 
                #lolo = lolo.reshape(math.ceil(lo.shape[0]/float(stag)), 
                #                    math.ceil(lo.shape[1]/2.0), order='F')
                lohi = numpy.array( pyPyrCcode.corrDn(hi.shape[0], hi.shape[1],
                                                      hi.T, 1, filt.shape[0],
                                                      filt, edges, 1, 2, 0, 
                                                      stag-1) ).T
                #lohi = lohi.reshape(hi.shape[0], math.ceil(hi.shape[1]/2.0), 
                #                    order='F')
                hilo = numpy.array( pyPyrCcode.corrDn(lo.shape[0], lo.shape[1],
                                                      lo.T, 1, hfilt.shape[0],
                                                      hfilt, edges, 1, 2, 0, 
                                                      1) ).T
                #hilo = hilo.reshape(lo.shape[0], math.floor(lo.shape[1]/2.0), 
                #                    order='F')
                hihi = numpy.array( pyPyrCcode.corrDn(hi.shape[0], hi.shape[1],
                                                      hi.T, 1, hfilt.shape[0],
                                                      hfilt, edges, 1, 2, 0, 
                                                      1) ).T
                #hihi = hihi.reshape(hi.shape[0], math.floor(hi.shape[1]/2.0), 
                #                    order='F')
            if im_sz[0] == 1 or im_sz[1] == 1:
                self.pyr.append(hihi)
                self.pyrSize.append(hihi.shape)
            else:
                self.pyr.append(lohi)
                self.pyrSize.append(lohi.shape)
                self.pyr.append(hilo)
                self.pyrSize.append(hilo.shape)
                self.pyr.append(hihi)
                self.pyrSize.append(hihi.shape)
            im = lolo
        self.pyr.append(lolo)
        self.pyrSize.append(lolo.shape)

    # methods

    def wpyrHt(self):
        if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or 
             self.pyrSize[0][1] == 1 ): 
            nbands = 1
        else:
            nbands = 3

        ht = (len(self.pyrSize)-1)/float(nbands)

        return ht


    #def reconWpyr(self, *args):
    def reconPyr(self, *args):
        # Optional args
        if len(args) > 0:
            filt = args[0]
        else:
            filt = 'qmf9'
            
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

        #------------------------------------------------------

        print self.pyrSize
        maxLev = self.wpyrHt() + 1
        print "maxLev = %d" % maxLev
        if levs == 'all':
            levs = numpy.array(range(maxLev))
        else:
            tmpLevs = []
            for l in levs:
                tmpLevs.append((maxLev-1)-l)
            levs = numpy.array(tmpLevs)
            if (levs > maxLev).any():
                print "Error: level numbers must be in the range [0, %d]" % (maxLev)
        allLevs = numpy.array(range(maxLev))

        print "levs:"
        print levs
        
        if bands == "all":
            bands = numpy.array(range(3))
        else:
            bands = numpy.array(bands)
            if (bands < 0).any() or (bands > 2).any():
                print "Error: band numbers must be in the range [0,2]."
        
        if isinstance(filt, basestring):
            filt = pyPyrUtils.namedFilter(filt)

        print "filt"
        print filt
        hfilt = pyPyrUtils.modulateFlip(filt)
        print "hfilt"
        print hfilt

        # for odd-length filters, stagger the sampling lattices:
        if len(filt) % 2 == 0:
            stag = 2
        else:
            stag = 1
        print "stag = %d" % (stag)

        #if 0 in levs:
        #    res = self.pyr[len(self.pyr)-1]
        #else:
        #    res = numpy.zeros(self.pyr[len(self.pyr)-1].shape)
        #print res

        print "pyrSize[0]:"
        print self.pyrSize[0]
        print "pyrSize[3]:"
        print self.pyrSize[3]

        print 'len pyrSize = %d' % (len(self.pyr))

        idx = len(self.pyrSize)-1

        print "levs:"
        print levs
        print "bands:"
        print bands

        #for lev in levs:
        for lev in allLevs:
            print "starting levs loop lev = %d" % lev

            if lev == 0:
                if 0 in levs:
                    res = self.pyr[len(self.pyr)-1]
                else:
                    res = numpy.zeros(self.pyr[len(self.pyr)-1].shape)
                print res
            elif lev > 0:
                # compute size of result image: assumes critical sampling
                resIdx = len(self.pyrSize)-(3*(lev-1))-3
                res_sz = self.pyrSize[resIdx]
                print "resIdx = %d" % resIdx
                if res_sz[0] == 1:
                    res_sz[1] = sum(self.pyrSize[:,1])
                elif res_sz[1] == 1:
                    res_sz[0] = sum(self.pyrSize[:,0])
                else:
                #horizontal + vertical bands
                    res_sz = (self.pyrSize[resIdx][0]+self.pyrSize[resIdx-1][0],
                              self.pyrSize[resIdx][1]+self.pyrSize[resIdx-1][1])
                    lres_sz = numpy.array([self.pyrSize[resIdx][0], res_sz[1]])
                    hres_sz = numpy.array([self.pyrSize[resIdx-1][0], res_sz[1]])

                print 'pyrSizes'
                print self.pyrSize[resIdx]
                print self.pyrSize[resIdx+1]
                print self.pyrSize[resIdx-1]
                print "res_sz"
                print res_sz
                print "hres_sz"
                print hres_sz
                print "lres_sz"
                print lres_sz

                # FIX: how is this changed with subsets of levs?
                #if lev <= 1:
                #    print "lev = %d" % lev
                #    print "levs"
                #    print levs
                #    if lev in levs:
                #        print "idx = %d" % (idx)
                #        print self.pyrSize
                #        imageIn = self.band(idx)
                #    else:
                #        imageIn = numpy.zeros(self.band(idx).shape)
                #    print "input image"
                #    print imageIn
                #else:
                #    imageIn = res
                imageIn = res
                #fp = open('tmp.txt', 'a')
                #fp.write('%d ires\n' % lev)
                #fp.close()
                print "filt shape = %d %d\n" % (filt.shape[0], filt.shape[1])
                ires = pyPyrCcode.upConv(imageIn.shape[1], imageIn.shape[0],
                                         imageIn.T, filt.shape[1],
                                         filt.shape[0], filt, edges, 1, 2, 0,
                                         stag-1, lres_sz[0], lres_sz[1])
                ires = numpy.array(ires).T
                #ires = ires.reshape(lres_sz[1], lres_sz[0]).T
                print "%d ires" % (lev)
                print ires
                print ires.shape
                print "hfilt"
                print hfilt
                #fp = open('tmp.txt', 'a')
                #fp.write("%d res\n" % (lev))
                #fp.close()
                res = pyPyrCcode.upConv(ires.shape[1], ires.shape[0], ires.T, 
                                        filt.shape[0], filt.shape[1], filt, 
                                        edges, 2, 1, stag-1, 0, res_sz[0], 
                                        res_sz[1]).T
                res = numpy.array(res)
                #res = res.reshape(res_sz[1], res_sz[0]).T
                print "%d res" % (lev)
                print res

                idx = resIdx - 1

                ### FIX: do we need stag here?! Check with even size filter
                #if 0 in bands:
                if 0 in bands and lev in levs:
                    print "0 band"
                    print "idx = %d" % idx
                    print "resIdx = %d" % resIdx
                    print "input band"
                    print self.band(idx)
                    #fp = open('tmp.txt', 'a')
                    #fp.write("ires 0 band\n")
                    #fp.close()
                    # broken for even filters
                    #ires = upConv(self.band(idx).shape[0], 
                    #              self.band(idx).shape[1],
                    #              self.band(idx).T, 
                    #              filt.shape[1], filt.shape[0], filt, 
                    #              edges, 1, 2, stag-1, 0, hres_sz[0], 
                    #              hres_sz[1])
                    # works with even filter
                    ires = pyPyrCcode.upConv(self.band(idx).shape[0], 
                                             self.band(idx).shape[1],
                                             self.band(idx).T, filt.shape[1],
                                             filt.shape[0], filt, edges, 1, 2,
                                             0, stag-1, hres_sz[0], hres_sz[1])
                    ires = numpy.array(ires).T
                    #ires = ires.reshape(hres_sz[1], hres_sz[0]).T
                    print "ires"
                    print ires
                    print ires.shape

                    print "pre upconv res"
                    print res
                    #fp = open('tmp.txt', 'a')
                    #fp.write("res 0 band\n")
                    #fp.close()
                    print "pre res size"
                    print res.shape
                    res = pyPyrCcode.upConv(ires.shape[0], ires.shape[1],
                                            ires.T, hfilt.shape[1],
                                            hfilt.shape[0], hfilt, edges, 2, 1,
                                            1, 0, res_sz[0], res_sz[1], res.T)
                    res = numpy.array(res).T
                    print "res size %d %d" % (res_sz[1], res_sz[0])
                    print "post res size"
                    print res.shape
                    print "post upconv res"
                    print res
                idx += 1
                #if 1 in bands:
                if 1 in bands and lev in levs:
                    print "1 band"
                    print "idx = %d" % idx
                    print "lres_sz"
                    print lres_sz
                    print lres_sz[0]
                    print lres_sz[1]
                    print "self.band(idx)"
                    print self.band(idx)
                    print self.band(idx).shape
                    print hfilt
                    #fp = open('tmp.txt', 'a')
                    #fp.write("ires 1 band\n")
                    #fp.close()
                    ires = pyPyrCcode.upConv(self.band(idx).shape[0], 
                                             self.band(idx).shape[1], 
                                             self.band(idx).T, hfilt.shape[0],
                                             hfilt.shape[1], hfilt, edges,
                                             1, 2, 0, 1, lres_sz[0], lres_sz[1])
                    ires = numpy.array(ires).T
                    #ires = ires.reshape(lres_sz[1], lres_sz[0]).T
                    print "ires"
                    print ires
                    print ires.shape
                    print filt.shape
                    print "pre res"
                    print res
                    print filt
                    print edges
                    print "stag = %d" % stag
                    # FIX: stag is not correct here. does this need to flip
                    #      because python is 0 based?
                    #fp = open('tmp.txt', 'a')
                    #fp.write("res 1 band\n")
                    #fp.close()
                    # works with odd filters
                    #res = upConv(ires.shape[0], ires.shape[1], ires.T, 
                    #             filt.shape[0], filt.shape[1], filt, 
                    #             edges, 2, 1, 0, 0, 
                    #             res_sz[0], res_sz[1], res.T)
                    # works with even filters
                    #res = upConv(ires.shape[0], ires.shape[1], ires.T, 
                    #             filt.shape[0], filt.shape[1], filt, 
                    #             edges, 2, 1, 1, 0, 
                    #             res_sz[0], res_sz[1], res.T)
                    res = pyPyrCcode.upConv(ires.shape[0], ires.shape[1],
                                            ires.T, filt.shape[0],
                                            filt.shape[1], filt, edges, 2, 1,
                                            stag-1, 0, res_sz[0], res_sz[1],
                                            res.T)
                    res = numpy.array(res).T
                    #res = res.reshape(res_sz[1], res_sz[0]).T
                    print "res"
                    print res
                idx += 1
                #if 2 in bands:
                if 2 in bands and lev in levs:
                    print "2 band"
                    print "idx = %d" % idx
                    print "input image"
                    print self.band(idx)
                    #fp = open('tmp.txt', 'a')
                    #fp.write("ires 2 band\n")
                    #fp.close()
                    ires = pyPyrCcode.upConv(self.band(idx).shape[0],
                                             self.band(idx).shape[1],
                                             self.band(idx).T, hfilt.shape[0],
                                             hfilt.shape[1], hfilt, edges, 1, 2,
                                             0, 1, hres_sz[0], hres_sz[1])
                    ires = numpy.array(ires).T
                    #ires = ires.reshape(hres_sz[1], hres_sz[0]).T
                    print "ires"
                    print ires
                    print "pre res"
                    print res
                    #fp = open('tmp.txt', 'a')
                    #fp.write("res 2 band\n")
                    #fp.close()
                    res = pyPyrCcode.upConv(ires.shape[1], ires.shape[0],
                                            ires.T, hfilt.shape[1],
                                            hfilt.shape[0], hfilt, edges, 2, 1,
                                            1, 0, res_sz[0], res_sz[1], res.T)
                    res = numpy.array(res).T
                    print "res"
                    print res
                idx += 1
                # need to jump back n bands in the idx each loop
                idx -= 2*len(bands)
        return res
'''
class Wpyr(Lpyr):
    filt = ''
    edges = ''
    height = ''

    #constructor
    def __init__(self, *args):    # (image, height, order, twidth)
        self.pyr = []
        self.pyrSize = []
        self.pyrType = 'wavelet'

        if len(args) > 0:
            im = args[0]
        else:
            print "First argument (image) is required."
            return

        #------------------------------------------------
        # defaults:

        if len(args) > 2:
            filt = args[2]
        else:
            filt = "qmf9"
        if isinstance(filt, basestring):
            filt = pyPyrUtils.namedFilter(filt)

        if len(filt.shape) != 1 and filt.shape[0] != 1 and filt.shape[1] != 1:
            print "Error: filter should be 1D (i.e., a vector)";
            return
        hfilt = pyPyrUtils.modulateFlip(filt).T

        if len(args) > 3:
            edges = args[3]
        else:
            edges = "reflect1"

        # Stagger sampling if filter is odd-length:
        if filt.shape[0] % 2 == 0:
            stag = 2
        else:
            stag = 1

        #im_sz = im.shape
        #if len(im.shape) == 1:
        #    im_sz = (1, im.shape[0])
        #elif im.shape[1] == 1:
        #    im_sz = (im.shape[1], im.shape[0])
        #print "im_sz"
        #print im_sz
        #if len(im.shape) == 1 or im.shape[1] == 1:
        #    im = im.reshape(1, im.shape[0])

        #if len(filt.shape) == 1:
        #    filt_sz = (1, filt.shape[0])
        #else:
        #    filt_sz = filt.shape
        #print "filt_sz"
        #print filt_sz
        # if 1D filter, match to image dimensions
        if len(filt.shape) == 1 or filt.shape[1] == 1:
            if im.shape[0] == 1:
                filt = filt.reshape(1, filt.shape[0])
            elif im.shape[1] == 1:
                filt = filt.reshape(filt.shape[0], 1)
        #elif filt.shape[0] == 1:
        #    if im.shape[0] == 1:
        #        filt = filt.reshape(1, filt.shape[1])
        #    elif im.shape[1] == 1:
        #        filt = filt.reshape(filt.shape[1], 1)

        #max_ht = ppu.maxPyrHt(im_sz, filt_sz)
        max_ht = pyPyrUtils.maxPyrHt(im.shape, filt.shape)
        #print "max_ht = %d" % (max_ht)
        if len(args) > 1:
            ht = args[1]
            if ht == 'auto':
                ht = max_ht
            elif(ht > max_ht):
                print "Error: cannot build pyramid higher than %d levels." % (max_ht)
        else:
            ht = max_ht
        ht = int(ht)
        self.height = ht + 1  # used with showPyr() method
        #im_test = im
        for lev in range(ht):
            #print "lev = %d" % (lev)
            #im_sz = im.shape
            #if len(im.shape) == 1:
            #    im_sz = (1, im.shape[0])
            #elif im.shape[1] == 1:
            #    im_sz = (im.shape[1], im.shape[0])
            #    print "im_sz"
            #    print im_sz
            #if len(im.shape) == 1 or im.shape[1] == 1:
            #    im = im.reshape(1, im.shape[0])
            #    print 'flag 0'
            if len(im.shape) == 1 or im.shape[1] == 1:
                lolo = pyPyrUtils.corrDn(image = im, filt = filt, 
                                         edges = edges, step = (2,1), 
                                         start = (stag-1,0))
                lolo = numpy.array(lolo)
                hihi = pyPyrUtils.corrDn(image = im, filt = hfilt, 
                                         edges = edges, step = (2,1), 
                                         start = (1, 0))
                hihi = numpy.array(hihi)
            #elif im_sz[0] == 1:
            elif im.shape[0] == 1:
                lolo = pyPyrUtils.corrDn(image = im, filt = filt, 
                                              edges = edges, step = (1,2), 
                                              start = (0, stag-1))
                lolo = numpy.array(lolo)
                hihi = pyPyrUtils.corrDn(image = im, filt = hfilt.T, 
                                              edges = edges, step = (1,2), 
                                              start = (0,1))
                hihi = numpy.array(hihi)
            else:
                ## orig code
                #lo = numpy.array( pyPyrCcode.corrDn(im.shape[0], im.shape[1], 
                #                                    im.T, filt.shape[0], 1, 
                #                                    filt, edges, 2, 1, stag-1,
                #                                    0) ).T
                ##lo = lo.reshape(math.ceil(im.shape[0]/2.0), 
                ##                math.ceil(im.shape[1]/stag), order='F')
                #hi = numpy.array( pyPyrCcode.corrDn(im.shape[0], im.shape[1], 
                #                                    im.T, hfilt.shape[0], 1, 
                #                                    hfilt, edges, 2, 1, 1,
                #                                    0) ).T
                ##hi = hi.reshape(math.floor(im.shape[0]/2.0), im.shape[1], 
                ##                order='F')
                #lolo = numpy.array( pyPyrCcode.corrDn(lo.shape[0], lo.shape[1],
                #                                      lo.T, 1, filt.shape[0],
                #                                      filt, edges, 1, 2, 0, 
                #                                      stag-1) ).T 
                ##lolo = lolo.reshape(math.ceil(lo.shape[0]/float(stag)), 
                ##                    math.ceil(lo.shape[1]/2.0), order='F')
                #lohi = numpy.array( pyPyrCcode.corrDn(hi.shape[0], hi.shape[1],
                #                                      hi.T, 1, filt.shape[0],
                #                                      filt, edges, 1, 2, 0, 
                #                                      stag-1) ).T
                ##lohi = lohi.reshape(hi.shape[0], math.ceil(hi.shape[1]/2.0), 
                ##                    order='F')
                #hilo = numpy.array( pyPyrCcode.corrDn(lo.shape[0], lo.shape[1],
                #                                      lo.T, 1, hfilt.shape[0],
                #                                      hfilt, edges, 1, 2, 0, 
                #                                      1) ).T
                ##hilo = hilo.reshape(lo.shape[0], math.floor(lo.shape[1]/2.0), 
                ##                    order='F')
                #hihi = numpy.array( pyPyrCcode.corrDn(hi.shape[0], hi.shape[1],
                #                                      hi.T, 1, hfilt.shape[0],
                #                                      hfilt, edges, 1, 2, 0, 
                #                                      1) ).T
                ##hihi = hihi.reshape(hi.shape[0], math.floor(hi.shape[1]/2.0), 
                ##                    order='F')
                ## orig code - working version
                '''
                print 'filt'
                print filt
                print 'stag = %d' % (stag)
                lo = numpy.array( pyPyrCcode.corrDn(im.shape[0], im.shape[1], 
                                                    im.T, filt.shape[0], 1, 
                                                    filt, edges, 2, 1, stag-1,
                                                    0) ).T
                print 'lo'
                print lo
                hi = numpy.array( pyPyrCcode.corrDn(im.shape[0], im.shape[1], 
                                                    im.T, hfilt.shape[0], 1, 
                                                    hfilt, edges, 2, 1, 1,
                                                    0) ).T
                print 'hi'
                print hi
                lolo = numpy.array( pyPyrCcode.corrDn(lo.shape[0], lo.shape[1],
                                                      lo.T, 1, filt.shape[0],
                                                      filt, edges, 1, 2, 0, 
                                                      stag-1) ).T 
                print 'lolo'
                print lolo
                lohi = numpy.array( pyPyrCcode.corrDn(hi.shape[0], hi.shape[1],
                                                      hi.T, 1, filt.shape[0],
                                                      filt, edges, 1, 2, 0, 
                                                      stag-1) ).T
                print 'lohi'
                print lohi
                hilo = numpy.array( pyPyrCcode.corrDn(lo.shape[0], lo.shape[1],
                                                      lo.T, 1, hfilt.shape[0],
                                                      hfilt, edges, 1, 2, 0, 
                                                      1) ).T
                print 'hilo'
                print hilo
                hihi = numpy.array( pyPyrCcode.corrDn(hi.shape[0], hi.shape[1],
                                                      hi.T, 1, hfilt.shape[0],
                                                      hfilt, edges, 1, 2, 0, 
                                                      1) ).T
                print 'hihi'
                print hihi
                
                ## new code - work first time through loop then is wrong?!!
                
                print 'filt'
                print filt
                lo_test = pyPyrUtils.corrDn(image = im_test, filt = filt, 
                                            edges = edges, step = (2,1), 
                                            start = (stag-1,0))
                lo_test = numpy.array(lo_test)
                print 'lo_test'
                print lo_test
                hi_test = pyPyrUtils.corrDn(image = im_test, filt = hfilt, 
                                            edges = edges, step = (2,1), 
                                            start = (1,0))
                hi_test = numpy.array(hi_test)
                print 'hi_test'
                print hi_test
                lolo_test = pyPyrUtils.corrDn(image = lo_test, filt = filt.T, 
                                              edges = edges, step = (1,2), 
                                              start = (0, stag-1))
                lolo_test = numpy.array(lolo_test)
                print 'lolo_test'
                print lolo_test
                lohi_test = pyPyrUtils.corrDn(image = hi_test, filt = filt.T, 
                                              edges = edges, step = (1,2),
                                              start = (0,stag-1))
                lohi_test = numpy.array(lohi_test)
                print 'lohi_test'
                print lohi_test
                # close
                hilo_test = pyPyrUtils.corrDn(image = lo_test, filt = hfilt.T, 
                                              edges = edges, step = (1,2), 
                                              start = (0,1))
                hilo_test = numpy.array(hilo_test)
                print 'hilo_test'
                print hilo_test
                hihi_test = pyPyrUtils.corrDn(image = hi_test, filt = hfilt.T, 
                                              edges = edges, step = (1,2), 
                                              start = (0,1))
                hihi_test = numpy.array(hihi_test)
                print 'hihi_test'
                print hihi_test
                '''
                ### another try  -- correct for all unit tests
                #print 'filt'
                #print filt
                #print 'stag = %d' % (stag)
                lo = pyPyrUtils.corrDn(image = im, filt = filt, 
                                       edges = edges, step = (2,1), 
                                       start = (stag-1,0))
                lo = numpy.array(lo)
                #print 'lo_test'
                #print lo_test
                hi = pyPyrUtils.corrDn(image = im, filt = hfilt, 
                                       edges = edges, step = (2,1), 
                                       start = (1,0))
                hi = numpy.array(hi)
                #print 'hi_test'
                #print hi_test
                lolo = pyPyrUtils.corrDn(image = lo, filt = filt.T, 
                                         edges = edges, step = (1,2), 
                                         start = (0, stag-1))
                lolo = numpy.array(lolo)
                #print 'lolo_test'
                #print lolo_test
                lohi = pyPyrUtils.corrDn(image = hi, filt = filt.T, 
                                         edges = edges, step = (1,2),
                                         start = (0,stag-1))
                lohi = numpy.array(lohi)
                #print 'lohi_test'
                #print lohi_test
                # close
                hilo = pyPyrUtils.corrDn(image = lo, filt = hfilt.T, 
                                         edges = edges, step = (1,2), 
                                         start = (0,1))
                hilo = numpy.array(hilo)
                #print 'hilo_test'
                #print hilo_test
                hihi = pyPyrUtils.corrDn(image = hi, filt = hfilt.T, 
                                         edges = edges, step = (1,2), 
                                         start = (0,1))
                hihi = numpy.array(hihi)
                #print 'hihi_test'
                #print hihi_test

            #if im_sz[0] == 1 or im_sz[1] == 1:
            if im.shape[0] == 1 or im.shape[1] == 1:
                self.pyr.append(hihi)
                self.pyrSize.append(hihi.shape)
            else:
                self.pyr.append(lohi)
                self.pyrSize.append(lohi.shape)
                self.pyr.append(hilo)
                self.pyrSize.append(hilo.shape)
                self.pyr.append(hihi)
                self.pyrSize.append(hihi.shape)
            im = lolo.copy()
            #im_test = lolo.copy()
        self.pyr.append(lolo)
        self.pyrSize.append(lolo.shape)

    # methods

    def wpyrHt(self):
        if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or 
             self.pyrSize[0][1] == 1 ): 
            nbands = 1
        else:
            nbands = 3

        ht = (len(self.pyrSize)-1)/float(nbands)

        return ht

    def numBands(self):
        if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or 
             self.pyrSize[0][1] == 1 ): 
            nbands = 1
        else:
            nbands = 3
        return nbands

    '''
    def reconWpyr_old(self, *args):
        # Optional args
        if len(args) > 0:
            filt = args[0]
        else:
            filt = 'qmf9'
            
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

        #------------------------------------------------------

        print self.pyrSize
        maxLev = int(self.wpyrHt() + 1)
        print "maxLev = %d" % maxLev
        if levs == 'all':
            levs = numpy.array(range(maxLev))
        else:
            tmpLevs = []
            for l in levs:
                tmpLevs.append((maxLev-1)-l)
            levs = numpy.array(tmpLevs)
            if (levs > maxLev).any():
                print "Error: level numbers must be in the range [0, %d]" % (maxLev)
        allLevs = numpy.array(range(maxLev))

        print "levs:"
        print levs
        
        if bands == "all":
            if ( len(self.band(0)) == 1 or self.band(0).shape[0] == 1 or 
                 self.band(0).shape[1] == 1 ):
                bands = numpy.array([0]);
            else:
                bands = numpy.array(range(3))
        else:
            bands = numpy.array(bands)
            if (bands < 0).any() or (bands > 2).any():
                print "Error: band numbers must be in the range [0,2]."
        
        if isinstance(filt, basestring):
            filt = pyPyrUtils.namedFilter(filt)

        print "filt"
        print filt
        hfilt = pyPyrUtils.modulateFlip(filt)
        print "hfilt"
        print hfilt

        # for odd-length filters, stagger the sampling lattices:
        if len(filt) % 2 == 0:
            stag = 2
        else:
            stag = 1
        print "stag = %d" % (stag)

        #if 0 in levs:
        #    res = self.pyr[len(self.pyr)-1]
        #else:
        #    res = numpy.zeros(self.pyr[len(self.pyr)-1].shape)
        #print res

        print "pyrSize[0]:"
        print self.pyrSize[0]
        print "pyrSize[3]:"
        print self.pyrSize[3]

        print 'len pyrSize = %d' % (len(self.pyr))

        idx = len(self.pyrSize)-1
        print 'idx = %d' % (idx)

        print "levs:"
        print levs
        print "bands:"
        print bands

        #for lev in levs:
        for lev in allLevs:
            print "starting levs loop lev = %d" % lev

            if lev == 0:
                if 0 in levs:
                    res = self.pyr[len(self.pyr)-1]
                else:
                    res = numpy.zeros(self.pyr[len(self.pyr)-1].shape)
                print res
            elif lev > 0:
                # compute size of result image: assumes critical sampling
                if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or
                     self.pyrSize[0][1] == 1 ):
                    resIdx = len(self.pyrSize)-2
                else:
                    resIdx = len(self.pyrSize)-(3*(lev-1))-3
                print "resIdx = %d" % resIdx
                res_sz = self.pyrSize[resIdx]
                print 'pre res_sz'
                print res_sz
                if res_sz[0] == 1:
                    #res_sz[1] = sum(self.pyrSize[:,1])
                    res_sz = (res_sz[0], sum([x[1] for x in self.pyrSize]))
                elif res_sz[1] == 1:
                    #res_sz[0] = sum(self.pyrSize[:,0])
                    res_sz = (res_sz[0], sum([x[0] for x in self.pyrSize]))
                else:
                    #horizontal + vertical bands
                    res_sz = (self.pyrSize[resIdx][0]+self.pyrSize[resIdx-1][0],
                              self.pyrSize[resIdx][1]+self.pyrSize[resIdx-1][1])
                    lres_sz = numpy.array([self.pyrSize[resIdx][0], res_sz[1]])
                    hres_sz = numpy.array([self.pyrSize[resIdx-1][0], res_sz[1]])
                print 'post res_sz'
                print res_sz

                print 'pyrSizes'
                print self.pyrSize[resIdx]
                #print self.pyrSize[resIdx+1]
                #print self.pyrSize[resIdx-1]
                print "res_sz"
                print res_sz
                #print "hres_sz"
                #print hres_sz
                #print "lres_sz"
                #print lres_sz

                # FIX: how is this changed with subsets of levs?
                #if lev <= 1:
                #    print "lev = %d" % lev
                #    print "levs"
                #    print levs
                #    if lev in levs:
                #        print "idx = %d" % (idx)
                #        print self.pyrSize
                #        imageIn = self.band(idx)
                #    else:
                #        imageIn = numpy.zeros(self.band(idx).shape)
                #    print "input image"
                #    print imageIn
                #else:
                #    imageIn = res
                imageIn = res
                #fp = open('tmp.txt', 'a')
                #fp.write('%d ires\n' % lev)
                #fp.close()
                if res_sz[0] == 1:
                    #res = upConv(nres, filt', edges, [1 2], [1 stag], res_sz);
                    print 'flag 1'
                    res = pyPyrCcode.upConv(imageIn.shape[0], imageIn.shape[1],
                                            imageIn, filt.shape[1],
                                            filt.shape[0], filt, edges, 1, 2, 0,
                                            stag-1, res_sz[0], res_sz[1])
                    res = numpy.array(res).T
                elif res_sz[1] == 1:
                    #res = upConv(nres, filt, edges, [2 1], [stag 1], res_sz);
                    print 'flag 2'
                    res = pyPyrCcode.upConv(imageIn.shape[0], imageIn.shape[1],
                                            imageIn, filt.shape[0],
                                            filt.shape[1], filt, edges, 2, 1,
                                            stag-1, 1, res_sz[0], res_sz[1])
                    res = numpy.array(res).T
                else:
                    print "filt shape = %d %d\n" % (filt.shape[0], 
                                                    filt.shape[1])
                    ires = pyPyrCcode.upConv(imageIn.shape[1], imageIn.shape[0],
                                             imageIn.T, filt.shape[1],
                                             filt.shape[0], filt, edges, 1, 2,
                                             0, stag-1, lres_sz[0], lres_sz[1])
                    ires = numpy.array(ires).T
                    #ires = ires.reshape(lres_sz[1], lres_sz[0]).T
                    print "%d ires" % (lev)
                    print ires
                    print ires.shape
                    print "hfilt"
                    print hfilt
                    #fp = open('tmp.txt', 'a')
                    #fp.write("%d res\n" % (lev))
                    #fp.close()
                    res = pyPyrCcode.upConv(ires.shape[1], ires.shape[0],
                                            ires.T, filt.shape[0], 
                                            filt.shape[1], filt, edges, 2, 1,
                                            stag-1, 0, res_sz[0], res_sz[1]).T
                    res = numpy.array(res)
                    #res = res.reshape(res_sz[1], res_sz[0]).T
                    print "%d res" % (lev)
                    print res

                print 'self.pyrSize'
                print self.pyrSize[0]
                if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or
                     self.pyrSize[0][1] == 1 ):
                    print 'idx flag 1'
                    idx = idx
                else:
                    print 'idx flag 2'
                    idx = resIdx - 1
                print '1### idx = %d' % (idx)
                if res_sz[0] ==1 and lev in levs:
                    #upConv(pyrBand(pyr,ind,1), hfilt', edges, [1 2], [1 2], res_sz, res);
                    print '1d band'
                    print self.band(idx)
                    res = pyPyrCcode.upConv(self.band(idx).shape[0], 
                                            self.band(idx).shape[1],
                                            self.band(idx), hfilt.shape[0],
                                            hfilt.shape[1], hfilt, edges, 1, 2,
                                            0, 1, res_sz[0], res_sz[1], res)
                    res = numpy.array(res).T
                    idx -= 1
                    print '2### idx = %d' % (idx)
                elif res_sz[1] == 1:
                    #upConv(pyrBand(pyr,ind,1), hfilt, edges, [2 1], [2 1], res_sz, res);
                    print '1d band'
                    print self.band(idx)
                    res = pyPyrCcode.upConv(self.band(idx).shape[0], 
                                            self.band(idx).shape[1],
                                            self.band(idx), hfilt.shape[1],
                                            hfilt.shape[0], hfilt, edges, 2, 1,
                                            1, 0, res_sz[0], res_sz[1], res)
                    res = numpy.array(res).T
                    idx -= 1
                    print '3### idx = %d' % (idx)
                else:
                    #if 0 in bands:
                    if 0 in bands and lev in levs:
                        print "0 band"
                        print "idx = %d" % idx
                        print "resIdx = %d" % resIdx
                        print "input band"
                        print self.band(idx)
                        #fp = open('tmp.txt', 'a')
                        #fp.write("ires 0 band\n")
                        #fp.close()
                        ires = pyPyrCcode.upConv(self.band(idx).shape[0], 
                                                 self.band(idx).shape[1],
                                                 self.band(idx).T, 
                                                 filt.shape[1], filt.shape[0],
                                                 filt, edges, 1, 2, 0, stag-1,
                                                 hres_sz[0], hres_sz[1])
                        ires = numpy.array(ires).T
                        #ires = ires.reshape(hres_sz[1], hres_sz[0]).T
                        print "ires"
                        print ires
                        print ires.shape

                        print "pre upconv res"
                        print res
                        #fp = open('tmp.txt', 'a')
                        #fp.write("res 0 band\n")
                        #fp.close()
                        print "pre res size"
                        print res.shape
                        res = pyPyrCcode.upConv(ires.shape[0], ires.shape[1],
                                                ires.T, hfilt.shape[1],
                                                hfilt.shape[0], hfilt, edges,
                                                2, 1, 1, 0, res_sz[0],
                                                res_sz[1], res.T)
                        res = numpy.array(res).T
                        print "res size %d %d" % (res_sz[1], res_sz[0])
                        print "post res size"
                        print res.shape
                        print "post upconv res"
                        print res
                    idx += 1
                    #if 1 in bands:
                    if 1 in bands and lev in levs:
                        print "1 band"
                        print "idx = %d" % idx
                        print "lres_sz"
                        print lres_sz
                        print lres_sz[0]
                        print lres_sz[1]
                        print "self.band(idx)"
                        print self.band(idx)
                        print self.band(idx).shape
                        print hfilt
                        #fp = open('tmp.txt', 'a')
                        #fp.write("ires 1 band\n")
                        #fp.close()
                        ires = pyPyrCcode.upConv(self.band(idx).shape[0], 
                                                 self.band(idx).shape[1], 
                                                 self.band(idx).T, 
                                                 hfilt.shape[0], hfilt.shape[1],
                                                 hfilt, edges, 1, 2, 0, 1,
                                                 lres_sz[0], lres_sz[1])
                        ires = numpy.array(ires).T
                        #ires = ires.reshape(lres_sz[1], lres_sz[0]).T
                        print "ires"
                        print ires
                        print ires.shape
                        print filt.shape
                        print "pre res"
                        print res
                        print filt
                        print edges
                        print "stag = %d" % stag
                        #fp = open('tmp.txt', 'a')
                        #fp.write("res 1 band\n")
                        #fp.close()
                        res = pyPyrCcode.upConv(ires.shape[0], ires.shape[1],
                                                ires.T,  filt.shape[0],
                                                filt.shape[1], filt, edges, 2,
                                                1, stag-1, 0, res_sz[0],
                                                res_sz[1], res.T)
                        res = numpy.array(res).T
                        print "res"
                        print res
                    idx += 1
                    if 2 in bands and lev in levs:
                        print "2 band"
                        print "idx = %d" % idx
                        print "input image"
                        print self.band(idx)
                        #fp = open('tmp.txt', 'a')
                        #fp.write("ires 2 band\n")
                        #fp.close()
                        ires = pyPyrCcode.upConv(self.band(idx).shape[0],
                                                 self.band(idx).shape[1],
                                                 self.band(idx).T,
                                                 hfilt.shape[0], hfilt.shape[1],
                                                 hfilt, edges, 1, 2, 0, 1,
                                                 hres_sz[0], hres_sz[1])
                        ires = numpy.array(ires).T
                        #ires = ires.reshape(hres_sz[1], hres_sz[0]).T
                        print "ires"
                        print ires
                        print "pre res"
                        print res
                        #fp = open('tmp.txt', 'a')
                        #fp.write("res 2 band\n")
                        #fp.close()
                        res = pyPyrCcode.upConv(ires.shape[1], ires.shape[0],
                                                ires.T, hfilt.shape[1],
                                                hfilt.shape[0], hfilt, edges,
                                                2, 1, 1, 0, res_sz[0],
                                                res_sz[1], res.T)
                        res = numpy.array(res).T
                        print "res"
                        print res
                    idx += 1
                # need to jump back n bands in the idx each loop
                print 'self.pyrSize[0]'
                print self.pyrSize[0]
                if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or
                     self.pyrSize[0][1] == 1 ):
                    #idx -= 1
                    idx = idx
                else:
                    idx -= 2*len(bands)
                    print '4### idx = %d' % (idx)
        return res
'''
    #def reconWpyr(self, *args):
    def reconPyr(self, *args):
        # Optional args
        if len(args) > 0:
            filt = args[0]
        else:
            filt = 'qmf9'
            
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

        #------------------------------------------------------

        maxLev = int(self.wpyrHt() + 1)

        if levs == 'all':
            levs = numpy.array(range(maxLev))
        else:
            tmpLevs = []
            for l in levs:
                tmpLevs.append((maxLev-1)-l)
            levs = numpy.array(tmpLevs)
            if (levs > maxLev).any():
                print "Error: level numbers must be in the range [0, %d]" % (maxLev)
        allLevs = numpy.array(range(maxLev))

        if bands == "all":
            if ( len(self.band(0)) == 1 or self.band(0).shape[0] == 1 or 
                 self.band(0).shape[1] == 1 ):
                bands = numpy.array([0]);
            else:
                bands = numpy.array(range(3))
        else:
            bands = numpy.array(bands)
            if (bands < 0).any() or (bands > 2).any():
                print "Error: band numbers must be in the range [0,2]."
        
        if isinstance(filt, basestring):
            filt = pyPyrUtils.namedFilter(filt)

        hfilt = pyPyrUtils.modulateFlip(filt)

        # for odd-length filters, stagger the sampling lattices:
        if len(filt) % 2 == 0:
            stag = 2
        else:
            stag = 1

        idx = len(self.pyrSize)-1
        #print 'idx = %d' % (idx)

        #for lev in levs:
        for lev in allLevs:
            #print "starting levs loop lev = %d" % lev

            if lev == 0:
                if 0 in levs:
                    res = self.pyr[len(self.pyr)-1]
                else:
                    res = numpy.zeros(self.pyr[len(self.pyr)-1].shape)
                #print res
                #print 'lev 0 res'
                #print res
            elif lev > 0:
                # compute size of result image: assumes critical sampling
                #if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or
                #     self.pyrSize[0][1] == 1 ):
                #    resIdx = len(self.pyrSize)-lev-2
                #else:
                #    resIdx = len(self.pyrSize)-(3*(lev-1))-3
                #    #res_sz = self.pyrSize[resIdx]
                #print "resIdx = %d" % resIdx
                #res_sz = self.pyrSize[resIdx]
                #print 'res_sz'
                #print res_sz
                #if res_sz[0] == 1:
                #    #res_sz = (res_sz[0], sum([x[1] for x in self.pyrSize]))
                #    if lev == 1:
                #        res_sz = res_sz
                #    else:
                #        res_sz = (1, res_sz[1]*2)
                #elif res_sz[1] == 1:
                #    #res_sz = (res_sz[0], sum([x[0] for x in self.pyrSize]))
                #    if lev == 1:
                #        res_sz = res_sz
                #    else:
                #        res_sz = (res_sz[1]*2, 1)
                #else:
                #    #horizontal + vertical bands
                #    res_sz = (self.pyrSize[resIdx][0]+self.pyrSize[resIdx-1][0],
                #self.pyrSize[resIdx][1]+self.pyrSize[resIdx-1][1])
                #    lres_sz = numpy.array([self.pyrSize[resIdx][0], res_sz[1]])
                #    hres_sz = numpy.array([self.pyrSize[resIdx-1][0], res_sz[1]])
                #print 'post res_sz'
                #print res_sz
                if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or
                     self.pyrSize[0][1] == 1 ):
                    resIdx = len(self.pyrSize)-lev-2
                    #print "resIdx = %d" % resIdx
                    #res_sz = self.pyrSize[resIdx]
                    #print 'lev = %d' % (lev)
                    #print 'allLevs'
                    #print allLevs
                    if self.pyrSize[0][0] == 1:
                        print 'res_sz 1'
                        #res_sz = (res_sz[0], sum([x[1] for x in self.pyrSize]))
                        if lev == allLevs[-1]:
                            #print 'lev flag 1'
                            res_sz = (1, res_sz[1]*2)
                        else:
                            #print 'lev flag 2'
                            res_sz = self.pyrSize[resIdx]
                    elif self.pyrSize[0][1] == 1:
                        print 'res_sz 2'
                        #res_sz = (res_sz[0], sum([x[0] for x in self.pyrSize]))
                        if lev == allLevs[-1]:
                            #print 'lev flag 1'
                            res_sz = (res_sz[0]*2, 1)
                        else:
                            #print 'lev flag 2'
                            res_sz = self.pyrSize[resIdx]
                    print 'res_sz'
                    print res_sz
                else:
                    resIdx = len(self.pyrSize)-(3*(lev-1))-3
                    #print "resIdx = %d" % resIdx
                    #horizontal + vertical bands
                    res_sz = (self.pyrSize[resIdx][0]+self.pyrSize[resIdx-1][0],
                              self.pyrSize[resIdx][1]+self.pyrSize[resIdx-1][1])
                    lres_sz = numpy.array([self.pyrSize[resIdx][0], res_sz[1]])
                    hres_sz = numpy.array([self.pyrSize[resIdx-1][0], res_sz[1]])
                #print 'post res_sz'
                #print res_sz

                imageIn = res.copy()
                if res_sz[0] == 1:
                    #print 'flag 1'
                    ## orig code
                    #res = pyPyrCcode.upConv(imageIn.shape[0], imageIn.shape[1],
                    #                        imageIn, filt.shape[1],
                    #                        filt.shape[0], filt, edges, 1, 2,
                    #                        0, stag-1, res_sz[0], res_sz[1])
                    #res = numpy.array(res).T
                    #print 'res'
                    #print res
                    ## new code
                    res = pyPyrUtils.upConv(image = imageIn, filt = filt.T,
                                            edges = edges, step = (1,2),
                                            start = (0,stag-1), 
                                            stop = res_sz)
                    res = numpy.array(res).T
                elif res_sz[1] == 1:
                    #print 'flag 2'
                    #res = pyPyrCcode.upConv(imageIn.shape[0], imageIn.shape[1],
                    #                        imageIn, filt.shape[1],
                    #                        filt.shape[0], filt, edges, 2, 1,
                    #                        stag-1, 1, res_sz[0], res_sz[1])
                    #res = numpy.array(res).T
                    res = pyPyrUtils.upConv(image = imageIn, filt = filt,
                                            edges = edges, step = (2,1),
                                            start = (stag-1,0), stop = res_sz)
                    res = numpy.array(res).T
                    #print 'res'
                    #print res
                else:
                    #imageIn_test = imageIn
                    #print 'flag 0'
                    ## orig code
                    #ires = pyPyrCcode.upConv(imageIn.shape[1], imageIn.shape[0],
                    #                         imageIn.T, filt.shape[1],
                    #                         filt.shape[0], filt, edges, 1, 2,
                    #                         0, stag-1, lres_sz[0], lres_sz[1])
                    #ires = numpy.array(ires).T
                    #print 'ires'
                    #print ires
                    #res = pyPyrCcode.upConv(ires.shape[1], ires.shape[0],
                    #                        ires.T, filt.shape[0],
                    #                        filt.shape[1], filt, edges, 2, 1,
                    #                        stag-1, 0, res_sz[0], res_sz[1]).T
                    #res = numpy.array(res)
                    #print 'res'
                    #print res
                    ## new code
                    ires = pyPyrUtils.upConv(image = imageIn.T, filt = filt.T,
                                             edges = edges, step = (1,2),
                                             start = (0,stag-1), 
                                             stop = lres_sz).T
                    ires = numpy.array(ires)
                    #print 'ires'
                    #print ires
                    res = pyPyrUtils.upConv(image = ires.T, filt = filt,
                                            edges = edges, step = (2,1), 
                                            start = (stag-1,0), 
                                            stop = res_sz).T
                    res = numpy.array(res)
                    #print 'res'
                    #print res

                if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or
                     self.pyrSize[0][1] == 1 ):
                    idx = resIdx + 1
                else:
                    idx = resIdx - 1

                if res_sz[0] ==1 and lev in levs:
                    #print 'flag 4'
                    #res_test = res
                    ## orig code
                    #res = pyPyrCcode.upConv(self.band(idx).shape[0], 
                    #                        self.band(idx).shape[1],
                    #                        self.band(idx), hfilt.shape[0],
                    #                        hfilt.shape[1], hfilt, edges, 1, 2,
                    #                        0, 1, res_sz[0], res_sz[1], res)
                    #res = numpy.array(res)
                    #print 'res'
                    #print res
                    ## new code
                    res = pyPyrUtils.upConv(image = self.band(idx), 
                                            filt = hfilt, edges = edges,
                                            step = (1,2), start = (0,1),
                                            stop = res_sz, result = res)
                    res = numpy.array(res)
                    #print 'res_test'
                    #print res_test
                    idx -= 1
                elif res_sz[1] == 1:
                    #res_test = res
                    #print 'flag 5'
                    ## orig code
                    #res = pyPyrCcode.upConv(self.band(idx).shape[0], 
                    #                        self.band(idx).shape[1],
                    #                        self.band(idx), hfilt.shape[1],
                    #                        hfilt.shape[0], hfilt, edges, 2, 1,
                    #                        1, 0, res_sz[0], res_sz[1], res)
                    #res = numpy.array(res)
                    #print 'res'
                    #print res
                    ## new code
                    res = pyPyrUtils.upConv(image = self.band(idx), 
                                            filt = hfilt.T, edges = edges,
                                            step = (2,1), start = (1,0),
                                            stop = res_sz, result = res)
                    res_test = numpy.array(res)
                    #print 'res_test'
                    #print res_test
                    idx -= 1
                else:
                    res_test = res
                    if 0 in bands and lev in levs:
                        print 'flag 1'
                        ## orig code
                        #ires = pyPyrCcode.upConv(self.band(idx).shape[0], 
                        #                         self.band(idx).shape[1],
                        #                         self.band(idx).T, 
                        #                         filt.shape[1], filt.shape[0],
                        #                         filt, edges, 1, 2, 0, stag-1,
                        #                         hres_sz[0], hres_sz[1])
                        #ires = numpy.array(ires).T
                        #print 'ires'
                        #print ires
                        #res = pyPyrCcode.upConv(ires.shape[0], ires.shape[1],
                        #                        ires.T, hfilt.shape[1],
                        #                        hfilt.shape[0], hfilt, edges,
                        #                        2, 1, 1, 0, res_sz[0],
                        #                        res_sz[1], res.T)
                        #res = numpy.array(res).T
                        #print 'res'
                        #print res
                        # new code - works for square not rect
                        #ires = pyPyrUtils.upConv(image = self.band(idx),
                        #                         filt = filt, 
                        #                         edges = edges,
                        #                         step = (2,1), 
                        #                         start = (stag-1,0), 
                        #                         stop = (hres_sz[1],
                        #                                 hres_sz[0]))
                        #ires = numpy.array(ires)
                        #print 'ires'
                        #print ires
                        #res = pyPyrUtils.upConv(image = ires, 
                        #                        filt = hfilt, 
                        #                        edges = edges, step = (1,2),
                        #                        start = (0,1), 
                        #                        stop = res_sz,
                        #                        result = res)
                        #res = numpy.array(res)
                        #print 'res'
                        #print res
                        # another try...
                        ires = pyPyrUtils.upConv(image = self.band(idx),
                                                 filt = filt, 
                                                 edges = edges,
                                                 step = (2,1), 
                                                 start = (stag-1,0), 
                                                 stop = (hres_sz[1],
                                                         hres_sz[0]))
                        ires = numpy.array(ires)
                        #print 'ires'
                        #print ires
                        res = pyPyrUtils.upConv(image = ires, 
                                                filt = hfilt, 
                                                edges = edges, step = (1,2),
                                                start = (0,1), 
                                                stop = (res_sz[1],
                                                        res_sz[0]),
                                                result = res)
                        res = numpy.array(res)
                        #print 'res'
                        #print res
                    idx += 1
                    if 1 in bands and lev in levs:
                        #print 'flag 2'
                        ## orig code
                        #ires = pyPyrCcode.upConv(self.band(idx).shape[0], 
                        #                         self.band(idx).shape[1], 
                        #                         self.band(idx).T,
                        #                         hfilt.shape[0], hfilt.shape[1],
                        #                         hfilt, edges, 1, 2, 0, 1,
                        #                         lres_sz[0], lres_sz[1])
                        #ires = numpy.array(ires).T
                        #print 'ires'
                        #print ires
                        #res = pyPyrCcode.upConv(ires.shape[0], ires.shape[1],
                        #                        ires.T, filt.shape[0],
                        #                        filt.shape[1], filt, edges, 2,
                        #                        1, stag-1, 0, res_sz[0],
                        #                        res_sz[1], res.T)
                        #res = numpy.array(res).T
                        #print 'res'
                        #print res
                        ## new code
                        ires = pyPyrUtils.upConv(image = self.band(idx).T,
                                                 filt = hfilt, 
                                                 edges = edges,
                                                 step = (1,2), 
                                                 start = (0,1),
                                                 stop = lres_sz)
                        ires = numpy.array(ires).T
                        #print 'ires'
                        #print ires
                        res = pyPyrUtils.upConv(image = ires.T, 
                                                filt = filt,
                                                edges = edges, step = (2,1),
                                                start = (stag-1,0), 
                                                stop = res_sz, 
                                                result = res.T)
                        res = numpy.array(res).T
                        #print 'res'
                        #print res
                    idx += 1
                    if 2 in bands and lev in levs:
                        #print 'flag 3'
                        ## orig code
                        #ires = pyPyrCcode.upConv(self.band(idx).shape[0],
                        #                         self.band(idx).shape[1],
                        #                         self.band(idx).T,
                        #                         hfilt.shape[0],
                        #                         hfilt.shape[1], hfilt, edges,
                        #                         1, 2, 0, 1, hres_sz[0],
                        #                         hres_sz[1])
                        #ires = numpy.array(ires).T
                        #print 'ires'
                        #print ires
                        #res = pyPyrCcode.upConv(ires.shape[1], ires.shape[0],
                        #                        ires.T, hfilt.shape[1],
                        #                        hfilt.shape[0], hfilt, edges,
                        #                        2, 1, 1, 0, res_sz[0],
                        #                        res_sz[1], res.T)
                        #res = numpy.array(res).T
                        #print 'res'
                        #print res
                        ## new code
                        ires = pyPyrUtils.upConv(image = self.band(idx).T,
                                                 filt = hfilt, 
                                                 edges = edges,
                                                 step = (1,2), 
                                                 start = (0,1),
                                                 stop = (hres_sz[0],
                                                         hres_sz[1]))
                        ires = numpy.array(ires).T
                        #print 'ires'
                        #print ires
                        res = pyPyrUtils.upConv(image = ires.T, 
                                                filt = hfilt.T,
                                                edges = edges, step = (2,1),
                                                start = (1,0), 
                                                stop = (res_sz[0],
                                                        res_sz[1]),
                                                result = res.T)
                        res = numpy.array(res).T
                        #print 'res'
                        #print res
                    idx += 1
                # need to jump back n bands in the idx each loop
                if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or
                     self.pyrSize[0][1] == 1 ):
                    idx = idx
                else:
                    idx -= 2*len(bands)
        return res

    def set_old(self, *args):
        if len(args) != 3:
            print 'Error: three input parameters required:'
            print '  set(band, location, value)'
            print '  where band and value are integer and location is a tuple'
        self.pyr[args[0]][args[1][0]][args[1][1]] = args[2] 

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
            

    def set1D(self, *args):
        if len(args) != 3:
            print 'Error: three input parameters required:'
            print '  set(band, location, value)'
            print '  where band and value are integer and location is a tuple'
        print '%d %d %d' % (args[0], args[1], args[2])
        print self.pyr[args[0]][0][1]
        #self.pyr[args[0]][args[1]] = args[2] 

    def pyrLow(self):
        return numpy.array(self.band(len(self.pyrSize)-1))

    #def showPyr(self, *args):
    def showPyr(self, prange = None, gap = 1, scale = None, disp = 'qt'):
        # determine 1D or 2D pyramid:
        if self.pyrSize[0][0] == 1 or self.pyrSize[0][1] == 1:
            nbands = 1
        else:
            nbands = 3

        #if len(args) > 0:
        #    prange = args[0]
        #else:
        #    if nbands == 1:
        #        prange = 'auto1'
        #    else:
        #        prange = 'auto2'

        #if len(args) > 1:
        #    gap = args[1]
        #else:
        #    gap = 1

        #if len(args) > 2:
        #    scale = args[2]
        #else:
        #    if nbands == 1:
        #        scale = numpy.sqrt(2)
        #    else:
        #        scale = 2

        if prange == None and nbands == 1:
            prange = 'auto1'
        elif prange == None and nbands == 3:
            prange = 'auto2'

        if scale == None and nbands == 1:
            scale = numpy.sqrt(2)
        elif scale == None and nbands == 3:
            scale = 2
        
        ht = int(self.wpyrHt())
        nind = len(self.pyr)

        ## Auto range calculations:
        if prange == 'auto1':
            prange = numpy.ones((nind,1))
            mn = 0.0
            mx = 0.0
            for lnum in range(1,ht+1):
                for bnum in range(nbands):
                    idx = pyPyrUtils.LB2idx(lnum, bnum, ht+2, nbands)
                    band = self.band(idx)/(numpy.power(scale,lnum))
                    prange[(lnum-1)*nbands+bnum+1] = numpy.power(scale,lnum-1)
                    bmn = numpy.amin(band)
                    bmx = numpy.amax(band)
                    mn = min([mn, bmn])
                    mx = max([mx, bmx])
            if nbands == 1:
                pad = (mx-mn)/12
                mn = mn-pad
                mx = mx+pad
            prange = numpy.outer(prange, numpy.array([mn, mx]))
            band = self.pyrLow()
            mn = numpy.amin(band)
            mx = numpy.amax(band)
            if nbands == 1:
                pad = (mx-mn)/12
                mn = mn-pad
                mx = mx+pad
            prange[nind-1,:] = numpy.array([mn, mx])
        elif prange == 'indep1':
            prange = numpy.zeros((nind,2))
            for bnum in range(nind):
                band = self.band(bnum)
                mn = band.min()
                mx = band.max()
                if nbands == 1:
                    pad = (mx-mn)/12
                    mn = mn-pad
                    mx = mx+pad
                prange[bnum,:] = numpy.array([mn, mx])
        elif prange == 'auto2':
            prange = numpy.ones(nind)
            sqsum = 0
            numpixels = 0
            for lnum in range(1,ht+1):
                for bnum in range(nbands):
                    #band = self.band(ppu.LB2idx(lnum, bnum, ht+2, nbands))
                    band = self.band(pyPyrUtils.LB2idx(lnum, bnum, ht, nbands))
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


        if nbands == 1:   # 1D signal
            fig = matplotlib.pyplot.figure()
            ax0 = fig.add_subplot(len(self.pyrSize), 1, 0)
            ax0.set_frame_on(False)
            ax0.get_xaxis().tick_bottom()
            ax0.get_xaxis().tick_top()
            ax0.get_yaxis().tick_right()
            ax0.get_yaxis().tick_left()
            ax0.get_yaxis().set_visible(False)
            for bnum in range(nind):
                band = self.band(bnum)
                pylab.subplot(len(self.pyrSize), 1, bnum+1)
                pylab.plot(band.T)
                #ylim(pRange[bnum,:])
                #xlim((0,self.band(bnum).shape[1]-1))
            matplotlib.pyplot.show()
        else:
            colormap = matplotlib.cm.Greys_r
            bg = 255

            # compute positions of subbands
            llpos = numpy.ones((nind,2));

            #for lnum in range(1,ht+1):
            for lnum in range(ht):
                #ind1 = (lnum-1)*nbands + 1
                #ind1 = lnum*nbands + 1
                ind1 = lnum*nbands
                xpos = self.pyrSize[ind1][1] + 1 + gap*(ht-lnum+1);
                ypos = self.pyrSize[ind1+1][0] + 1 + gap*(ht-lnum+1);
                llpos[ind1:ind1+3, :] = [[ypos, 1], [1, xpos], [ypos, xpos]]
            llpos[nind-1,:] = [1, 1]   # lowpass
    
            # make position list positive, and allocate appropriate image:
            llpos = llpos - ((numpy.ones((nind,1)) * numpy.amin(llpos, axis=0)) + 1) + 1
            #urpos = llpos + self.pyrSize - 1
            urpos = llpos + self.pyrSize
            d_im = numpy.ones((numpy.amax(urpos), numpy.amax(urpos))) * bg
        
            # paste bands into image, (im-r1)*(nshades-1)/(r2-r1) + 1.5
            nshades = 64;
            #print llpos
            #print urpos
            #for bnum in range(1,nind):
            for bnum in range(nind):
                mult = (nshades-1) / (prange[bnum,1]-prange[bnum,0])
                d_im[llpos[bnum,0]:urpos[bnum,0], 
                     llpos[bnum,1]:urpos[bnum,1]] = mult * self.band(bnum) + (1.5-mult*prange[bnum,0])
            
            if disp == 'qt':
                pyPyrUtils.showIm(d_im, 'auto', 2)
            elif disp == 'nb':
                JBhelpers.showIm(d_im, 'auto', 2)
