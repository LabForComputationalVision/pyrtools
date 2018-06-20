import numpy
#import pyPyrUtils as ppu
from . import pyPyrUtils
#import pyPyrCcode
import math
import matplotlib.cm
import os
import scipy.misc
import cmath
from . import JBhelpers
import pylab
import copy

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


class Spyr(pyramid):
    filt = ''
    edges = ''
    
    #constructor
    def __init__(self, *args):    # (image height, filter file, edges)
        self.pyrType = 'steerable'
        if len(args) > 0:
            self.image = numpy.array(args[0])
        else:
            print("First argument (image) is required.")
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
                print("Filter files not supported yet")
                return
            else:
                print("filter parameters value %s not supported" % (args[2]))
                return
        else:
            filters = pyPyrUtils.sp1Filters()

        harmonics = filters['harmonics']
        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']
        
        max_ht = pyPyrUtils.maxPyrHt(self.image.shape, lofilt.shape)
        if len(args) > 1:
            if args[1] == 'auto':
                ht = max_ht
            elif args[1] > max_ht:
                print("Error: cannot build pyramid higher than %d levels." % (
                    max_ht))
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

        hi0 = pyPyrUtils.corrDn(image = im, filt = hi0filt, edges = edges);

        self.pyr[pyrCtr] = hi0
        self.pyrSize[pyrCtr] = hi0.shape

        pyrCtr += 1

        lo = pyPyrUtils.corrDn(image = im, filt = lo0filt, edges = edges)
        for i in range(ht):
            lo_sz = lo.shape
            # assume square filters  -- start of buildSpyrLevs
            bfiltsz = int(math.floor(math.sqrt(bfilts.shape[0])))

            for b in range(bfilts.shape[1]):
                filt = bfilts[:,b].reshape(bfiltsz,bfiltsz).T
                band = pyPyrUtils.corrDn(image = lo, filt = filt, edges = edges)
                self.pyr[pyrCtr] = numpy.array(band)
                self.pyrSize[pyrCtr] = (band.shape[0], band.shape[1])
                pyrCtr += 1


            lo = pyPyrUtils.corrDn(image = lo, filt = lofilt, edges = edges,
                                   step = (2,2))

        self.pyr[pyrCtr] = numpy.array(lo)
        self.pyrSize[pyrCtr] = lo.shape

    # methods
    def set(self, *args):
        if len(args) != 3:
            print('Error: three input parameters required:')
            print('  set(band, location, value)')
            print('  where band and value are integer and location is a tuple')
        if isinstance(args[1], int):
            self.pyr[args[0]][0][args[1]] = args[2]
        elif isinstance(args[1], tuple):
            self.pyr[args[0]][args[1][0]][args[1][1]] = args[2] 
        else:
            print('Error: location parameter must be int or tuple!')
            return

    def spyrLev(self, lev):
        if lev < 0 or lev > self.spyrHt()-1:
            print('Error: level parameter must be between 0 and %d!' % (self.spyrHt()-1))
            return
        
        levArray = []
        for n in range(self.numBands()):
            levArray.append(self.spyrBand(lev, n))
        levArray = numpy.array(levArray)

        return levArray

    def spyrBand(self, lev, band):
        if lev < 0 or lev > self.spyrHt()-1:
            print('Error: level parameter must be between 0 and %d!' % (self.spyrHt()-1))
            return
        if band < 0 or band > self.numBands()-1:
            print('Error: band parameter must be between 0 and %d!' % (self.numBands()-1))

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
                filters = pyPyrUtils.sp0Filters()
            elif args[0] == 'sp1Filters':
                filters = pyPyrUtils.sp1Filters()
            elif args[0] == 'sp3Filters':
                filters = pyPyrUtils.sp3Filters()
            elif args[0] == 'sp5Filters':
                filters = pyPyrUtils.sp5Filters()
            elif os.path.isfile(args[0]):
                print("Filter files not supported yet")
                return
            else:
                print("filter %s not supported" % (args[0]))
                return
        else:
            filters = pyPyrUtils.sp1Filters()

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
                print("Error: level numbers must be in the range [0, %d]." % (maxLev-1))
                return
            else:
                levs = numpy.array(levs)
                if len(levs) > 1 and levs[0] < levs[1]:
                    levs = levs[::-1]  # we want smallest first
        if bands == 'all':
            bands = numpy.array(list(range(self.numBands())))
        else:
            bands = numpy.array(bands)
            if (bands < 0).any() or (bands > bfilts.shape[1]).any():
                print("Error: band numbers must be in the range [0, %d]." % (self.numBands()-1))
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
            recon = pyPyrUtils.upConv(image = recon, filt = lofilt, 
                                      edges = edges, step = (2,2),
                                      start = (0,0),
                                      stop = self.pyrSize[resSzIdx])

            bandImageIdx = 1 + (((Nlevs-1)-level) * Nbands)
            for band in range(Nbands-1,-1,-1):
                if bandImageIdx in reconList:
                    filt = bfilts[:,(Nbands-1)-band].reshape(bfiltsz, 
                                                             bfiltsz,
                                                             order='F')

                    recon = pyPyrUtils.upConv(image = self.pyr[bandImageIdx], 
                                              filt = filt, edges = edges,
                                              stop = (self.pyrSize[bandImageIdx][0],
                                                      self.pyrSize[bandImageIdx][1]),
                                              result = recon)
                    bandImageIdx += 1
             

        # apply lo0filt
        sz = recon.shape
        recon = pyPyrUtils.upConv(image = recon, filt = lo0filt, 
                                  edges = edges, stop = sz)

        # apply hi0filt if needed
        if 0 in reconList:
            recon = pyPyrUtils.upConv(image = self.pyr[0], filt = hi0filt,
                                      edges = edges, start = (0,0),
                                      step = (1,1), stop = recon.shape,
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
        elif isinstance(prange, str):
            print("Error:Bad RANGE argument: %s'" % (prange))
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
            nrows = 2
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
            print("First argument (image) is required.")
            return

        #------------------------------------------------
        # defaults:

        max_ht = numpy.floor( numpy.log2( min(self.image.shape) ) ) - 2
        if len(args) > 1:
            if(args[1] > max_ht):
                print("Error: cannot build pyramid higher than %d levels." % (max_ht))
            ht = args[1]
        else:
            ht = max_ht
        ht = int(ht)
            
        if len(args) > 2:
            if args[2] > 15 or args[2] < 0:
                print("Warning: order must be an integer in the range [0,15]. Truncating.")
                order = min( max(args[2],0), 15 )
            else:
                order = args[2]
        else:
            order = 3

        nbands = order+1

        if len(args) > 3:
            if args[3] <= 0:
                print("Warning: twidth must be positive. Setting to 1.")
                twidth = 1
            else:
                twidth = args[3]
        else:
            twidth = 1

        #------------------------------------------------------
        # steering stuff:

        if nbands % 2 == 0:
            harmonics = numpy.array(list(range(nbands/2))) * 2 + 1
        else:
            harmonics = numpy.array(list(range((nbands-1)/2))) * 2

        steermtx = pyPyrUtils.steer2HarmMtx(harmonics, 
                                            numpy.pi*numpy.array(list(range(nbands)))/nbands,
                                            'even')
        #------------------------------------------------------
        
        dims = numpy.array(self.image.shape)
        ctr = numpy.ceil((numpy.array(dims)+0.5)/2)
        
        (xramp, yramp) = numpy.meshgrid((numpy.array(list(range(1,dims[1]+1)))-ctr[1])/
                                     (dims[1]/2), 
                                     (numpy.array(list(range(1,dims[0]+1)))-ctr[0])/
                                     (dims[0]/2))
        angle = numpy.arctan2(yramp, xramp)
        log_rad = numpy.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        log_rad = numpy.log2(log_rad);

        ## Radial transition function (a raised cosine in log-frequency):
        (Xrcos, Yrcos) = pyPyrUtils.rcosFn(twidth, (-twidth/2.0), numpy.array([0,1]))
        Yrcos = numpy.sqrt(Yrcos)

        YIrcos = numpy.sqrt(1.0 - Yrcos**2)
        lo0mask = pyPyrUtils.pointOp(log_rad, YIrcos, Xrcos[0], 
                                     Xrcos[1]-Xrcos[0], 0)
        numpy.array(lo0mask)

        imdft = numpy.fft.fftshift(numpy.fft.fft2(self.image))

        self.pyr = []
        self.pyrSize = []

        hi0mask = pyPyrUtils.pointOp(log_rad, Yrcos, Xrcos[0], 
                                     Xrcos[1]-Xrcos[0], 0)
        hi0mask = numpy.array(hi0mask)

        hi0dft = imdft * hi0mask.reshape(imdft.shape[0], imdft.shape[1])
        hi0 = numpy.fft.ifft2(numpy.fft.ifftshift(hi0dft))

        self.pyr.append(numpy.real(hi0))
        self.pyrSize.append(hi0.shape)

        lo0mask = lo0mask.reshape(imdft.shape[0], imdft.shape[1])
        lodft = imdft * lo0mask

        for i in range(ht):
            bands = numpy.zeros((lodft.shape[0]*lodft.shape[1], nbands))
            bind = numpy.zeros((nbands, 2))
        
            Xrcos -= numpy.log2(2)

            lutsize = 1024
            Xcosn = numpy.pi * numpy.array(list(range(-(2*lutsize+1), (lutsize+2)))) / lutsize

            order = nbands -1
            const = (2**(2*order))*(scipy.misc.factorial(order, exact=True)**2)/float(nbands*scipy.misc.factorial(2*order, exact=True))
            Ycosn = numpy.sqrt(const) * (numpy.cos(Xcosn))**order
            log_rad_test = numpy.reshape(log_rad,(1,
                                                  log_rad.shape[0]*
                                                  log_rad.shape[1]))
            himask = pyPyrUtils.pointOp(log_rad_test, Yrcos, Xrcos[0], 
                                        Xrcos[1]-Xrcos[0], 0)
            himask = numpy.array(himask)
            himask = numpy.reshape(himask, 
                                   (lodft.shape[0], lodft.shape[1]))

            for b in range(nbands):
                angle_tmp = numpy.reshape(angle, 
                                          (1,angle.shape[0]*angle.shape[1]))
                anglemask = pyPyrUtils.pointOp(angle_tmp, Ycosn, 
                                               Xcosn[0]+numpy.pi*b/nbands,
                                               Xcosn[1]-Xcosn[0],0)
                anglemask = numpy.array(anglemask)

                anglemask = anglemask.reshape(lodft.shape[0], lodft.shape[1])
                banddft = ((-numpy.power(-1+0j,0.5))**order) * lodft * anglemask * himask
                band = numpy.fft.ifft2(numpy.fft.ifftshift(banddft))
                self.pyr.append(numpy.real(band.copy()))
                self.pyrSize.append(band.shape)

            dims = numpy.array(lodft.shape)
            ctr = numpy.ceil((dims+0.5)/2)
            lodims = numpy.ceil((dims-0.5)/2)
            loctr = numpy.ceil((lodims+0.5)/2)
            lostart = ctr - loctr
            loend = lostart + lodims

            log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = numpy.abs(numpy.sqrt(1.0 - Yrcos**2))
            log_rad_tmp = numpy.reshape(log_rad, 
                                        (1,log_rad.shape[0]*log_rad.shape[1]))
            lomask = pyPyrUtils.pointOp(log_rad_tmp, YIrcos, Xrcos[0], 
                                        Xrcos[1]-Xrcos[0], 0)
            lomask = numpy.array(lomask)
            
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

    def reconSFpyr(self, *args):
        
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
                print("Warning: twidth must be positive. Setting to 1.")
                twidth = 1
            else:
                twidth = args[2]
        else:
            twidth = 1

        #-----------------------------------------------------------------

        nbands = self.numBands()
        
        maxLev = 1 + self.spyrHt()
        if isinstance(levs, str) and levs == 'all':
            levs = numpy.array(list(range(maxLev+1)))
        elif isinstance(levs, str):
            print("Error: %s not valid for levs parameter." % (levs))
            print("levs must be either a 1D numpy array or the string 'all'.")
            return
        else:
            levs = numpy.array(levs)

        if isinstance(bands, str) and bands == 'all':
            bands = numpy.array(list(range(nbands)))
        elif isinstance(bands, str):
            print("Error: %s not valid for bands parameter." % (bands))
            print("bands must be either a 1D numpy array or the string 'all'.")
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

        (xramp, yramp) = numpy.meshgrid((numpy.array(list(range(1,dims[1]+1)))-ctr[1])/
                                     (dims[1]/2), 
                                     (numpy.array(list(range(1,dims[0]+1)))-ctr[0])/
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
        Xcosn = numpy.pi * numpy.array(list(range(-(2*lutsize+1), (lutsize+2)))) / lutsize
        
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

        nlog_rad_tmp = numpy.reshape(nlog_rad, 
                                     (1,nlog_rad.shape[0]*nlog_rad.shape[1]))
        lomask = pyPyrUtils.pointOp(nlog_rad_tmp, YIrcos, Xrcos[0], 
                                    Xrcos[1]-Xrcos[0], 0)
        lomask = numpy.array(lomask)
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
                nlog_rad2_tmp = numpy.reshape(nlog_rad2, 
                                              (1,nlog_rad2.shape[0]*
                                               nlog_rad2.shape[1]))
                lomask = pyPyrUtils.pointOp(nlog_rad2_tmp, YIrcos,
                                            Xrcos[0], Xrcos[1]-Xrcos[0], 0)
                lomask = numpy.array(lomask)

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
                        nlog_rad1_tmp = numpy.reshape(nlog_rad1, 
                                                      (1,nlog_rad1.shape[0]*
                                                       nlog_rad1.shape[1]))
                        himask = pyPyrUtils.pointOp(nlog_rad1_tmp, Yrcos, 
                                                    Xrcos[0], Xrcos[1]-Xrcos[0],
                                                    0)

                        himask = himask.reshape(nlog_rad1.shape)
                        nangle_tmp = numpy.reshape(nangle, (1,
                                                            nangle.shape[0]*
                                                            nangle.shape[1]))
                        anglemask = pyPyrUtils.pointOp(nangle_tmp, Ycosn, 
                                                       Xcosn[0]+numpy.pi*
                                                       b/nbands,
                                                       Xcosn[1]-Xcosn[0], 0)
                        anglemask = numpy.array(anglemask)

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
        lo0mask = pyPyrUtils.pointOp(log_rad, YIrcos, Xrcos[0], 
                                     Xrcos[1]-Xrcos[0], 0)
        lo0mask = numpy.array(lo0mask)

        lo0mask = lo0mask.reshape(dims[0], dims[1])
        resdft = resdft * lo0mask
        
        # residual highpass subband
        hi0mask = pyPyrUtils.pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0],
                                     0)
        hi0mask = numpy.array(hi0mask)

        hi0mask = hi0mask.reshape(resdft.shape[0], resdft.shape[1])
        if 0 in levs:
            hidft = numpy.fft.fftshift(numpy.fft.fft2(self.pyr[0]))
        else:
            hidft = numpy.zeros(self.pyr[0].shape)
        resdft += hidft * hi0mask

        outresdft = numpy.real(numpy.fft.ifft2(numpy.fft.ifftshift(resdft)))

        return outresdft

    reconPyr = reconSFpyr


class SCFpyr(SFpyr):
    filt = ''
    edges = ''
    
    #constructor
    def __init__(self, *args):    # (image, height, order, twidth)
        self.pyrType = 'steerableFrequency'

        if len(args) > 0:
            self.image = args[0]
        else:
            print("First argument (image) is required.")
            return

        #------------------------------------------------
        # defaults:

        max_ht = numpy.floor( numpy.log2( min(self.image.shape) ) ) - 2
        if len(args) > 1:
            if(args[1] > max_ht):
                print("Error: cannot build pyramid higher than %d levels." % (max_ht))
            ht = args[1]
        else:
            ht = max_ht
        ht = int(ht)
            
        if len(args) > 2:
            if args[2] > 15 or args[2] < 0:
                print("Warning: order must be an integer in the range [0,15]. Truncating.")
                order = min( max(args[2],0), 15 )
            else:
                order = args[2]
        else:
            order = 3

        nbands = order+1

        if len(args) > 3:
            if args[3] <= 0:
                print("Warning: twidth must be positive. Setting to 1.")
                twidth = 1
            else:
                twidth = args[3]
        else:
            twidth = 1

        #------------------------------------------------------
        # steering stuff:

        if nbands % 2 == 0:
            harmonics = numpy.array(list(range(nbands/2))) * 2 + 1
        else:
            harmonics = numpy.array(list(range((nbands-1)/2))) * 2

        steermtx = pyPyrUtils.steer2HarmMtx(harmonics, 
                                     numpy.pi*numpy.array(list(range(nbands)))/nbands,
                                     'even')
        #------------------------------------------------------
        
        dims = numpy.array(self.image.shape)
        ctr = numpy.ceil((numpy.array(dims)+0.5)/2)
        
        (xramp, yramp) = numpy.meshgrid((numpy.array(list(range(1,dims[1]+1)))-ctr[1])/
                                     (dims[1]/2), 
                                     (numpy.array(list(range(1,dims[0]+1)))-ctr[0])/
                                     (dims[0]/2))
        angle = numpy.arctan2(yramp, xramp)
        log_rad = numpy.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        log_rad = numpy.log2(log_rad);

        ## Radial transition function (a raised cosine in log-frequency):
        (Xrcos, Yrcos) = pyPyrUtils.rcosFn(twidth, (-twidth/2.0), numpy.array([0,1]))
        Yrcos = numpy.sqrt(Yrcos)

        YIrcos = numpy.sqrt(1.0 - Yrcos**2)
        lo0mask = pyPyrUtils.pointOp(log_rad, YIrcos, Xrcos[0],
                                     Xrcos[1]-Xrcos[0], 0)
        lo0mask = numpy.array(lo0mask)

        imdft = numpy.fft.fftshift(numpy.fft.fft2(self.image))

        self.pyr = []
        self.pyrSize = []

        hi0mask = pyPyrUtils.pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0],
                                     0)
        hi0mask = numpy.array(hi0mask)

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
            Xcosn = numpy.pi * numpy.array(list(range(-(2*lutsize+1), (lutsize+2)))) / lutsize

            order = nbands -1
            const = (2**(2*order))*(scipy.misc.factorial(order, exact=True)**2)/float(nbands*scipy.misc.factorial(2*order, exact=True))

            alfa = ( (numpy.pi+Xcosn) % (2.0*numpy.pi) ) - numpy.pi
            Ycosn = ( 2.0*numpy.sqrt(const) * (numpy.cos(Xcosn)**order) * 
                      (numpy.abs(alfa)<numpy.pi/2.0).astype(int) )
            log_rad_tmp = numpy.reshape(log_rad, (1,log_rad.shape[0]*
                                                  log_rad.shape[1]))
            himask = pyPyrUtils.pointOp(log_rad_tmp, Yrcos, Xrcos[0], 
                                        Xrcos[1]-Xrcos[0], 0)
            himask = numpy.array(himask)
            
            himask = himask.reshape(lodft.shape[0], lodft.shape[1])
            for b in range(nbands):
                angle_tmp = numpy.reshape(angle, 
                                          (1,angle.shape[0]*angle.shape[1]))
                anglemask = pyPyrUtils.pointOp(angle_tmp, Ycosn, 
                                               Xcosn[0]+numpy.pi*b/nbands, 
                                               Xcosn[1]-Xcosn[0], 0)
                anglemask = numpy.array(anglemask)
                anglemask = anglemask.reshape(lodft.shape[0], lodft.shape[1])
                banddft = (cmath.sqrt(-1)**order) * lodft * anglemask * himask
                band = numpy.negative(numpy.fft.ifft2(numpy.fft.ifftshift(banddft)))
                self.pyr.append(band.copy())
                self.pyrSize.append(band.shape)

            dims = numpy.array(lodft.shape)
            ctr = numpy.ceil((dims+0.5)/2)
            lodims = numpy.ceil((dims-0.5)/2)
            loctr = numpy.ceil((lodims+0.5)/2)
            lostart = ctr - loctr
            loend = lostart + lodims

            log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = numpy.abs(numpy.sqrt(1.0 - Yrcos**2))
            log_rad_tmp = numpy.reshape(log_rad, 
                                        (1,log_rad.shape[0]*log_rad.shape[1]))
            lomask = pyPyrUtils.pointOp(log_rad_tmp, YIrcos, Xrcos[0], 
                                        Xrcos[1]-Xrcos[0], 0)
            lomask = numpy.array(lomask)
            lodft = lodft * lomask.reshape(lodft.shape[0], lodft.shape[1])

        lodft = numpy.fft.ifft2(numpy.fft.ifftshift(lodft))
        self.pyr.append(numpy.real(numpy.array(lodft).copy()))
        self.pyrSize.append(lodft.shape)

    # methods
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
                print("Warning: twidth must be positive. Setting to 1.")
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
            dims = pind[firstBnum][:]
            ctr = (numpy.ceil((dims[0]+0.5)/2.0), numpy.ceil((dims[1]+0.5)/2.0)) #-1?
            ang = pyPyrUtils.mkAngle(dims, 0, ctr)
            ang[ctr[0]-1, ctr[1]-1] = -numpy.pi/2.0
            for nor in range(Nor):
                nband = nsc * Nor + nor + 1
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
            print("pyr = Lpyr(image, height, filter1, filter2, edges)")
            print("First argument (image) is required")
            return

        if len(args) > 2:
            filt1 = args[2]
            if isinstance(filt1, str):
                filt1 = pyPyrUtils.namedFilter(filt1)
            elif len(filt1.shape) != 1 and ( filt1.shape[0] != 1 and
                                             filt1.shape[1] != 1 ):
                print("Error: filter1 should be a 1D filter (i.e., a vector)")
                return
        else:
            filt1 = pyPyrUtils.namedFilter('binom5')
        if len(filt1.shape) == 1:
            filt1 = filt1.reshape(1,len(filt1))
        elif self.image.shape[0] == 1:
            filt1 = filt1.reshape(filt1.shape[1], filt1.shape[0])

        if len(args) > 3:
            filt2 = args[3]
            if isinstance(filt2, str):
                filt2 = pyPyrUtils.namedFilter(filt2)
            elif len(filt2.shape) != 1 and ( filt2.shape[0] != 1 and
                                            filt2.shape[1] != 1 ):
                print("Error: filter2 should be a 1D filter (i.e., a vector)")
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
                    print(( "Error: cannot build pyramid higher than %d levels"
                            % (maxHeight) ))
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
        for ht in range(self.height-1,0,-1):
            im_sz = im.shape
            filt1_sz = filt1.shape
            if im_sz[0] == 1:
                lo2 = pyPyrUtils.corrDn(image = im, filt = filt1, edges = edges,
                                        step = (1,2))
                lo2 = numpy.array(lo2)
            elif len(im_sz) == 1 or im_sz[1] == 1:
                lo2  = pyPyrUtils.corrDn(image = im, filt = filt1, edges = edges,
                                         step = (2,1))
                lo2 = numpy.array(lo2)
            else:
                lo = pyPyrUtils.corrDn(image = im, filt = filt1.T, edges = edges,
                                       step = (1,2), start = (0,0))
                lo = numpy.array(lo)
                lo2 = pyPyrUtils.corrDn(image = lo, filt = filt1, edges = edges,
                                        step = (2,1), start = (0,0))
                lo2 = numpy.array(lo2)

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
            filt2_sz = filt2.shape
            if len(im_sz) == 1 or im_sz[1] == 1:
                hi2 = pyPyrUtils.upConv(image = im, filt = filt2.T, 
                                        edges = edges, step = (1,2),
                                        stop = (los[ht].shape[1],
                                                los[ht].shape[0])).T
            elif im_sz[0] == 1:
                hi2 = pyPyrUtils.upConv(image = im, filt = filt2.T, 
                                        edges = edges, step = (2,1), 
                                        stop = (los[ht].shape[1],
                                                los[ht].shape[0])).T
            else:
                hi = pyPyrUtils.upConv(image = im, filt = filt2, 
                                       edges = edges, step = (2,1), 
                                       stop = (los[ht].shape[0], im_sz[1]))
                hi2 = pyPyrUtils.upConv(image = hi, filt = filt2.T, 
                                        edges = edges, step = (1,2), 
                                        stop = (los[ht].shape[0], 
                                                los[ht].shape[1]))
                                       
            hi2 = los[ht] - hi2
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
            print('Error: three input parameters required:')
            print('  set(band, element, value)')
        print('band=%d  element=%d  value=%d' % (args[0],args[1],args[2]))
        print(self.pyr[args[0]].shape)
        self.pyr[args[0]][args[1]] = args[2] 

    def set(self, *args):
        if len(args) != 3:
            print('Error: three input parameters required:')
            print('  set(band, element(tuple), value)')
        self.pyr[args[0]][args[1][0]][args[1][1]] = args[2] 

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
            levs = list(range(0,maxLev))
        else:
            if (levs > maxLev-1).any():
                print(( "Error: level numbers must be in the range [0, %d]." % 
                        (maxLev-1) ))
                return

        if isinstance(filt2, str):
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
                    hi2 = pyPyrUtils.upConv(image = res, filt = filt2,
                                            edges = edges, step = (2,1), 
                                            stop = (new_sz[1], new_sz[0])).T
                elif res_sz[1] == 1:
                    hi2 = pyPyrUtils.upConv(image = res, filt = filt2.T,
                                            edges = edges, step = (1,2), 
                                            stop = (new_sz[1], new_sz[0])).T
                else:
                    hi = pyPyrUtils.upConv(image = res, filt = filt2, 
                                           edges = edges, step = (2,1), 
                                           stop = (new_sz[0], res_sz[1]))
                    hi2 = pyPyrUtils.upConv(image = hi, filt = filt2.T, 
                                            edges = edges, step = (1,2),
                                            stop = (new_sz[0], new_sz[1]))
                if lev in levs:
                    bandIm = self.band(lev)
                    bandIm_sz = bandIm.shape
                    res = hi2 + bandIm
                else:
                    res = hi2
        return res                           
                
    def pyrLow(self):
        return numpy.array(self.band(self.height-1))

    # options for disp are 'qt' and 'nb'
    def showPyr(self, pRange = None, gap = 1, scale = None, disp = 'qt'):
        if ( len(self.band(0).shape) == 1 or self.band(0).shape[0] == 1 or
             self.band(0).shape[1] == 1 ):
            oned = 1
        else:
            oned = 0

        if pRange == None and oned == 1:
            pRange = 'auto1'
        elif pRange == None and oned == 0:
            pRange = 'auto2'

        if scale == None and oned == 1:
            scale = math.sqrt(2)
        elif scale == None and oned == 0:
            scale = 2

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
            pRange[nind-1,:] = numpy.array([av-2*stdev, av+2*stdev]);
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
        elif isinstance(pRange, str):
            print("Error: band range argument: %s" % (pRange))
            return
        elif pRange.shape[0] == 1 and pRange.shape[1] == 2:
            scales = numpy.power( numpy.array( list(range(0,nind)) ), scale)
            pRange = numpy.outer( scales, pRange )
            band = self.pyrLow()
            pRange[nind,:] = ( pRange[nind,:] + numpy.mean(band) - 
                               numpy.mean(pRange[nind,:]) )

        # draw
        if oned == 1:
            #fig = matplotlib.pyplot.figure()
            pyplot.figure()
            #pyplot.subplot()...
            #ax0 = fig.add_subplot(nind, 1, 0)
            #ax0.set_frame_on(False)
            #ax0.get_xaxis().tick_bottom()
            #ax0.get_xaxis().tick_top()
            #ax0.get_yaxis().tick_right()
            #ax0.get_yaxis().tick_left()
            #ax0.get_yaxis().set_visible(False)
            #for bnum in range(0,nind):
            #    pylab.subplot(nind, 1, bnum+1)
            #    pylab.plot(numpy.array(range(numpy.amax(self.band(bnum).shape))).T, 
            #               self.band(bnum).T)
            #    ylim(pRange[bnum,:])
            #    xlim((0,self.band(bnum).shape[1]-1))
            #matplotlib.pyplot.show()
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
            pind = list(range(self.height))
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
            print("pyr = Gpyr(image, height, filter, edges)")
            print("First argument (image) is required")
            return
        else:
            self.image = args[0]

        if len(args) > 2:
            filt = args[2]
            if not (filt.shape == 1).any():
                print("Error: filt should be a 1D filter (i.e., a vector)")
                return
        else:
            print("no filter set, so filter is binom5")
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
                    print(( "Error: cannot build pyramid higher than %d levels"
                            % (maxHeight) ))
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
                lo2 = pyPyrUtils.corrDn(image = im, filt = filt, step = (1,2))
                lo2 = numpy.array(lo2)
            elif len(im_sz) == 1 or im_sz[1] == 1:
                lo2 = pyPyrUtils.corrDn(image = im, filt = filt, step = (2,1))
                lo2 = numpy.array(lo2)
            else:
                lo = pyPyrUtils.corrDn(image = im, filt = filt.T, 
                                       step = (1,2), start = (0,0))
                lo = numpy.array(lo)
                lo2 = pyPyrUtils.corrDn(image = lo, filt = filt, 
                                        step = (2,1), start = (0,0))
                lo2 = numpy.array(lo2)                

            self.pyr.append(lo2.copy())
            self.pyrSize.append(lo2.shape)
            pyrCtr += 1

            im = lo2
        
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
            print("First argument (image) is required.")
            return

        #------------------------------------------------
        # defaults:

        if len(args) > 2:
            filt = args[2]
        else:
            filt = "qmf9"
        if isinstance(filt, str):
            filt = pyPyrUtils.namedFilter(filt)

        if len(filt.shape) != 1 and filt.shape[0] != 1 and filt.shape[1] != 1:
            print("Error: filter should be 1D (i.e., a vector)");
            return
        hfilt = pyPyrUtils.modulateFlip(filt)
        #hfilt = pyPyrUtils.modulateFlip(filt).T

        if len(args) > 3:
            edges = args[3]
        else:
            edges = "reflect1"

        # Stagger sampling if filter is odd-length:
        if filt.shape[0] % 2 == 0:
            stag = 2
        else:
            stag = 1

        # if 1D filter, match to image dimensions
        if len(filt.shape) == 1 or filt.shape[1] == 1:
            if im.shape[0] == 1:
                filt = filt.reshape(1, filt.shape[0])
            elif im.shape[1] == 1:
                filt = filt.reshape(filt.shape[0], 1)

        max_ht = pyPyrUtils.maxPyrHt(im.shape, filt.shape)

        if len(args) > 1:
            ht = args[1]
            if ht == 'auto':
                ht = max_ht
            elif(ht > max_ht):
                print("Error: cannot build pyramid higher than %d levels." % (max_ht))
        else:
            ht = max_ht
        ht = int(ht)
        self.height = ht + 1  # used with showPyr() method
        for lev in range(ht):
            if len(im.shape) == 1 or im.shape[1] == 1:
                lolo = pyPyrUtils.corrDn(image = im, filt = filt, 
                                         edges = edges, step = (2,1), 
                                         start = (stag-1,0))
                lolo = numpy.array(lolo)
                hihi = pyPyrUtils.corrDn(image = im, filt = hfilt, 
                                         edges = edges, step = (2,1), 
                                         start = (1, 0))
                hihi = numpy.array(hihi)
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
                lo = pyPyrUtils.corrDn(image = im, filt = filt, 
                                       edges = edges, step = (2,1), 
                                       start = (stag-1,0))
                lo = numpy.array(lo)
                hi = pyPyrUtils.corrDn(image = im, filt = hfilt, 
                                       edges = edges, step = (2,1), 
                                       start = (1,0))
                hi = numpy.array(hi)
                lolo = pyPyrUtils.corrDn(image = lo, filt = filt.T, 
                                         edges = edges, step = (1,2), 
                                         start = (0, stag-1))
                lolo = numpy.array(lolo)
                lohi = pyPyrUtils.corrDn(image = hi, filt = filt.T, 
                                         edges = edges, step = (1,2),
                                         start = (0,stag-1))
                lohi = numpy.array(lohi)
                hilo = pyPyrUtils.corrDn(image = lo, filt = hfilt.T, 
                                         edges = edges, step = (1,2), 
                                         start = (0,1))
                hilo = numpy.array(hilo)
                hihi = pyPyrUtils.corrDn(image = hi, filt = hfilt.T, 
                                         edges = edges, step = (1,2), 
                                         start = (0,1))
                hihi = numpy.array(hihi)

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
            levs = numpy.array(list(range(maxLev)))
        else:
            tmpLevs = []
            for l in levs:
                tmpLevs.append((maxLev-1)-l)
            levs = numpy.array(tmpLevs)
            if (levs > maxLev).any():
                print("Error: level numbers must be in the range [0, %d]" % (maxLev))
        allLevs = numpy.array(list(range(maxLev)))

        if bands == "all":
            if ( len(self.band(0)) == 1 or self.band(0).shape[0] == 1 or 
                 self.band(0).shape[1] == 1 ):
                bands = numpy.array([0]);
            else:
                bands = numpy.array(list(range(3)))
        else:
            bands = numpy.array(bands)
            if (bands < 0).any() or (bands > 2).any():
                print("Error: band numbers must be in the range [0,2].")
        
        if isinstance(filt, str):
            filt = pyPyrUtils.namedFilter(filt)

        hfilt = pyPyrUtils.modulateFlip(filt).T
        #hfilt = pyPyrUtils.modulateFlip(filt)

        # for odd-length filters, stagger the sampling lattices:
        if len(filt) % 2 == 0:
            stag = 2
        else:
            stag = 1

        idx = len(self.pyrSize)-1

        for lev in allLevs:

            if lev == 0:
                if 0 in levs:
                    res = self.pyr[len(self.pyr)-1]
                else:
                    res = numpy.zeros(self.pyr[len(self.pyr)-1].shape)
            elif lev > 0:
                # compute size of result image: assumes critical sampling
                if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or
                     self.pyrSize[0][1] == 1 ):
                    resIdx = len(self.pyrSize)-lev-2
                    if self.pyrSize[0][0] == 1:
                        if lev == allLevs[-1]:
                            res_sz = (1, res_sz[1]*2)
                        else:
                            res_sz = self.pyrSize[resIdx]
                    elif self.pyrSize[0][1] == 1:
                        if lev == allLevs[-1]:
                            res_sz = (res_sz[0]*2, 1)
                        else:
                            res_sz = self.pyrSize[resIdx]
                else:
                    resIdx = len(self.pyrSize)-(3*(lev-1))-3
                    res_sz = (self.pyrSize[resIdx][0]+self.pyrSize[resIdx-1][0],
                              self.pyrSize[resIdx][1]+self.pyrSize[resIdx-1][1])
                    lres_sz = numpy.array([self.pyrSize[resIdx][0], res_sz[1]])
                    hres_sz = numpy.array([self.pyrSize[resIdx-1][0], res_sz[1]])
                imageIn = res.copy()
                if res_sz[0] == 1:
                    res = pyPyrUtils.upConv(image = imageIn, filt = filt.T,
                                            edges = edges, step = (1,2),
                                            start = (0,stag-1), 
                                            stop = res_sz).T
                    res = numpy.array(res)
                elif res_sz[1] == 1:
                    res = pyPyrUtils.upConv(image = imageIn, filt = filt,
                                            edges = edges, step = (2,1),
                                            start = (stag-1,0), stop = res_sz)
                    res = numpy.array(res).T
                else:
                    ires = pyPyrUtils.upConv(image = imageIn, filt = filt.T,
                                             edges = edges, step = (1,2),
                                             start = (0,stag-1), 
                                             stop = lres_sz)
                    ires = numpy.array(ires)
                    res = pyPyrUtils.upConv(image = ires, filt = filt,
                                            edges = edges, step = (2,1), 
                                            start = (stag-1,0), 
                                            stop = res_sz)
                    res = numpy.array(res)

                if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or
                     self.pyrSize[0][1] == 1 ):
                    idx = resIdx + 1
                else:
                    idx = resIdx - 1

                if res_sz[0] ==1 and lev in levs:
                    res = pyPyrUtils.upConv(image = self.band(idx), 
                                            filt = hfilt, edges = edges,
                                            step = (1,2), start = (0,1),
                                            stop = res_sz, result = res)
                    res = numpy.array(res)
                    idx -= 1
                elif res_sz[1] == 1 and lev in levs:
                    res = pyPyrUtils.upConv(image = self.band(idx), 
                                            filt = hfilt.T, edges = edges,
                                            step = (2,1), start = (1,0),
                                            stop = res_sz, result = res)
                    res = numpy.array(res)
                    idx -= 1
                elif res_sz[0] != 1 and res_sz[1] != 1 and lev in levs:
                    res_test = res
                    if 0 in bands and lev in levs:
                        ires = pyPyrUtils.upConv(image = self.band(idx),
                                                 filt = filt.T, 
                                                 edges = edges,
                                                 step = (1,2), 
                                                 start = (0, stag-1), 
                                                 stop = hres_sz)
                        ires = numpy.array(ires)
                        res = pyPyrUtils.upConv(image = ires, 
                                                filt = hfilt.T, 
                                                edges = edges,
                                                step = (2,1),
                                                start = (1,0), 
                                                stop = (res_sz[0],
                                                        res_sz[1]),
                                                result = res)
                        res = numpy.array(res)
                    idx += 1
                    if 1 in bands and lev in levs:
                        ires = pyPyrUtils.upConv(image = self.band(idx),
                                                 filt = hfilt, 
                                                 edges = edges,
                                                 step = (1,2), 
                                                 start = (0,1),
                                                 stop = lres_sz)
                        ires = numpy.array(ires)
                        res = pyPyrUtils.upConv(image = ires, 
                                                filt = filt,
                                                edges = edges, 
                                                step = (2,1),
                                                start = (stag-1,0), 
                                                stop = (res_sz[0],res_sz[1]), 
                                                result = res)
                        res = numpy.array(res)
                    idx += 1
                    if 2 in bands and lev in levs:
                        ires = pyPyrUtils.upConv(image = self.band(idx),
                                                 filt = hfilt, 
                                                 edges = edges,
                                                 step = (1,2), 
                                                 start = (0,1),
                                                 stop = (hres_sz[0],
                                                         hres_sz[1]))
                        ires = numpy.array(ires)
                        res = pyPyrUtils.upConv(image = ires, 
                                                filt = hfilt.T,
                                                edges = edges, 
                                                step = (2,1),
                                                start = (1,0), 
                                                stop = (res_sz[0],
                                                        res_sz[1]),
                                                result = res)
                        res = numpy.array(res)
                    idx += 1
                # need to jump back n bands in the idx each loop
                if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or
                     self.pyrSize[0][1] == 1 ):
                    idx = idx
                else:
                    idx -= 2*len(bands)
        return res

    #def set_old(self, *args):
    #    if len(args) != 3:
    #        print 'Error: three input parameters required:'
    #        print '  set(band, location, value)'
    #        print '  where band and value are integer and location is a tuple'
    #    self.pyr[args[0]][args[1][0]][args[1][1]] = args[2] 

    def set(self, *args):
        if len(args) != 3:
            print('Error: three input parameters required:')
            print('  set(band, location, value)')
            print('  where band and value are integer and location is a tuple')
        if isinstance(args[1], int):
            self.pyr[args[0]][0][args[1]] = args[2]
        elif isinstance(args[1], tuple):
            self.pyr[args[0]][args[1][0]][args[1][1]] = args[2] 
        else:
            print('Error: location parameter must be int or tuple!')
            return
            

    def set1D(self, *args):
        if len(args) != 3:
            print('Error: three input parameters required:')
            print('  set(band, location, value)')
            print('  where band and value are integer and location is a tuple')
        print('%d %d %d' % (args[0], args[1], args[2]))
        print(self.pyr[args[0]][0][1])

    def pyrLow(self):
        return numpy.array(self.band(len(self.pyrSize)-1))

    def showPyr(self, prange = None, gap = 1, scale = None, disp = 'qt'):
        # determine 1D or 2D pyramid:
        if self.pyrSize[0][0] == 1 or self.pyrSize[0][1] == 1:
            nbands = 1
        else:
            nbands = 3

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
        elif isinstance(prange, str):
            print("Error:Bad RANGE argument: %s'" % (prange))
        elif prange.shape[0] == 1 and prange.shape[1] == 2:
            scales = numpy.power(scale, list(range(ht)))
            scales = numpy.outer( numpy.ones((nbands,1)), scales )
            scales = numpy.array([1, scales, numpy.power(scale, ht)])
            prange = numpy.outer(scales, prange)
            band = self.pyrLow()
            prange[nind,:] += numpy.mean(band) - numpy.mean(prange[nind,:])


        if nbands == 1:   # 1D signal
            fig = matplotlib.pyplot.figure()
            #ax0 = fig.add_subplot(len(self.pyrSize), 1, 1)
            #ax0.set_frame_on(False)
            #ax0.get_xaxis().tick_bottom()
            #ax0.get_xaxis().tick_top()
            #ax0.get_yaxis().tick_right()
            #ax0.get_yaxis().tick_left()
            #ax0.get_yaxis().set_visible(False)
            for bnum in range(nind):
                band = self.band(bnum)
                pylab.subplot(len(self.pyrSize), 1, bnum+1)
                pylab.plot(band.T)
            matplotlib.pyplot.show()
        else:
            colormap = matplotlib.cm.Greys_r
            bg = 255

            # compute positions of subbands
            llpos = numpy.ones((nind,2));

            for lnum in range(ht):
                ind1 = lnum*nbands
                xpos = self.pyrSize[ind1][1] + 1 + gap*(ht-lnum+1);
                ypos = self.pyrSize[ind1+1][0] + 1 + gap*(ht-lnum+1);
                llpos[ind1:ind1+3, :] = [[ypos, 1], [1, xpos], [ypos, xpos]]
            llpos[nind-1,:] = [1, 1]   # lowpass
    
            # make position list positive, and allocate appropriate image:
            llpos = llpos - ((numpy.ones((nind,1)) * numpy.amin(llpos, axis=0)) + 1) + 1
            urpos = llpos + self.pyrSize
            d_im = numpy.ones((numpy.amax(urpos), numpy.amax(urpos))) * bg
        
            # paste bands into image, (im-r1)*(nshades-1)/(r2-r1) + 1.5
            nshades = 64;
            for bnum in range(nind):
                mult = (nshades-1) / (prange[bnum,1]-prange[bnum,0])
                d_im[llpos[bnum,0]:urpos[bnum,0], 
                     llpos[bnum,1]:urpos[bnum,1]] = mult * self.band(bnum) + (1.5-mult*prange[bnum,0])
            
            if disp == 'qt':
                pyPyrUtils.showIm(d_im, 'auto', 2)
            elif disp == 'nb':
                JBhelpers.showIm(d_im, 'auto', 2)
