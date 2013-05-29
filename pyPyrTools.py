import numpy as np
import pyPyrUtils as ppu
from myModule import *
import math
import matplotlib.cm as cm
from pylab import *
from operator import mul

class pyramid:  # pyramid
    # properties
    pyr = {}
    pyrSize = {}
    pyrType = ''
    image = ''
    height = ''

    # constructor
    def __init__(self):
        print "please specify type of pyramid to create (Gpry, Lpyr, etc.)"
        exit(1)

    # methods
    def band(self, bandNum):
        sortedKeys = sorted(self.pyr.keys(), reverse=True, 
                            key=lambda element: (element[0], element[1]))
        return np.array(self.pyr[sortedKeys[bandNum]])

    def showPyr(self, *args):
        if self.pyrType == 'Gaussian':
            self.showLpyr(args)
        elif self.pyrType == 'Laplacian':
            self.showLpyr(args)
        else:
            print "pyramid type %s not currently supported" % (args[0])
            exit(1)

class Spyr(pyramid):
    filt = ''
    edges = ''
    
    #constructor
    def __init__(self, *args):    # (image height, filter file, edges)
        self.pyrType = 'steerable'
        if len(args) > 0:
            self.image = args[0]
        else:
            print "First argument (image) is required."
            exit(1)

        #------------------------------------------------
        # defaults:

        if len(args) > 1:
            if args[1] == 'sp0Filters':
                filters = ppu.sp0Filters()
            elif args[1] == 'sp1Filters':
                filters = ppu.sp1Filters()
            elif os.path.isfile(args[1]):
                print "Filter files not supported yet"
                exit(1)
            else:
                print "supported filter parameters are 'sp0Filters' and 'sp1Filters'"
                exit(1)
        else:
            filters = ppu.sp1Filters()

        harmonics = filters['harmonics']
        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']
            
        if len(args) > 2:
            edges = args[2]
        else:
            edges = 'reflect1'

        max_ht = ppu.maxPyrHt(self.image.shape, lofilt.shape)  # just lofilt[1]?
        if len(args) > 3:
            if args[3] == 'auto':
                ht = max_ht
            elif args[3] > max_ht:
                print "Error: cannot build pyramid higher than %d levels." % (
                    max_ht)
                exit(1)
            else:
                ht = args[3]
        else:
            ht = max_ht

        #------------------------------------------------------

        self.pyr = {}
        im = self.image
        im_sz = im.shape
        pyrCtr = 0

        hi0 = corrDn(im_sz[0], im_sz[1], im, hi0filt.shape[0], hi0filt.shape[1],
                     hi0filt, edges);
        hi0 = np.array(hi0).reshape(im_sz[0], im_sz[1])
        #self.pyr[hi0.shape] = hi0
        self.pyr[pyrCtr] = hi0
        pyrCtr += 1

        lo = corrDn(im_sz[0], im_sz[1], im, lo0filt.shape[0], lo0filt.shape[1],
                    lo0filt, edges);
        lo = np.array(lo).reshape(im_sz[0], im_sz[1])

        for i in range(ht):
            lo_sz = lo.shape
            print "i = %d" % (i)
            print lo_sz
            # assume square filters  -- start of buildSpyrLevs
            bfiltsz = math.floor(math.sqrt(bfilts.shape[0]))
            #bands = np.zeros((reduce(mul, lo.shape, 1), bfilts.shape[1]))
            #bind = np.zeros((bfilts.shape[0], 2))

            for b in range(bfilts.shape[1]):
                filt = bfilts[:,b].reshape(bfiltsz,bfiltsz)
                band = corrDn(lo_sz[0], lo_sz[1], lo, bfiltsz, bfiltsz, filt, 
                              edges)
                #print bands.shape
                #print band.shape
                #bands[:,b] = band.copy()
                #bind[b,:] = band.shape
                self.pyr[pyrCtr] = np.array(band.copy()).reshape(lo_sz[0], 
                                                                 lo_sz[1])
                pyrCtr += 1

            #self.pyr[bind] = bands.copy()
            #self.pyr[bands.shape] = bands.copy()
            #pyrCtr += 1

            lo = corrDn(lo_sz[0], lo_sz[1], lo, lofilt.shape[0], 
                        lofilt.shape[1], lofilt, edges, 2, 2)
            lo = np.array(lo).reshape(lo_sz[0]/2, lo_sz[1]/2)

        self.pyr[pyrCtr] = lo
    # methods

class Lpyr(pyramid):
    filt = ''
    edges = ''

    # constructor
    def __init__(self, *args):    # (image, height, filter1, filter2, edges)
        #args = args[0]        
        self.pyrType = 'Laplacian'
        if len(args) > 0:
            self.image = args[0]
        else:
            print "pyr = Lpyr(image, height, filter1, filter2, edges)"
            print "First argument (image) is required"
            exit(1)

        if len(args) > 2:
            filt = args[2]
            if not (filt.shape == 1).any():
                print "Error: filter1 should be a 1D filter (i.e., a vector)"
                exit(1)
        else:
            filt = ppu.namedFilter('binom5')
            if self.image.shape[0] == 1:
                filt = filt.reshape(1,5)

        if len(args) > 3:
            filt = args[3]
            if not (filt.shape == 1).any():
                print "Error: filter2 should be a 1D filter (i.e., a vector)"
                exit(1)
        else:
            filt = ppu.namedFilter('binom5')
            if self.image.shape[0] == 1:
                filt = filt.reshape(1,5)

        maxHeight = 1 + ppu.maxPyrHt(self.image.shape, filt.shape)

        if len(args) > 1:
            if args[1] is "auto":
                self.height = maxHeight
            else:
                self.height = args[1]
                if self.height > maxHeight:
                    print ( "Error: cannot build pyramid higher than %d levels"
                            % (maxHeight) )
                    exit(1)
        else:
            self.height = maxHeight

        if len(args) > 4:
            edges = args[4]
        else:
            edges = "reflect1"

        # make pyramid
        self.pyr = {}
        im = np.array(self.image).astype(float)
        if len(im.shape) == 1:
            im = im.reshape(im.shape[0], 1)
        los = {}
        los[self.height] = im
        # compute low bands
        for ht in range(self.height-1,0,-1):
            im_sz = im.shape
            filt_sz = filt.shape
            if len(im_sz) == 1:
                lo2 = corrDn(im_sz[0], 1, im, filt_sz[0], filt_sz[1], filt, 
                             edges, 2, 1, 0, 0, im_sz[0], 1)
                lo2 = np.array(lo2).reshape(math.ceil(im_sz[0]/2.0), 1, 
                                            order='C')
            elif im_sz[0] == 1:
                lo2 = corrDn(1, im_sz[1], im, filt_sz[0], filt_sz[1], filt, 
                             edges, 1, 2, 0, 0, 1, im_sz[1])
                lo2 = np.array(lo2).reshape(1, math.ceil(im_sz[1]/2.0), 
                                            order='C')
            elif im_sz[1] == 1:
                lo2 = corrDn(im_sz[0], 1, im, filt_sz[0], filt_sz[1], filt, 
                             edges, 2, 1, 0, 0, math.ceil(im_sz[0]/2.0), 1)
                lo2 = np.array(lo2).reshape(math.ceil(im_sz[0]/2.0), 1, 
                                            order='C')
            else:
                lo = corrDn(im_sz[0], im_sz[1], im, filt_sz[0], filt_sz[1], 
                            filt, edges, 2, 1, 0, 0, im_sz[0], im_sz[1])
                lo = np.array(lo).reshape(im_sz[0]/1, math.ceil(im_sz[1]/2.0), 
                                          order='C')
                #int_sz = lo.shape
                lo2 = corrDn(im_sz[0]/1, math.ceil(im_sz[1]/2.0), lo.T, 
                             filt_sz[0], filt_sz[1], filt, edges, 2, 1, 0, 0, 
                             im_sz[0], im_sz[1])
                lo2 = np.array(lo2).reshape(math.ceil(im_sz[0]/2.0), 
                                            math.ceil(im_sz[1]/2.0), order='F')
            
            los[ht] = lo2
                
            im = lo2

        self.pyr[lo2.shape] = lo2
        # compute hi bands
        im = self.image
        for ht in range(self.height, 1, -1):
            im = los[ht-1]
            im_sz = los[ht-1].shape
            filt_sz = filt.shape
            if len(im_sz) == 1:
                hi2 = upConv(im_sz[0], 1, im.T, filt_sz[0], filt_sz[1], 
                            filt, edges, 2, 1, 0, 0, los[ht].shape[0], im_sz[1])
                hi2 = np.array(hi2).reshape(los[ht].shape[0], 1, order='F')
            elif im_sz[0] == 1:
                hi2 = upConv(1, im_sz[1], im.T, filt_sz[0], filt_sz[1], 
                            filt, edges, 1, 2, 0, 0, 1, los[ht].shape[1])
                hi2 = np.array(hi2).reshape(1, los[ht].shape[1], order='F')
            elif im_sz[1] == 1:
                hi2 = upConv(im_sz[0], 1, im.T, filt_sz[0], filt_sz[1], 
                            filt, edges, 2, 1, 0, 0, los[ht].shape[0], 1)
                hi2 = np.array(hi2).reshape(los[ht].shape[0], im_sz[1], 
                                            order='F')
            else:
                hi = upConv(im_sz[0], im_sz[1], im.T, filt_sz[0], filt_sz[1], 
                            filt, edges, 2, 1, 0, 0, los[ht].shape[0], 
                            im_sz[1])
                hi = np.array(hi).reshape(los[ht].shape[0], im_sz[1], order='F')
                #int_sz = hi.shape
                hi2 = upConv(im_sz[0], los[ht].shape[1], hi, filt_sz[0], 
                             filt_sz[1], filt, edges, 2, 1, 0, 0, 
                             los[ht].shape[0], los[ht].shape[1])
                hi2 = np.array(hi2).reshape(los[ht].shape[0], los[ht].shape[1],
                                            order='C')

            hi2 = los[ht] - hi2
            self.pyr[hi2.shape] = hi2

    # methods
    def reconLpyr(self, *args):
        if len(args) > 0:
            levs = args[0]
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
            #if levs.any() > maxLev:
            if any(x > maxLev for x in levs):
                print ( "Error: level numbers must be in the range [1, %d]." % 
                        (maxLev) )
                exit(1)

        if isinstance(filt2, basestring):
            filt2 = ppu.namedFilter(filt2)

        res = []
        lastLev = -1
        #for lev in levs:
        for lev in range(maxLev-1, -1, -1):
            if lev in levs and len(res) == 0:
                res = self.band(lev)
            elif len(res) != 0:
                res_sz = res.shape
                filt2_sz = filt2.shape
                hi = upConv(res_sz[0], res_sz[1], res.T, filt2_sz[0], 
                            filt2_sz[1], filt2, edges, 2, 1, 0, 0, 
                            res_sz[0]*2, res_sz[1])
                hi = np.array(hi).reshape(res_sz[0]*2, res_sz[1], order='F')
                hi2 = upConv(res_sz[0], res_sz[1]*2, hi, filt2_sz[0], 
                             filt2_sz[1], filt2, edges, 2, 1, 0, 0,
                             res_sz[0]*2, res_sz[1]*2)
                hi2 = np.array(hi2).reshape(res_sz[0]*2, res_sz[1]*2, order='C')
                if lev in levs:
                    bandIm = self.band(lev)
                    bandIm_sz = bandIm.shape

                    res = hi2 + bandIm
                else:
                    res = hi2

        return res                           
                
    def pyrLow(self):
        return np.array(self.band(self.height-1))

    def showLpyr(self, *args):
        #if any(x == 1 for x in args[0][0].band(0).shape):
        if any(x == 1 for x in self.band(0).shape):
            oned = 1
        else:
            oned = 0

        if len(args) <= 1:
            if oned == 1:
                pRange = 'auto1'
            else:
                pRange = 'auto2'
        
        if len(args) <= 2:
            gap = 1

        if len(args) <= 3:
            if oned == 1:
                scale = math.sqrt(2)
            else:
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
            stdev = math.sqrt( sqsum / (numpixels-1) )
            pRange = np.outer( pRange, np.array([-3*stdev, 3*stdev]) )
            band = self.pyrLow()
            av = np.mean(band)
            stdev = np.std(band)
            pRange[nind-1,:] = np.array([av-2*stdev, av+2*stdev]);#by ref? safe?
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
        elif isinstance(pRange, basestring):
            print "Error: band range argument: %s" % (pRange)
            exit(1)
        elif pRange.shape[0] == 1 and pRange.shape[1] == 2:
            scales = np.power( np.array( range(0,nind) ), scale)
            pRange = np.outer( scales, pRange )
            band = self.pyrLow()
            pRange[nind,:] = ( pRange[nind,:] + np.mean(band) - 
                               np.mean(pRange[nind,:]) )

        # draw
        if oned == 1:
            fig = plt.figure()
            ax0 = fig.add_subplot(nind, 1, 0)
            ax0.set_frame_on(False)
            ax0.get_xaxis().tick_bottom()
            ax0.get_xaxis().tick_top()
            ax0.get_yaxis().tick_right()
            ax0.get_yaxis().tick_left()
            ax0.get_yaxis().set_visible(False)
            for bnum in range(0,nind):
                subplot(nind, 1, bnum+1)
                plot(np.array(range(np.amax(self.band(bnum).shape))).T, 
                     self.band(bnum).T)
                ylim(pRange[bnum,:])
                xlim((0,self.band(bnum).shape[1]-1))
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
            #llpos = llpos - np.ones((nind,1))*np.min(llpos) + 1
            llpos = llpos - np.ones((nind,1))*np.min(llpos)
            pind = range(self.height)
            for i in pind:
                pind[i] = self.band(i).shape
            #urpos = llpos + pind - 1
            urpos = llpos + pind
            d_im = np.zeros((np.max(urpos), np.max(urpos)))  # need bg here?
            
            # paste bands info image, (im-r1)*(nshades-1)/(r2-r1) + 1.5
            nshades = 256
            for bnum in range(nind):
                mult = (nshades-1) / (pRange[bnum,1]-pRange[bnum,0])
                d_im[llpos[bnum,0]:urpos[bnum,0], llpos[bnum,1]:urpos[bnum,1]]=(
                    mult*self.band(bnum) + (1.5-mult*pRange[bnum,0]) )
                # layout works
                #d_im[llpos[bnum,0]:urpos[bnum,0],llpos[bnum,1]:urpos[bnum,1]]=(
                    #(bnum+1)*10 )

            
            ppu.showIm(d_im)

class Gpyr(Lpyr):
    filt = ''
    edges = ''

    # constructor
    def __init__(self, *args):    # (image, height, filter, edges)
        #args = args[0]
        self.pyrType = 'Gaussian'
        if len(args) < 1:
            print "pyr = Gpyr(image, height, filter, edges)"
            print "First argument (image) is required"
            exit(1)
        else:
            self.image = args[0]

        if len(args) > 2:
            filt = args[2]
            if not (filt.shape == 1).any():
                print "Error: filt should be a 1D filter (i.e., a vector)"
                exit(1)
        else:
            filt = ppu.namedFilter('binom5')
            if self.image.shape[0] == 1:
                filt = filt.reshape(1,5)

        maxHeight = 1 + ppu.maxPyrHt(self.image.shape, filt.shape)

        if len(args) > 1:
            if args[1] is "auto":
                self.height = maxHeight
            else:
                self.height = args[1]
                if self.height > maxHeight:
                    print ( "Error: cannot build pyramid higher than %d levels"
                            % (maxHeight) )
                    exit(1)
        else:
            self.height = maxHeight

        if len(args) > 3:
            edges = args[3]
        else:
            edges = "reflect1"

        # make pyramid
        self.pyr = {}
        im = np.array(self.image).astype(float)

        if len(im.shape) == 1:
            im = im.reshape(im.shape[0], 1)

        self.pyr[im.shape] = im
        for ht in range(self.height,0,-1):
            if ht <= 1:
                self.pyr[im.shape] = im
            else:
                im_sz = im.shape
                filt_sz = filt.shape
                if len(im_sz) == 1:
                    lo2 = corrDn(im_sz[0], 1, im, filt_sz[0], filt_sz[1], filt, 
                                 edges, 2, 1, 0, 0, im_sz[0], 1)
                    lo2 = np.array(lo2).reshape(im_sz[0]/2, 1, order='C')
                elif im_sz[0] == 1:
                    lo2 = corrDn(1, im_sz[1], im, filt_sz[0], filt_sz[1], filt, 
                                 edges, 1, 2, 0, 0, 1, im_sz[1])
                    lo2 = np.array(lo2).reshape(1, im_sz[1]/2, order='C')
                elif im_sz[1] == 1:
                    lo2 = corrDn(im_sz[0], 1, im, filt_sz[0], filt_sz[1], filt, 
                                 edges, 2, 1, 0, 0, im_sz[0], 1)
                    lo2 = np.array(lo2).reshape(im_sz[0]/2, 1, order='C')
                else:
                    lo = corrDn(im_sz[0], im_sz[1], im, filt_sz[0], filt_sz[1], 
                                filt, edges, 2, 1, 0, 0, im_sz[0], im_sz[1])
                    lo = np.array(lo).reshape(im_sz[0]/1, 
                                              math.ceil(im_sz[1]/2.0), 
                                              order='C')
                    lo2 = corrDn(im_sz[0]/1, math.ceil(im_sz[1]/2.0), lo.T, 
                                 filt_sz[0], 
                                 filt_sz[1], filt, edges, 2, 1, 0, 0, im_sz[0], 
                                 im_sz[1])
                    lo2 = np.array(lo2).reshape(math.ceil(im_sz[0]/2.0), 
                                                math.ceil(im_sz[1]/2.0), 
                                                order='F')

                self.pyr[lo2.shape] = lo2

                im = lo2
        
    # methods

                 
            
