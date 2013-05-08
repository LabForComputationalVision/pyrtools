import numpy as np
import pyPyrUtils as ppu
from myModule import *
import math
import matplotlib.cm as cm

class pyramid:  # pyramid
    # properties
    pyr = {}
    pyrType = ''
    image = ''
    height = ''

    # constructor
    def __init__(self, *args):
        if len(args) < 2:
            print "at least two input parameters (type and image) are required"
            exit(1)
        
        if args[0] == 'Gaussian':
            Gpyr(args[1:])
        elif args[0] == 'Laplacian':
            Lpyr(args[1:])
        else:
            print "pyramid type %s not currently supported" % (args[0])
            exit(1)

    # methods
    def band(self, bandNum):
        sortedKeys = sorted(self.pyr.keys(), reverse=True, 
                            key=lambda element: (element[0], element[1]))
        return self.pyr[sortedKeys[bandNum]]

    def showPyr(self, *args):
        print self.pyrType
        if self.pyrType == 'Gaussian':
            self.showLpyr(args)
        elif self.pyrType == 'Laplacian':
            self.showLpyr(args)
        else:
            print "pyramid type %s not currently supported" % (args[0])
            exit(1)

    def pyrLow(self):
        return self.band(self.height-1)

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
            print "1D 'images currently not supported"
            exit(1)
            #pRange = np.zeros(nind,1)
            #mn = 0.0
            #mx = 0.0
            #for bnum in range(1,nind):
            #    band = self.band(bnum)
            #    band /= np.power(scale, bnum-1)
            #    pRange(bnum) = range2(band)   range2 is a function
        elif pRange == 'indep1':
            print "1D 'images currently not supported"
            exit(1)
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
            print "one dimensional 'images' not supported yet"
            exit(1)
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
        

class Gpyr(pyramid):
    filt = ''
    edges = ''

    # constructor
    def __init__(self, *args):    # (image, height, filter, edges)
        args = args[0]
        pyramid.pyrType = 'Gaussian'
        if len(args) < 1:
            print "pyr = Gpyr(image, height, filter, edges)"
            print "First argument (image) is required"
            exit(1)
        else:
            pyramid.image = args[0]

        if len(args) > 2:
            filt = args[2]
            if not (filt.shape == 1).any():
                print "Error: filt should be a 1D filter (i.e., a vector)"
                exit(1)
        else:
            filt = ppu.namedFilter('binom5')

        print pyramid.image
        maxHeight = 1 + ppu.maxPyrHt(pyramid.image.shape, filt.shape)
        print "maxHeight = %d" % (maxHeight)
        if len(args) > 1:
            if args[1] is "auto":
                pyramid.height = maxHeight
            else:
                pyramid.height = args[1]
                if pyramid.height > maxHeight:
                    print ( "Error: cannot build pyramid higher than %d levels"
                            % (maxHeight) )
                    exit(1)
        else:
            pyramid.height = maxHeight

        if len(args) > 3:
            edges = args[3]
        else:
            edges = "reflect1"

        # make pyramid
        im = pyramid.image
        pyramid.pyr[im.shape] = im
        for ht in range(pyramid.height,0,-1):
            if ht <= 1:
                pyramid.pyr[im.shape] = im
            else:
                im_sz = im.shape
                filt_sz = filt.shape
                lo = corrDn(im_sz[0], im_sz[1], im, filt_sz[0], filt_sz[1], 
                            filt, edges, 2, 1, 0, 0, im_sz[0], im_sz[1])
                lo = np.array(lo).reshape(im_sz[0]/1, im_sz[1]/2, order='C')
                lo2 = corrDn(im_sz[0]/1, im_sz[1]/2, lo.T, filt_sz[0], 
                             filt_sz[1], filt, edges, 2, 1, 0, 0, im_sz[0], 
                             im_sz[1])
                lo2 = np.array(lo2).reshape(im_sz[0]/2, im_sz[1]/2, order='F')

                pyramid.pyr[lo2.shape] = lo2
                
                im = lo2

    # medthods

class Lpyr(pyramid):
    filt = ''
    edges = ''

    # constructor
    def __init__(self, *args):    # (image, height, filter1, filter2, edges)
        args = args[0]        
        pyramid.pyrType = 'Laplacian'
        if len(args) > 0:
            pyramid.image = args[0]
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

        if len(args) > 3:
            filt = args[3]
            if not (filt.shape == 1).any():
                print "Error: filter1 should be a 1D filter (i.e., a vector)"
                exit(1)
        else:
            filt = ppu.namedFilter('binom5')

        maxHeight = 1 + ppu.maxPyrHt(pyramid.image.shape, filt.shape)
        print "maxHeight = %d" % (maxHeight)
        if len(args) > 1:
            if args[1] is "auto":
                pyramid.height = maxHeight
            else:
                pyramid.height = args[1]
                if pyramid.height > maxHeight:
                    print ( "Error: cannot build pyramid higher than %d levels"
                            % (maxHeight) )
                    exit(1)
        else:
            pyramid.height = maxHeight
        print "pyramid.height = %d" % (pyramid.height)

        if len(args) > 4:
            edges = args[4]
        else:
            edges = "reflect1"

        # make pyramid
        im = pyramid.image
        los = {}
        los[pyramid.height] = im
        # compute low bands
        for ht in range(pyramid.height-1,0,-1):
            im_sz = im.shape
            filt_sz = filt.shape
            lo = corrDn(im_sz[0], im_sz[1], im, filt_sz[0], filt_sz[1], 
                        filt, edges, 2, 1, 0, 0, im_sz[0], im_sz[1])
            lo = np.array(lo).reshape(im_sz[0]/1, im_sz[1]/2, order='C')
            int_sz = lo.shape
            lo2 = corrDn(im_sz[0]/1, im_sz[1]/2, lo.T, filt_sz[0], 
                         filt_sz[1], filt, edges, 2, 1, 0, 0, im_sz[0], 
                         im_sz[1])
            lo2 = np.array(lo2).reshape(im_sz[0]/2, im_sz[1]/2, order='F')
            
            los[ht] = lo2
                
            im = lo2

        pyramid.pyr[lo2.shape] = lo2
        # compute hi bands
        im = pyramid.image
        for ht in range(pyramid.height, 1, -1):
            im = los[ht-1]
            im_sz = los[ht-1].shape
            filt_sz = filt.shape
            hi = upConv(im_sz[0], im_sz[1], im.T, filt_sz[0], filt_sz[1], 
                         filt, edges, 2, 1, 0, 0, im_sz[0]*2, im_sz[1])
            hi = np.array(hi).reshape(im_sz[0]*2, im_sz[1], order='F')
            int_sz = hi.shape
            hi2 = upConv(im_sz[0], im_sz[1]*2, hi, filt_sz[0], filt_sz[1], 
                         filt, edges, 2, 1, 0, 0, im_sz[0]*2, im_sz[1]*2)
            hi2 = np.array(hi2).reshape(im_sz[0]*2, im_sz[1]*2, order='C')
            hi2 = los[ht] - hi2
            pyramid.pyr[hi2.shape] = hi2

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

        maxLev = pyramid.height
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
        return self.band(self.height-1)

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
            print "1D 'images currently not supported"
            exit(1)
            #pRange = np.zeros(nind,1)
            #mn = 0.0
            #mx = 0.0
            #for bnum in range(1,nind):
            #    band = self.band(bnum)
            #    band /= np.power(scale, bnum-1)
            #    pRange(bnum) = range2(band)   range2 is a function
        elif pRange == 'indep1':
            print "1D 'images currently not supported"
            exit(1)
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
            print "one dimensional 'images' not supported yet"
            exit(1)
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

                 
            
