import numpy as np
import pyPyrUtils as ppu
from myModule import *

class pyramid:  # pyramid
    # properties
    pyr = {}
    pyrType = ''
    image = ''
    height = ''

    # methods
    def band(self, bandNum):
        sortedKeys = sorted(self.pyr.keys(), reverse=True, 
                            key=lambda element: (element[0], element[1]))
        return self.pyr[sortedKeys[bandNum]]

    #def showPyr(self, *args):
        

class Gpyr(pyramid):
    filt = ''
    edges = ''

    # constructor
    def __init__(self, *args):    # (image, height, filter, edges)
        pyramid.pyrType = 'Gaussian'
        if len(args) < 1:
            print "[pyr, indices] = Gpyr(image, height, filter, edges)"
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

        




            
    
