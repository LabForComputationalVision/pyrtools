from pyramid import pyramid
from corrDn import corrDn
from namedFilter import namedFilter
from maxPyrHt import maxPyrHt
from upConv import upConv
from showIm import showIm
import JBhelpers
import numpy
import math
import matplotlib

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
                filt1 = namedFilter(filt1)
            elif len(filt1.shape) != 1 and ( filt1.shape[0] != 1 and
                                             filt1.shape[1] != 1 ):
                print "Error: filter1 should be a 1D filter (i.e., a vector)"
                return
        else:
            filt1 = namedFilter('binom5')
        if len(filt1.shape) == 1:
            filt1 = filt1.reshape(1,len(filt1))
        elif self.image.shape[0] == 1:
            filt1 = filt1.reshape(filt1.shape[1], filt1.shape[0])

        if len(args) > 3:
            filt2 = args[3]
            if isinstance(filt2, basestring):
                filt2 = namedFilter(filt2)
            elif len(filt2.shape) != 1 and ( filt2.shape[0] != 1 and
                                            filt2.shape[1] != 1 ):
                print "Error: filter2 should be a 1D filter (i.e., a vector)"
                return
        else:
            filt2 = filt1

        maxHeight = 1 + maxPyrHt(self.image.shape, filt1.shape)

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
        for ht in range(self.height-1,0,-1):
            im_sz = im.shape
            filt1_sz = filt1.shape
            if im_sz[0] == 1:
                lo2 = corrDn(image = im, filt = filt1, edges = edges,
                             step = (1,2))
                #lo2 = numpy.array(lo2)
            elif len(im_sz) == 1 or im_sz[1] == 1:
                lo2  = corrDn(image = im, filt = filt1, edges = edges,
                              step = (2,1))
                #lo2 = numpy.array(lo2)
            else:
                lo = corrDn(image = im, filt = filt1.T, edges = edges,
                            step = (1,2), start = (0,0))
                #lo = numpy.array(lo)
                lo2 = corrDn(image = lo, filt = filt1, edges = edges,
                             step = (2,1), start = (0,0))
                #lo2 = numpy.array(lo2)

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
                hi2 = upConv(image = im, filt = filt2.T, edges = edges,
                             step = (1,2), stop = (los[ht].shape[1],
                                                   los[ht].shape[0])).T
            elif im_sz[0] == 1:
                hi2 = upConv(image = im, filt = filt2.T, edges = edges,
                             step = (2,1), stop = (los[ht].shape[1],
                                                   los[ht].shape[0])).T
            else:
                hi = upConv(image = im, filt = filt2, edges = edges,
                            step = (2,1), stop = (los[ht].shape[0], im_sz[1]))
                hi2 = upConv(image = hi, filt = filt2.T, edges = edges,
                             step = (1,2), stop = (los[ht].shape[0], 
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
    def set(self, *args):
        if len(args) != 3:
            print 'Error: three input parameters required:'
            print '  set(band, element(tuple), value)'
        self.pyr[args[0]][args[1][0]][args[1][1]] = args[2] 

    def reconPyr(self, *args):
        if len(args) > 0:
            if not isinstance(args[0], str):
                levs = numpy.array(args[0])
            else:
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

        if isinstance(levs, (str,basestring)) and levs == 'all':
            levs = range(0,maxLev)
        else:
            if (levs > maxLev-1).any():
                print ( "Error: level numbers must be in the range [0, %d]." % 
                        (maxLev-1) )
                return

        if isinstance(filt2, basestring):
            filt2 = namedFilter(filt2)
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
                    hi2 = upConv(image = res, filt = filt2, edges = edges,
                                 step = (2,1), stop = (new_sz[1], new_sz[0])).T
                elif res_sz[1] == 1:
                    hi2 = upConv(image = res, filt = filt2.T, edges = edges,
                                 step = (1,2), stop = (new_sz[1], new_sz[0])).T
                else:
                    hi = upConv(image = res, filt = filt2, edges = edges,
                                step = (2,1), stop = (new_sz[0], res_sz[1]))
                    hi2 = upConv(image = hi, filt = filt2.T, edges = edges,
                                 step = (1,2), stop = (new_sz[0], new_sz[1]))
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

        if pRange is None and oned == 1:
            pRange = 'auto1'
        elif pRange is None and oned == 0:
            pRange = 'auto2'

        if scale is None and oned == 1:
            scale = math.sqrt(2)
        elif scale is None and oned == 0:
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
            # FIX: something here broke with Anaconda update!!
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
            
            # FIX: need a mode to switch between above and below display
            if disp == 'nb':
                JBhelpers.showIm(d_im[:self.band(0).shape[0]][:])
            elif disp == 'qt':
                showIm(d_im[:self.band(0).shape[0]][:])
