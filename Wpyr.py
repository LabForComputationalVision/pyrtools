from Lpyr import Lpyr
from LB2idx import LB2idx
from namedFilter import namedFilter
from modulateFlip import modulateFlip
from maxPyrHt import maxPyrHt
from corrDn import corrDn
from upConv import upConv
import JBhelpers
import numpy
import matplotlib
import pylab

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
            filt = namedFilter(filt)

        if len(filt.shape) != 1 and filt.shape[0] != 1 and filt.shape[1] != 1:
            print "Error: filter should be 1D (i.e., a vector)";
            return
        hfilt = modulateFlip(filt)

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

        max_ht = maxPyrHt(im.shape, filt.shape)

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
            if len(im.shape) == 1 or im.shape[1] == 1:
                lolo = corrDn(image = im, filt = filt, edges = edges,
                              step = (2,1), start = (stag-1,0))
                hihi = corrDn(image = im, filt = hfilt, edges = edges,
                              step = (2,1), start = (1, 0))
            elif im.shape[0] == 1:
                lolo = corrDn(image = im, filt = filt, edges = edges,
                              step = (1,2), start = (0, stag-1))
                hihi = corrDn(image = im, filt = hfilt.T, edges = edges,
                              step = (1,2), start = (0,1))
            else:
                lo = corrDn(image = im, filt = filt, edges = edges,
                            step = (2,1), start = (stag-1,0))
                hi = corrDn(image = im, filt = hfilt, edges = edges,
                            step = (2,1), start = (1,0))
                lolo = corrDn(image = lo, filt = filt.T, edges = edges,
                              step = (1,2), start = (0, stag-1))
                lohi = corrDn(image = hi, filt = filt.T, edges = edges,
                              step = (1,2), start = (0,stag-1))
                hilo = corrDn(image = lo, filt = hfilt.T, edges = edges,
                              step = (1,2), start = (0,1))
                hihi = corrDn(image = hi, filt = hfilt.T, edges = edges,
                              step = (1,2), start = (0,1))

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
            if not isinstance(args[2], str):
                levs = numpy.array(args[2])
            else:
                levs = args[2]
        else:
            levs = 'all'

        if len(args) > 3:
            if not isinstance(args[3], str):
                bands = numpy.array(args[3])
            else:
                bands = args[3]
        else:
            bands = 'all'

        #------------------------------------------------------

        maxLev = int(self.wpyrHt() + 1)

        if isinstance(levs, str) and levs == 'all':
            levs = numpy.array(range(maxLev))
        else:
            tmpLevs = []
            for l in levs:
                tmpLevs.append((maxLev-1)-l)
            levs = numpy.array(tmpLevs)
            if (levs > maxLev).any():
                print "Error: level numbers must be in the range [0, %d]" % (maxLev)
        allLevs = numpy.array(range(maxLev))

        if isinstance(bands, str) and bands == "all":
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
            filt = namedFilter(filt)

        hfilt = modulateFlip(filt).T

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
                    res = upConv(image = imageIn, filt = filt.T, edges = edges,
                                 step = (1,2), start = (0,stag-1),
                                 stop = res_sz).T
                elif res_sz[1] == 1:
                    res = upConv(image = imageIn, filt = filt, edges = edges,
                                 step = (2,1), start = (stag-1,0),
                                 stop = res_sz).T
                else:
                    ires = upConv(image = imageIn, filt = filt.T,
                                  edges = edges, step = (1,2),
                                  start = (0,stag-1), stop = lres_sz)
                    res = upConv(image = ires, filt = filt, edges = edges,
                                 step = (2,1), start = (stag-1,0),
                                 stop = res_sz)

                if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or
                     self.pyrSize[0][1] == 1 ):
                    idx = resIdx + 1
                else:
                    idx = resIdx - 1

                if res_sz[0] ==1 and lev in levs:
                    res = upConv(image = self.band(idx), filt = hfilt,
                                 edges = edges, step = (1,2), start = (0,1),
                                 stop = res_sz, result = res)
                    idx -= 1
                elif res_sz[1] == 1 and lev in levs:
                    res = upConv(image = self.band(idx), filt = hfilt.T,
                                 edges = edges, step = (2,1), start = (1,0),
                                 stop = res_sz, result = res)
                    idx -= 1
                elif res_sz[0] != 1 and res_sz[1] != 1 and lev in levs:
                    res_test = res
                    if 0 in bands and lev in levs:
                        ires = upConv(image = self.band(idx), filt = filt.T, 
                                      edges = edges, step = (1,2), 
                                      start = (0, stag-1), stop = hres_sz)
                        res = upConv(image = ires, filt = hfilt.T,
                                     edges = edges, step = (2,1),
                                     start = (1,0), stop = (res_sz[0],
                                                            res_sz[1]),
                                     result = res)
                    idx += 1
                    if 1 in bands and lev in levs:
                        ires = upConv(image = self.band(idx), filt = hfilt, 
                                      edges = edges, step = (1,2),
                                      start = (0,1), stop = lres_sz)
                        res = upConv(image = ires, filt = filt, edges = edges, 
                                     step = (2,1), start = (stag-1,0),
                                     stop = (res_sz[0],res_sz[1]), result = res)
                    idx += 1
                    if 2 in bands and lev in levs:
                        ires = upConv(image = self.band(idx), filt = hfilt, 
                                      edges = edges, step = (1,2),
                                      start = (0,1), stop = (hres_sz[0],
                                                             hres_sz[1]))
                        res = upConv(image = ires, filt = hfilt.T,
                                     edges = edges, step = (2,1),
                                     start = (1,0), stop = (res_sz[0],
                                                            res_sz[1]),
                                     result = res)
                    idx += 1
                # need to jump back n bands in the idx each loop
                if ( len(self.pyrSize[0]) == 1 or self.pyrSize[0][0] == 1 or
                     self.pyrSize[0][1] == 1 ):
                    idx = idx
                else:
                    idx -= 2*len(bands)
        return res

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

    def pyrLow(self):
        return numpy.array(self.band(len(self.pyrSize)-1))

    def showPyr(self, prange = None, gap = 1, scale = None, disp = 'qt'):
        # determine 1D or 2D pyramid:
        if self.pyrSize[0][0] == 1 or self.pyrSize[0][1] == 1:
            nbands = 1
        else:
            nbands = 3

        if prange is None and nbands == 1:
            prange = 'auto1'
        elif prange is None and nbands == 3:
            prange = 'auto2'

        if scale is None and nbands == 1:
            scale = numpy.sqrt(2)
        elif scale is None and nbands == 3:
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
                    idx = LB2idx(lnum, bnum, ht+2, nbands)
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
                    band = self.band(LB2idx(lnum, bnum, ht, nbands))
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
                showIm(d_im, 'auto', 2)
            elif disp == 'nb':
                JBhelpers.showIm(d_im, 'auto', 2)
