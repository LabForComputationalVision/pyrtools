import numpy as np
from .Lpyr import Lpyr
from .pyr_utils import LB2idx
from .namedFilter import namedFilter
from .modulateFlip import modulateFlip
from .maxPyrHt import maxPyrHt
from .c.wrapper import corrDn, upConv
from ..tools.showIm import showIm
import matplotlib.pyplot as plt
from matplotlib import cm

# import pylab
# from ..tools import JBhelpers

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
            filt = namedFilter(filt)

        if len(filt.shape) != 1 and filt.shape[0] != 1 and filt.shape[1] != 1:
            print("Error: filter should be 1D (i.e., a vector)");
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
                print("Error: cannot build pyramid higher than %d levels." % (max_ht))
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
                levs = np.array(args[2])
            else:
                levs = args[2]
        else:
            levs = 'all'

        if len(args) > 3:
            if not isinstance(args[3], str):
                bands = np.array(args[3])
            else:
                bands = args[3]
        else:
            bands = 'all'

        #------------------------------------------------------

        maxLev = int(self.wpyrHt() + 1)

        if isinstance(levs, str) and levs == 'all':
            levs = np.array(list(range(maxLev)))
        else:
            tmpLevs = []
            for l in levs:
                tmpLevs.append((maxLev-1)-l)
            levs = np.array(tmpLevs)
            if (levs > maxLev).any():
                print("Error: level numbers must be in the range [0, %d]" % (maxLev))
        allLevs = np.array(list(range(maxLev)))

        if isinstance(bands, str) and bands == "all":
            if ( len(self.band(0)) == 1 or self.band(0).shape[0] == 1 or
                 self.band(0).shape[1] == 1 ):
                bands = np.array([0]);
            else:
                bands = np.array(list(range(3)))
        else:
            bands = np.array(bands)
            if (bands < 0).any() or (bands > 2).any():
                print("Error: band numbers must be in the range [0,2].")

        if isinstance(filt, str):
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
                    res = np.zeros(self.pyr[len(self.pyr)-1].shape)
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
                    lres_sz = ([self.pyrSize[resIdx][0], res_sz[1]])
                    hres_sz = ([self.pyrSize[resIdx-1][0], res_sz[1]])
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
                                     start = (1,0), stop = res_sz,
                                     result = res)
                    idx += 1
                    if 1 in bands and lev in levs:
                        ires = upConv(image = self.band(idx), filt = hfilt,
                                      edges = edges, step = (1,2),
                                      start = (0,1), stop = lres_sz)
                        res = upConv(image = ires, filt = filt, edges = edges,
                                     step = (2,1), start = (stag-1,0),
                                     stop = res_sz, result = res)
                    idx += 1
                    if 2 in bands and lev in levs:
                        ires = upConv(image = self.band(idx), filt = hfilt,
                                      edges = edges, step = (1,2),
                                      start = (0,1), stop = hres_sz)
                        res = upConv(image = ires, filt = hfilt.T,
                                     edges = edges, step = (2,1),
                                     start = (1,0), stop = res_sz,
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
        return np.array(self.band(len(self.pyrSize)-1))

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
            scale = np.sqrt(2)
        elif scale is None and nbands == 3:
            scale = 2

        ht = int(self.wpyrHt())
        nind = len(self.pyr)

        ## Auto range calculations:
        if prange == 'auto1':
            prange = np.ones((nind,1))
            mn = 0.0
            mx = 0.0
            for lnum in range(1,ht+1):
                for bnum in range(nbands):
                    idx = LB2idx(lnum, bnum, ht+2, nbands)
                    band = self.band(idx)/(np.power(scale,lnum))
                    prange[(lnum-1)*nbands+bnum+1] = np.power(scale,lnum-1)
                    bmn = np.amin(band)
                    bmx = np.amax(band)
                    mn = min([mn, bmn])
                    mx = max([mx, bmx])
            if nbands == 1:
                pad = (mx-mn)/12
                mn = mn-pad
                mx = mx+pad
            prange = np.outer(prange, np.array([mn, mx]))
            band = self.pyrLow()
            mn = np.amin(band)
            mx = np.amax(band)
            if nbands == 1:
                pad = (mx-mn)/12
                mn = mn-pad
                mx = mx+pad
            prange[nind-1,:] = np.array([mn, mx])
        elif prange == 'indep1':
            prange = np.zeros((nind,2))
            for bnum in range(nind):
                band = self.band(bnum)
                mn = band.min()
                mx = band.max()
                if nbands == 1:
                    pad = (mx-mn)/12
                    mn = mn-pad
                    mx = mx+pad
                prange[bnum,:] = np.array([mn, mx])
        elif prange == 'auto2':
            prange = np.ones(nind)
            sqsum = 0
            numpixels = 0
            for lnum in range(1,ht+1):
                for bnum in range(nbands):
                    band = self.band(LB2idx(lnum, bnum, ht, nbands))
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
            print("Error:Bad RANGE argument: %s'" % (prange))
        elif prange.shape[0] == 1 and prange.shape[1] == 2:
            scales = np.power(scale, list(range(ht)))
            scales = np.outer( np.ones((nbands,1)), scales )
            scales = np.array([1, scales, np.power(scale, ht)])
            prange = np.outer(scales, prange)
            band = self.pyrLow()
            prange[nind,:] += np.mean(band) - np.mean(prange[nind,:])


        if nbands == 1:   # 1D signal
            fig = plt.figure()
            #ax0 = fig.add_subplot(len(self.pyrSize), 1, 1)
            #ax0.set_frame_on(False)
            #ax0.get_xaxis().tick_bottom()
            #ax0.get_xaxis().tick_top()
            #ax0.get_yaxis().tick_right()
            #ax0.get_yaxis().tick_left()
            #ax0.get_yaxis().set_visible(False)
            for bnum in range(nind):
                band = self.band(bnum)
                plt.subplot(len(self.pyrSize), 1, bnum+1)
                plt.plot(band.T)
            plt.tight_layout()
            plt.show()
        else:
            colormap = cm.Greys_r
            bg = 255

            # compute positions of subbands
            llpos = np.ones((nind,2));

            for lnum in range(ht):
                ind1 = lnum*nbands
                xpos = self.pyrSize[ind1][1] + 1 + gap*(ht-lnum+1);
                ypos = self.pyrSize[ind1+1][0] + 1 + gap*(ht-lnum+1);
                llpos[ind1:ind1+3, :] = [[ypos, 1], [1, xpos], [ypos, xpos]]
            llpos[nind-1,:] = [1, 1]   # lowpass

            # make position list positive, and allocate appropriate image:
            llpos = llpos - ((np.ones((nind,1)) * np.amin(llpos, axis=0)) + 1) + 1
            llpos = llpos.astype(int)
            urpos = llpos + self.pyrSize
            d_im = np.ones((np.amax(urpos), np.amax(urpos))) * bg

            # paste bands into image, (im-r1)*(nshades-1)/(r2-r1) + 1.5
            nshades = 64;
            for bnum in range(nind):
                mult = (nshades-1) / (prange[bnum,1]-prange[bnum,0])
                d_im[llpos[bnum,0]:urpos[bnum,0],
                     llpos[bnum,1]:urpos[bnum,1]] = mult * self.band(bnum) + (1.5-mult*prange[bnum,0])

            if disp == 'qt':
                showIm(d_im, 'auto', 2)
            # elif disp == 'nb':
            #     JBhelpers.showIm(d_im, 'auto', 2)
