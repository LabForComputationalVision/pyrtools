from Spyr import Spyr
import numpy
from steer2HarmMtx import steer2HarmMtx
from rcosFn import rcosFn
from pointOp import pointOp
import scipy

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

        steermtx = steer2HarmMtx(harmonics, numpy.pi *
                                 numpy.array(range(nbands))/nbands, 'even')

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
        (Xrcos, Yrcos) = rcosFn(twidth, (-twidth/2.0), numpy.array([0,1]))
        Yrcos = numpy.sqrt(Yrcos)

        YIrcos = numpy.sqrt(1.0 - Yrcos**2)
        lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)

        imdft = numpy.fft.fftshift(numpy.fft.fft2(self.image))

        self.pyr = []
        self.pyrSize = []

        hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)

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
            Xcosn = numpy.pi * numpy.array(range(-(2*lutsize+1), (lutsize+2))) / lutsize

            order = nbands -1
            const = (2**(2*order))*(scipy.misc.factorial(order, exact=True)**2)/float(nbands*scipy.misc.factorial(2*order, exact=True))
            Ycosn = numpy.sqrt(const) * (numpy.cos(Xcosn))**order
            log_rad_test = numpy.reshape(log_rad,(1,
                                                  log_rad.shape[0]*
                                                  log_rad.shape[1]))
            himask = pointOp(log_rad_test, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0],
                             0)
            himask = numpy.reshape(himask, (lodft.shape[0], lodft.shape[1]))

            for b in range(nbands):
                angle_tmp = numpy.reshape(angle, 
                                          (1,angle.shape[0]*angle.shape[1]))
                anglemask = pointOp(angle_tmp, Ycosn,
                                    Xcosn[0]+numpy.pi*b/nbands,
                                    Xcosn[1]-Xcosn[0],0)

                anglemask = anglemask.reshape(lodft.shape[0], lodft.shape[1])
                banddft = ( ((-numpy.power(-1+0j,0.5))**order) * lodft *
                            anglemask * himask )
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
            lomask = pointOp(log_rad_tmp, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0],
                             0)
            
            lodft = lodft * lomask.reshape(lodft.shape[0], lodft.shape[1])

        lodft = numpy.fft.ifft2(numpy.fft.ifftshift(lodft))
        self.pyr.append(numpy.real(numpy.array(lodft).copy()))
        self.pyrSize.append(lodft.shape)

    # methods
    def numBands(self):      # FIX: why isn't this inherited
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
        (Xrcos, Yrcos) = rcosFn(twidth, (-twidth/2.0), numpy.array([0,1]))
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

        nlog_rad_tmp = numpy.reshape(nlog_rad, 
                                     (1,nlog_rad.shape[0]*nlog_rad.shape[1]))
        lomask = pointOp(nlog_rad_tmp, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)
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
            nlog_rad1 = log_rad[bounds1[0]:bounds1[2], bounds1[1]:bounds1[3]]
            nlog_rad2 = log_rad[bounds2[0]:bounds2[2],bounds2[1]:bounds2[3]]
            dims = dimList[idx]
            nangle = angle[bounds1[0]:bounds1[2], bounds1[1]:bounds1[3]]
            YIrcos = numpy.abs(numpy.sqrt(1.0 - Yrcos**2))
            if idx > 1:
                Xrcos += numpy.log2(2.0)
                nlog_rad2_tmp = numpy.reshape(nlog_rad2, 
                                              (1,nlog_rad2.shape[0]*
                                               nlog_rad2.shape[1]))
                lomask = pointOp(nlog_rad2_tmp, YIrcos, Xrcos[0],
                                 Xrcos[1]-Xrcos[0], 0)

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
                        himask = pointOp(nlog_rad1_tmp, Yrcos, Xrcos[0],
                                         Xrcos[1]-Xrcos[0], 0)

                        himask = himask.reshape(nlog_rad1.shape)
                        nangle_tmp = numpy.reshape(nangle, (1,
                                                            nangle.shape[0]*
                                                            nangle.shape[1]))
                        anglemask = pointOp(nangle_tmp, Ycosn,
                                            Xcosn[0]+numpy.pi*b/nbands,
                                            Xcosn[1]-Xcosn[0], 0)

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
        lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)

        lo0mask = lo0mask.reshape(dims[0], dims[1])
        resdft = resdft * lo0mask
        
        # residual highpass subband
        hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)

        hi0mask = hi0mask.reshape(resdft.shape[0], resdft.shape[1])
        if 0 in levs:
            hidft = numpy.fft.fftshift(numpy.fft.fft2(self.pyr[0]))
        else:
            hidft = numpy.zeros(self.pyr[0].shape)
        resdft += hidft * hi0mask

        outresdft = numpy.real(numpy.fft.ifft2(numpy.fft.ifftshift(resdft)))

        return outresdft

    reconPyr = reconSFpyr
