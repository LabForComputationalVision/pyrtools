import numpy as np
import scipy
import warnings
from .Spyr import Spyr
from .c.wrapper import pointOp
from .steer import steer2HarmMtx
from ..tools.utils import rcosFn

class SFpyr(Spyr):
    filt = None
    edges = None

    #constructor
    def __init__(self, image, height='auto', order=3, twidth=1):
        """Steerable frequency pyramid.

        Construct a steerable pyramid on matrix IM, in the Fourier domain.
        This is similar to Spyr, except that:

            + Reconstruction is exact (within floating point errors)
            + It can produce any number of orientation bands.
            - Typically slower, especially for non-power-of-two sizes.
            - Boundary-handling is circular.

        The squared radial functions tile the Fourier plane with a raised-cosine
        falloff. Angular functions are cos(theta- k*pi/`order`+1)^(`order`).

        Arguments
        ================

        - `image` - a 2D numpy array.

        - `height` (optional) specifies the number of pyramid levels to build.
        'auto' (default) is floor(log2(min(image.shape)))-2

        - `order` (optional), int. Default value is 3.
        The number of orientation bands - 1.

        - `twidth` (optional), int. Default value is 1.
        The width of the transition region of the radial lowpass function, in octaves
        """

        self.pyrType = 'steerableFrequency'
        self.image = np.array(image)

        max_ht = np.floor(np.log2(min(self.image.shape))) - 2
        if height == 'auto':
            ht = max_ht
        elif height > max_ht:
            raise Exception("Cannot build pyramid higher than %d levels." % (max_ht))
        else:
            ht = height
        ht = int(ht)

        if order > 15 or order < 0:
            warnings.warn("order must be an integer in the range [0,15]. Truncating.")
            order = min(max(order, 0), 15)
        order = int(order)

        nbands = order+1

        if twidth <= 0:
            warnings.warn("twidth must be positive. Setting to 1.")
            twidth = 1
        twidth = int(twidth)

        #------------------------------------------------------
        # steering stuff:

        if nbands % 2 == 0:
            harmonics = np.arange(nbands // 2) * 2 + 1
        else:
            harmonics = np.arange((nbands-1) // 2) * 2
        if harmonics.size == 0:
            # in this case, harmonics is an empty matrix. This happens when
            # nbands=0 and (based on how the matlab code acts), in that situation,
            # we actually want harmonics to be 0.
            harmonics = np.array([0])
        self.harmonics = harmonics

        self.steermtx = steer2HarmMtx(harmonics,
                                np.pi * np.arange(nbands)/nbands, even_phase=True)

        #------------------------------------------------------

        dims = np.array(self.image.shape)
        ctr = np.ceil((np.array(dims)+0.5)/2).astype(int)

        (xramp, yramp) = np.meshgrid(np.linspace(-1,1,dims[1]+1)[:-1],
                                     np.linspace(-1,1,dims[0]+1)[:-1])

        angle = np.arctan2(yramp, xramp)
        log_rad = np.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        log_rad = np.log2(log_rad)

        ## Radial transition function (a raised cosine in log-frequency):
        (Xrcos, Yrcos) = rcosFn(twidth, (-twidth/2.0), np.array([0,1]))
        Yrcos = np.sqrt(Yrcos)

        YIrcos = np.sqrt(1.0 - Yrcos**2)
        lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)
        self._lo0mask = lo0mask

        imdft = np.fft.fftshift(np.fft.fft2(self.image))

        self.pyr = []
        self.pyrSize = []

        hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)
        self._hi0mask = hi0mask

        hi0dft = imdft * hi0mask.reshape(imdft.shape[0], imdft.shape[1])
        hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))

        self.pyr.append(np.real(hi0))
        self.pyrSize.append(hi0.shape)

        lo0mask = lo0mask.reshape(imdft.shape[0], imdft.shape[1])
        lodft = imdft * lo0mask

        self._anglemasks = []
        self._himasks = []
        self._lomasks = []

        for i in range(ht):
            bands = np.zeros((lodft.shape[0]*lodft.shape[1], nbands))
            bind = np.zeros((nbands, 2))

            Xrcos -= np.log2(2)

            lutsize = 1024
            Xcosn = np.pi * np.arange(-(2*lutsize+1), (lutsize+2)) / lutsize

            order = nbands -1
            const = (2**(2*order))*(scipy.special.factorial(order, exact=True)**2)/ float(nbands*scipy.special.factorial(2*order, exact=True))
            Ycosn = np.sqrt(const) * (np.cos(Xcosn))**order
            log_rad_test = np.reshape(log_rad,(1,
                                                  log_rad.shape[0]*
                                                  log_rad.shape[1]))
            himask = pointOp(log_rad_test, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0],
                             0)
            himask = np.reshape(himask, (lodft.shape[0], lodft.shape[1]))
            self._himasks.append(himask)

            anglemasks = []
            for b in range(nbands):
                angle_tmp = np.reshape(angle,
                                          (1,angle.shape[0]*angle.shape[1]))
                anglemask = pointOp(angle_tmp, Ycosn,
                                    Xcosn[0]+np.pi*b/nbands,
                                    Xcosn[1]-Xcosn[0],0)

                anglemask = anglemask.reshape(lodft.shape[0], lodft.shape[1])
                anglemasks.append(anglemask)
                banddft = ( ((-np.power(-1+0j,0.5))**order) * lodft *
                            anglemask * himask )
                band = np.fft.ifft2(np.fft.ifftshift(banddft))
                self.pyr.append(np.real(band.copy()))
                self.pyrSize.append(band.shape)

            self._anglemasks.append(anglemasks)
            dims = np.array(lodft.shape)
            ctr = np.ceil((dims+0.5)/2).astype(int)
            lodims = np.ceil((dims-0.5)/2).astype(int)
            loctr = np.ceil((lodims+0.5)/2).astype(int)
            lostart = ctr - loctr
            loend = lostart + lodims

            log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = np.abs(np.sqrt(1.0 - Yrcos**2))
            log_rad_tmp = np.reshape(log_rad,
                                        (1,log_rad.shape[0]*log_rad.shape[1]))
            lomask = pointOp(log_rad_tmp, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0],
                             0).reshape(lodft.shape[0], lodft.shape[1])
            self._lomasks.append(lomask)

            lodft = lodft * lomask

        lodft = np.fft.ifft2(np.fft.ifftshift(lodft))
        self.pyr.append(np.real(np.array(lodft).copy()))
        self.pyrSize.append(lodft.shape)

        self.height = self.spyrHt()

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

    def _reconSFpyr(self, levs='all', bands='all', twidth=1):

        if twidth <= 0:
            warnings.warn("twidth must be positive. Setting to 1.")
            twidth = 1

        #-----------------------------------------------------------------

        nbands = self.numBands()

        maxLev = 1 + self.spyrHt()
        if isinstance(levs, str) and levs == 'all':
            levs = np.arange(maxLev + 1)
        elif isinstance(levs, str):
            raise Exception("%s not valid for levs parameter. "
                           "levs must be either a 1D numpy array"
                            " or the string 'all'." % levs)
        else:
            levs = np.array(levs)

        if isinstance(bands, str) and bands == 'all':
            bands = np.arange(nbands)
        elif isinstance(bands, str):
            raise Exception("%s not valid for bands parameter."
                            "bands must be either a 1D numpy"
                            " array or the string 'all'." % bands)
            return
        else:
            bands = np.array(bands)

        #-------------------------------------------------------------------
        # make list of dims and bounds
        boundList = []
        dimList = []
        for dimIdx in range(len(self.pyrSize)-1,-1,-1):
            dims = np.array(self.pyrSize[dimIdx])
            if (dims[0], dims[1]) not in dimList:
                dimList.append( (dims[0], dims[1]) )
            ctr = np.ceil((dims+0.5)/2).astype(int)
            lodims = np.ceil((dims-0.5)/2).astype(int)
            loctr = np.ceil((lodims+0.5)/2).astype(int)
            lostart = ctr - loctr
            loend = lostart + lodims
            bounds = (lostart[0], lostart[1], loend[0], loend[1])
            if bounds not in boundList:
                boundList.append(bounds)
        boundList.append((0, 0, dimList[len(dimList)-1][0],
                          dimList[len(dimList)-1][1]))
        dimList.append((dimList[len(dimList)-1][0], dimList[len(dimList)-1][1]))

        # matlab code starts here
        dims = np.array(self.pyrSize[0])
        ctr = np.ceil((dims+0.5)/2.0).astype(int)

        (xramp, yramp) = np.meshgrid((np.arange(1,dims[1]+1)-ctr[1])/
                                     (dims[1]/2.),
                                     (np.arange(1,dims[0]+1)-ctr[0])/
                                     (dims[0]/2.))
        angle = np.arctan2(yramp, xramp)
        log_rad = np.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        log_rad = np.log2(log_rad)

        ## Radial transition function (a raised cosine in log-frequency):
        (Xrcos, Yrcos) = rcosFn(twidth, (-twidth/2.0), np.array([0,1]))
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(1.0 - Yrcos**2)

        # from reconSFpyrLevs
        lutsize = 1024
        Xcosn = np.pi * np.arange(-(2*lutsize+1), (lutsize+2)) / lutsize

        order = nbands -1
        const = (2**(2*order))*(scipy.special.factorial(order, exact=True)**2)/ float(nbands*scipy.special.factorial(2*order, exact=True))
        Ycosn = np.sqrt(const) * (np.cos(Xcosn))**order

        # lowest band
        nres = self.pyr[len(self.pyr)-1]
        if self.spyrHt()+1 in levs:
            nresdft = np.fft.fftshift(np.fft.fft2(nres))
        else:
            nresdft = np.zeros(nres.shape)
        resdft = np.zeros(dimList[1]) + 0j

        bounds = (0, 0, 0, 0)
        for idx in range(len(boundList)-2, 0, -1):
            diff = (boundList[idx][2]-boundList[idx][0],
                    boundList[idx][3]-boundList[idx][1])
            bounds = (bounds[0]+boundList[idx][0], bounds[1]+boundList[idx][1],
                      bounds[0]+boundList[idx][0] + diff[0],
                      bounds[1]+boundList[idx][1] + diff[1])
            Xrcos -= np.log2(2.0)
        nlog_rad = log_rad[bounds[0]:bounds[2], bounds[1]:bounds[3]]

        nlog_rad_tmp = np.reshape(nlog_rad,
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
            YIrcos = np.abs(np.sqrt(1.0 - Yrcos**2))
            if idx > 1:
                Xrcos += np.log2(2.0)
                nlog_rad2_tmp = np.reshape(nlog_rad2,
                                              (1,nlog_rad2.shape[0]*
                                               nlog_rad2.shape[1]))
                lomask = pointOp(nlog_rad2_tmp, YIrcos, Xrcos[0],
                                 Xrcos[1]-Xrcos[0], 0)

                lomask = lomask.reshape(bounds2[2]-bounds2[0],
                                        bounds2[3]-bounds2[1])
                lomask = lomask + 0j
                nresdft = np.zeros(dimList[idx]) + 0j
                nresdft[boundList[idx][0]:boundList[idx][2],
                        boundList[idx][1]:boundList[idx][3]] = resdft * lomask
                resdft = nresdft.copy()

            bandIdx -= 2 * nbands

            # reconSFpyrLevs
            if idx != 0 and idx != len(boundList)-1:
                for b in range(nbands):
                    if (bands == b).any():
                        nlog_rad1_tmp = np.reshape(nlog_rad1,
                                                      (1,nlog_rad1.shape[0]*
                                                       nlog_rad1.shape[1]))
                        himask = pointOp(nlog_rad1_tmp, Yrcos, Xrcos[0],
                                         Xrcos[1]-Xrcos[0], 0)

                        himask = himask.reshape(nlog_rad1.shape)
                        nangle_tmp = np.reshape(nangle, (1,
                                                            nangle.shape[0]*
                                                            nangle.shape[1]))
                        anglemask = pointOp(nangle_tmp, Ycosn,
                                            Xcosn[0]+np.pi*b/nbands,
                                            Xcosn[1]-Xcosn[0], 0)

                        anglemask = anglemask.reshape(nangle.shape)
                        band = self.pyr[bandIdx]
                        curLev = self.spyrHt() - (idx-1)
                        if curLev in levs and b in bands:
                            banddft = np.fft.fftshift(np.fft.fft2(band))
                        else:
                            banddft = np.zeros(band.shape)
                        resdft += ( (np.power(-1+0j,0.5))**(nbands-1) *
                                    banddft * anglemask * himask )
                    bandIdx += 1

        # apply lo0mask
        Xrcos += np.log2(2.0)
        lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)

        lo0mask = lo0mask.reshape(dims[0], dims[1])
        resdft = resdft * lo0mask

        # residual highpass subband
        hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)

        hi0mask = hi0mask.reshape(resdft.shape[0], resdft.shape[1])
        if 0 in levs:
            hidft = np.fft.fftshift(np.fft.fft2(self.pyr[0]))
        else:
            hidft = np.zeros(self.pyr[0].shape)
        resdft += hidft * hi0mask

        outresdft = np.real(np.fft.ifft2(np.fft.ifftshift(resdft)))

        return outresdft

    reconPyr = _reconSFpyr
