import warnings
import numpy as np
from scipy.special import factorial
from .pyramid import Pyramid
from .c.wrapper import pointOp
from .steer import steer_to_harmonics_mtx
from ..tools.utils import rcosFn


class SteerablePyramidFreq(Pyramid):

    def __init__(self, image, height='auto', num_orientations=4, twidth=1, is_complex=False):
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

        - `num_orientations` (optional), int in range [1, 16]. Default
          value is 4. The number of orientation bands. This is the
          order of the steerable pyramid + 1.

        - `twidth` (optional), int. Default value is 1.
        The width of the transition region of the radial lowpass function, in octaves

        - `is_complex` (optional), boolean. Default False. Whether the pyramid coefficients should
        be complex or not. If True, the real and imaginary parts correspond to a pair of even and
        odd symmetric filters. If False, the coefficients only include the real part / even
        symmetric filter.

        """
        # in the Fourier domain, there's only one choice for how do edge-handling: circular. to
        # emphasize that thisisn'ta choice, we use None here.
        super().__init__(image=image, edge_type=None)

        self.pyr_type = 'SteerableFrequency'
        self.is_complex = is_complex
        # SteerablePyramidFreq doesn't have filters, they're constructed in the frequency space
        self.filters = {}

        # we can't use the base class's _set_num_scales method because the max height is calculated
        # slightly differently
        max_ht = np.floor(np.log2(min(self.image.shape))) - 2
        if height == 'auto':
            self.num_scales = int(max_ht)
        elif height > max_ht:
            raise Exception("Cannot build pyramid higher than %d levels." % (max_ht))
        else:
            self.num_scales = int(height)

        if num_orientations > 16 or num_orientations < 1:
            warnings.warn("num_orientations must be an integer in the range [1,16]. Truncating.")
            num_orientations = min(max(num_orientations, 1), 16)
        self.num_orientations = int(num_orientations)

        if twidth <= 0:
            warnings.warn("twidth must be positive. Setting to 1.")
            twidth = 1
        twidth = int(twidth)

        # ------------------------------------------------------
        # steering stuff:

        if self.num_orientations % 2 == 0:
            harmonics = np.arange(self.num_orientations // 2) * 2 + 1
        else:
            harmonics = np.arange((self.num_orientations-1) // 2) * 2
        if harmonics.size == 0:
            # in this case, harmonics is an empty matrix. This happens when
            # self.num_orientations=0 and (based on how the matlab code acts), in that situation,
            # we actually want harmonics to be 0.
            harmonics = np.array([0])
        self.harmonics = harmonics

        angles = np.pi * np.arange(self.num_orientations)/self.num_orientations
        self.steermtx = steer_to_harmonics_mtx(harmonics, angles, even_phase=True)

        # ------------------------------------------------------

        dims = np.array(self.image.shape)
        ctr = np.ceil((np.array(dims)+0.5)/2).astype(int)

        (xramp, yramp) = np.meshgrid(np.linspace(-1, 1, dims[1]+1)[:-1],
                                     np.linspace(-1, 1, dims[0]+1)[:-1])

        angle = np.arctan2(yramp, xramp)
        log_rad = np.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        log_rad = np.log2(log_rad)

        # Radial transition function (a raised cosine in log-frequency):
        (Xrcos, Yrcos) = rcosFn(twidth, (-twidth/2.0), np.array([0, 1]))
        Yrcos = np.sqrt(Yrcos)

        YIrcos = np.sqrt(1.0 - Yrcos**2)
        lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)
        self._lo0mask = lo0mask

        imdft = np.fft.fftshift(np.fft.fft2(self.image))

        hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)
        self._hi0mask = hi0mask

        hi0dft = imdft * hi0mask.reshape(imdft.shape[0], imdft.shape[1])
        hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))

        self.pyr_coeffs['residual_highpass'] = np.real(hi0)
        self.pyr_size['residual_highpass'] = hi0.shape

        lo0mask = lo0mask.reshape(imdft.shape[0], imdft.shape[1])
        lodft = imdft * lo0mask

        self._anglemasks = []
        self._himasks = []
        self._lomasks = []

        for i in range(self.num_scales):
            Xrcos -= np.log2(2)

            lutsize = 1024
            Xcosn = np.pi * np.arange(-(2*lutsize+1), (lutsize+2)) / lutsize

            order = self.num_orientations - 1
            const = (2**(2*order))*(factorial(order, exact=True)**2)/ float(self.num_orientations*factorial(2*order, exact=True))

            if self.is_complex:
                # TODO clean that up and give comments
                alfa = ((np.pi+Xcosn) % (2.0*np.pi)) - np.pi
                Ycosn = (2.0 * np.sqrt(const) * (np.cos(Xcosn) ** order) *
                         (np.abs(alfa) < np.pi/2.0).astype(int))
            else:
                Ycosn = np.sqrt(const) * (np.cos(Xcosn))**order

            log_rad_test = np.reshape(log_rad, (1, log_rad.shape[0] * log_rad.shape[1]))
            himask = pointOp(log_rad_test, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)
            himask = himask.reshape((lodft.shape[0], lodft.shape[1]))
            self._himasks.append(himask)

            anglemasks = []
            for b in range(self.num_orientations):
                angle_tmp = np.reshape(angle, (1, angle.shape[0] * angle.shape[1]))
                anglemask = pointOp(angle_tmp, Ycosn, Xcosn[0]+np.pi*b/self.num_orientations,
                                    Xcosn[1]-Xcosn[0], 0)

                anglemask = anglemask.reshape(lodft.shape[0], lodft.shape[1])
                anglemasks.append(anglemask)
                # that (-1j)**order term in the beginning will be 1, -j, -1, j for order 0, 1, 2,
                # 3, and will then loop again
                banddft = (-1j) ** order * lodft * anglemask * himask
                band = np.fft.ifft2(np.fft.ifftshift(banddft))
                if not self.is_complex:
                    self.pyr_coeffs[(i, b)] = np.real(band.copy())
                else:
                    self.pyr_coeffs[(i, b)] = band.copy()
                self.pyr_size[(i, b)] = band.shape

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
            log_rad_tmp = np.reshape(log_rad, (1, log_rad.shape[0] * log_rad.shape[1]))
            lomask = pointOp(log_rad_tmp, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)
            lomask = lomask.reshape(lodft.shape[0], lodft.shape[1])
            self._lomasks.append(lomask)

            lodft = lodft * lomask

        lodft = np.fft.ifft2(np.fft.ifftshift(lodft))
        self.pyr_coeffs['residual_lowpass'] = np.real(np.array(lodft).copy())
        self.pyr_size['residual_lowpass'] = lodft.shape

    def recon_pyr(self, levels='all', bands='all', twidth=1):

        if twidth <= 0:
            warnings.warn("twidth must be positive. Setting to 1.")
            twidth = 1

        recon_keys = self._recon_keys(levels, bands)

        # make list of dims and bounds
        bound_list = []
        dim_list = []
        # we go through pyr_sizes from smallest to largest
        for dims in sorted(self.pyr_size.values()):
            if dims in dim_list:
                continue
            dim_list.append(dims)
            dims = np.array(dims)
            ctr = np.ceil((dims+0.5)/2).astype(int)
            lodims = np.ceil((dims-0.5)/2).astype(int)
            loctr = np.ceil((lodims+0.5)/2).astype(int)
            lostart = ctr - loctr
            loend = lostart + lodims
            bounds = (lostart[0], lostart[1], loend[0], loend[1])
            bound_list.append(bounds)
        bound_list.append((0, 0, dim_list[-1][0], dim_list[-1][1]))
        dim_list.append((dim_list[-1][0], dim_list[-1][1]))

        # matlab code starts here
        dims = np.array(self.pyr_size['residual_highpass'])
        ctr = np.ceil((dims+0.5)/2.0).astype(int)

        (xramp, yramp) = np.meshgrid((np.arange(1, dims[1]+1)-ctr[1]) / (dims[1]/2.),
                                     (np.arange(1, dims[0]+1)-ctr[0]) / (dims[0]/2.))
        angle = np.arctan2(yramp, xramp)
        log_rad = np.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        log_rad = np.log2(log_rad)

        # Radial transition function (a raised cosine in log-frequency):
        (Xrcos, Yrcos) = rcosFn(twidth, (-twidth/2.0), np.array([0, 1]))
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(1.0 - Yrcos**2)

        # from reconSFpyrLevs
        lutsize = 1024

        Xcosn = np.pi * np.arange(-(2*lutsize+1), (lutsize+2)) / lutsize

        order = self.num_orientations - 1
        const = (2**(2*order))*(factorial(order, exact=True)**2) / float(self.num_orientations*factorial(2*order, exact=True))
        Ycosn = np.sqrt(const) * (np.cos(Xcosn))**order

        # lowest band
        # initialize reconstruction
        if 'residual_lowpass' in recon_keys:
            nresdft = np.fft.fftshift(np.fft.fft2(self.pyr_coeffs['residual_lowpass']))
        else:
            nresdft = np.zeros_like(self.pyr_coeffs['residual_lowpass'])
        resdft = np.zeros(dim_list[1]) + 0j

        bounds = (0, 0, 0, 0)
        for idx in range(len(bound_list)-2, 0, -1):
            diff = (bound_list[idx][2]-bound_list[idx][0],
                    bound_list[idx][3]-bound_list[idx][1])
            bounds = (bounds[0]+bound_list[idx][0], bounds[1]+bound_list[idx][1],
                      bounds[0]+bound_list[idx][0] + diff[0],
                      bounds[1]+bound_list[idx][1] + diff[1])
            Xrcos -= np.log2(2.0)
        nlog_rad = log_rad[bounds[0]:bounds[2], bounds[1]:bounds[3]]

        nlog_rad_tmp = np.reshape(nlog_rad, (1, nlog_rad.shape[0]*nlog_rad.shape[1]))
        lomask = pointOp(nlog_rad_tmp, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)
        lomask = lomask.reshape(nresdft.shape[0], nresdft.shape[1])
        lomask = lomask + 0j
        resdft[bound_list[1][0]:bound_list[1][2],
               bound_list[1][1]:bound_list[1][3]] = nresdft * lomask

        # middle bands
        for idx in range(1, len(bound_list)-1):
            bounds1 = (0, 0, 0, 0)
            bounds2 = (0, 0, 0, 0)
            for boundIdx in range(len(bound_list) - 1, idx - 1, -1):
                diff = (bound_list[boundIdx][2]-bound_list[boundIdx][0],
                        bound_list[boundIdx][3]-bound_list[boundIdx][1])
                bound2tmp = bounds2
                bounds2 = (bounds2[0]+bound_list[boundIdx][0],
                           bounds2[1]+bound_list[boundIdx][1],
                           bounds2[0]+bound_list[boundIdx][0] + diff[0],
                           bounds2[1]+bound_list[boundIdx][1] + diff[1])
                bounds1 = bound2tmp
            nlog_rad1 = log_rad[bounds1[0]:bounds1[2], bounds1[1]:bounds1[3]]
            nlog_rad2 = log_rad[bounds2[0]:bounds2[2], bounds2[1]:bounds2[3]]
            dims = dim_list[idx]
            nangle = angle[bounds1[0]:bounds1[2], bounds1[1]:bounds1[3]]
            YIrcos = np.abs(np.sqrt(1.0 - Yrcos**2))
            if idx > 1:
                Xrcos += np.log2(2.0)
                nlog_rad2_tmp = np.reshape(nlog_rad2, (1, nlog_rad2.shape[0]*nlog_rad2.shape[1]))
                lomask = pointOp(nlog_rad2_tmp, YIrcos, Xrcos[0],
                                 Xrcos[1]-Xrcos[0], 0)
                lomask = lomask.reshape(bounds2[2]-bounds2[0],
                                        bounds2[3]-bounds2[1])
                lomask = lomask + 0j
                nresdft = np.zeros(dim_list[idx]) + 0j
                nresdft[bound_list[idx][0]:bound_list[idx][2],
                        bound_list[idx][1]:bound_list[idx][3]] = resdft * lomask
                resdft = nresdft.copy()

            # reconSFpyrLevs
            if idx != 0 and idx != len(bound_list)-1:
                for b in range(self.num_orientations):
                    nlog_rad1_tmp = np.reshape(nlog_rad1,
                                               (1, nlog_rad1.shape[0]*nlog_rad1.shape[1]))
                    himask = pointOp(nlog_rad1_tmp, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)

                    himask = himask.reshape(nlog_rad1.shape)
                    nangle_tmp = np.reshape(nangle, (1, nangle.shape[0]*nangle.shape[1]))
                    anglemask = pointOp(nangle_tmp, Ycosn,
                                        Xcosn[0]+np.pi*b/self.num_orientations,
                                        Xcosn[1]-Xcosn[0], 0)

                    anglemask = anglemask.reshape(nangle.shape)
                    # either the coefficients will already be real-valued (if
                    # self.is_complex=False) or complex (if self.is_complex=True). in the
                    # former case, this np.real() does nothing. in the latter, we want to only
                    # reconstruct with the real portion
                    curLev = self.num_scales-1 - (idx-1)
                    band = np.real(self.pyr_coeffs[(curLev, b)])
                    if (curLev, b) in recon_keys:
                        banddft = np.fft.fftshift(np.fft.fft2(band))
                    else:
                        banddft = np.zeros(band.shape)
                    resdft += ((np.power(-1+0j, 0.5))**(self.num_orientations-1) *
                               banddft * anglemask * himask)

        # apply lo0mask
        Xrcos += np.log2(2.0)
        lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)

        lo0mask = lo0mask.reshape(dims[0], dims[1])
        resdft = resdft * lo0mask

        # residual highpass subband
        hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)

        hi0mask = hi0mask.reshape(resdft.shape[0], resdft.shape[1])
        if 'residual_highpass' in recon_keys:
            hidft = np.fft.fftshift(np.fft.fft2(self.pyr_coeffs['residual_highpass']))
        else:
            hidft = np.zeros_like(self.pyr_coeffs['residual_highpass'])
        resdft += hidft * hi0mask

        outresdft = np.real(np.fft.ifft2(np.fft.ifftshift(resdft)))

        return outresdft
