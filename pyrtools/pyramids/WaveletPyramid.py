import numpy as np
from .pyramid import Pyramid
from .c.wrapper import corrDn, upConv


class WaveletPyramid(Pyramid):

    def __init__(self, image, height='auto', filters='qmf9', edge_type='reflect1'):

        super().__init__(image=image, edge_type=edge_type)
        self.pyr_type = 'Wavelet'
        self.pyr_coeffs = {}
        self.pyr_size = {}

        self.lo_filter = self._parse_filter(filters)
        # if the image is 1D, parseFilter will
        # match the filter to the image dimensions
        self.hi_filter = self._modulate_flip(self.lo_filter)
        # modulate_flip returns a filter that has
        # the same size as its input filter
        assert self.lo_filter.shape == self.hi_filter.shape

        # Stagger sampling if filter is odd-length
        self.stagger = (self.lo_filter.size + 1) % 2

        max_ht = self.maxPyrHt(self.image.shape, self.lo_filter.shape)
        if height == 'auto':
            self.num_scales = max_ht
        elif height > max_ht:
            raise Exception("cannot build pyramid higher than %d levels." % (max_ht))
        else:
            self.num_scales = int(height)

        # compute the number of channels per level
        if min(self.image.shape) == 1:
            self.num_orientations = 1
        else:
            self.num_orientations = 3

        im = self.image
        for i, lev in enumerate(range(self.num_scales)):
            im, higher_bands = self._build_next(im)
            for j, band in enumerate(higher_bands):
                self.pyr_coeffs[(i, j)] = band
                self.pyr_size[(i, j)] = band.shape
        self.pyr_coeffs['residual_lowpass'] = im
        self.pyr_size['residual_lowpass'] = im.shape

    def _modulate_flip(self, lo_filter):
        '''construct QMF/Wavelet highpass filter from lowpass filter

        modulate by (-1)^n, reverse order (and shift by one, which is handled by the convolution
        routines).  This is an extension of the original definition of QMF's (e.g., see
        Simoncelli90).
        '''
        # check lo_filter is effectively 1D
        lo_filter_shape = lo_filter.shape
        assert lo_filter.size == max(lo_filter_shape)
        lo_filter = lo_filter.flatten()
        ind = np.arange(lo_filter.size, 0, -1) - (lo_filter.size + 1) // 2
        hi_filter = lo_filter[::-1] * (-1.0) ** ind

        # OLD: matlab version always returns a column vector
        # NOW: same shape as input
        return hi_filter.reshape(lo_filter_shape)

    def _build_next(self, image):
        if image.shape[1] == 1:
            lolo = corrDn(image=image, filt=self.lo_filter, edges=self.edge_type, step=(2, 1),
                          start=(self.stagger, 0))
            hihi = corrDn(image=image, filt=self.hi_filter, edges=self.edge_type, step=(2, 1),
                          start=(1, 0))
            return lolo, (hihi, )
        elif image.shape[0] == 1:
            lolo = corrDn(image=image, filt=self.lo_filter, edges=self.edge_type, step=(1, 2),
                          start=(0, self.stagger))
            hihi = corrDn(image=image, filt=self.hi_filter, edges=self.edge_type, step=(1, 2),
                          start=(0, 1))
            return lolo, (hihi, )
        else:
            lo = corrDn(image=image, filt=self.lo_filter, edges=self.edge_type, step=(2, 1),
                        start=(self.stagger, 0))
            hi = corrDn(image=image, filt=self.hi_filter, edges=self.edge_type, step=(2, 1),
                        start=(1, 0))
            lolo = corrDn(image=lo, filt=self.lo_filter.T, edges=self.edge_type, step=(1, 2),
                          start=(0, self.stagger))
            lohi = corrDn(image=hi, filt=self.lo_filter.T, edges=self.edge_type, step=(1, 2),
                          start=(0, self.stagger))
            hilo = corrDn(image=lo, filt=self.hi_filter.T, edges=self.edge_type, step=(1, 2),
                          start=(0, 1))
            hihi = corrDn(image=hi, filt=self.hi_filter.T, edges=self.edge_type, step=(1, 2),
                          start=(0, 1))
            return lolo, (lohi, hilo, hihi)

    def _recon_prev(self, image, lev, recon_keys, output_size, lo_filter, hi_filter, edges,
                    stagger):

        if self.num_orientations == 1:
            print(output_size)
            if output_size[0] == 1:
                recon = upConv(image=image, filt=lo_filter, edges=edges,
                               step=(1, 2), start=(0, stagger), stop=output_size)
                if (lev, 0) in recon_keys:
                    recon += upConv(image=self.pyr_coeffs[(lev, 0)], filt=hi_filter, edges=edges,
                                    step=(1, 2), start=(0, 1), stop=output_size)
            elif output_size[1] == 1:
                recon = upConv(image=image, filt=lo_filter, edges=edges,
                               step=(2, 1), start=(stagger, 0), stop=output_size)
                if (lev, 0) in recon_keys:
                    recon += upConv(image=self.pyr_coeffs[(lev, 0)], filt=hi_filter, edges=edges,
                                    step=(2, 1), start=(1, 0), stop=output_size)
        else:
            lo_size = ([self.pyr_size[(lev, 1)][0], output_size[1]])
            hi_size = ([self.pyr_size[(lev, 0)][0], output_size[1]])

            tmp_recon = upConv(image=image, filt=lo_filter.T, edges=edges,
                               step=(1, 2), start=(0, stagger), stop=lo_size)
            recon = upConv(image=tmp_recon, filt=lo_filter, edges=edges,
                           step=(2, 1), start=(stagger, 0), stop=output_size)

            bands_recon_dict = {
                0: [{'filt': lo_filter.T, 'start': (0, stagger), 'stop': hi_size},
                    {'filt': hi_filter, 'start': (1, 0)}],
                1: [{'filt': hi_filter.T, 'start': (0, 1), 'stop': lo_size},
                    {'filt': lo_filter, 'start': (stagger, 0)}],
                2: [{'filt': hi_filter.T, 'start': (0, 1), 'stop': hi_size},
                    {'filt': hi_filter, 'start': (1, 0)}],
            }

            for band in range(self.num_orientations):
                if (lev, band) in recon_keys:
                    tmp_recon = upConv(image=self.pyr_coeffs[(lev, band)], edges=edges,
                                       step=(1, 2), **bands_recon_dict[band][0])
                    recon += upConv(image=tmp_recon, edges=edges, step=(2, 1), stop=output_size,
                                    **bands_recon_dict[band][1])

        return recon

    def recon_pyr(self, filt=None, edge_type=None, levels='all', bands='all'):
        # Optional args

        if filt is None:
            lo_filter = self.lo_filter
            hi_filter = self.hi_filter
            stagger = self.stagger
        else:
            lo_filter = self._parse_filter(filt)
            hi_filter = self._modulate_flip(lo_filter)
            stagger = (lo_filter.size + 1) % 2

        if edge_type is None:
            edges = self.edge_type
        else:
            edges = edge_type

        recon_keys = self._recon_keys(levels, bands)

        # initialize reconstruction
        if 'residual_lowpass' in recon_keys:
            recon = self.pyr_coeffs['residual_lowpass']
        else:
            recon = np.zeros(self.pyr_coeffs['residual_lowpass'].shape)

        for lev in reversed(range(self.num_scales)):
            if self.num_orientations == 1:
                print(lev)
                if lev == 0:
                    output_size = self.image.shape
                else:
                    output_size = self.pyr_size[(lev-1, 0)]
            else:
                output_size = (self.pyr_size[(lev, 0)][0] + self.pyr_size[(lev, 1)][0],
                               self.pyr_size[(lev, 0)][1] + self.pyr_size[(lev, 1)][1])
            recon = self._recon_prev(recon, lev, recon_keys, output_size, lo_filter,
                                     hi_filter, edges, stagger)

        return recon
