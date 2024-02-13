import numpy as np
from .pyramid import Pyramid
from .filters import parse_filter
from .c.wrapper import corrDn, upConv


class WaveletPyramid(Pyramid):
    """Multiscale wavelet pyramid

    Parameters
    ----------
    image : `array_like`
        1d or 2d image upon which to construct to the pyramid.
    height : 'auto' or `int`.
        The height of the pyramid. If 'auto', will automatically determine based on the size of
        `image`.
    filter_name : {'binomN', 'haar', 'qmf8', 'qmf12', 'qmf16', 'daub2', 'daub3', 'daub4', 'qmf5',
                   'qmf9', 'qmf13'}
        name of filter to use when constructing pyramid. All scaled so L-2 norm is 1.0

        * `'binomN'` - binomial coefficient filter of order N-1
        * `'haar'` - Haar wavelet
        * `'qmf8'`, `'qmf12'`, `'qmf16'` - Symmetric Quadrature Mirror Filters [1]_
        * `'daub2'`, `'daub3'`, `'daub4'` - Daubechies wavelet [2]_
        * `'qmf5'`, `'qmf9'`, `'qmf13'`   - Symmetric Quadrature Mirror Filters [3]_, [4]_
    edge_type : {'circular', 'reflect1', 'reflect2', 'repeat', 'zero', 'extend', 'dont-compute'}
        Specifies how to handle edges. Options are:

        * `'circular'` - circular convolution
        * `'reflect1'` - reflect about the edge pixels
        * `'reflect2'` - reflect, doubling the edge pixels
        * `'repeat'` - repeat the edge pixels
        * `'zero'` - assume values of zero outside image boundary
        * `'extend'` - reflect and invert
        * `'dont-compute'` - zero output when filter overhangs imput boundaries.

    Attributes
    ----------
    image : `array_like`
        The input image used to construct the pyramid.
    image_size : `tuple`
        The size of the input image.
    pyr_type : `str` or `None`
        Human-readable string specifying the type of pyramid. For base class, is None.
    edge_type : `str`
        Specifies how edges were handled.
    pyr_coeffs : `dict`
        Dictionary containing the coefficients of the pyramid. Keys are `(level, band)` tuples and
        values are 1d or 2d numpy arrays (same number of dimensions as the input image)
    pyr_size : `dict`
        Dictionary containing the sizes of the pyramid coefficients. Keys are `(level, band)`
        tuples and values are tuples.
    is_complex : `bool`
        Whether the coefficients are complex- or real-valued. Only `SteerablePyramidFreq` can have
        a value of True, all others must be False.

    References
    ----------
    .. [1] J D Johnston, "A filter family designed for use in quadrature mirror filter banks",
       Proc. ICASSP, pp 291-294, 1980.
    .. [2] I Daubechies, "Orthonormal bases of compactly supported wavelets", Commun. Pure Appl.
       Math, vol. 42, pp 909-996, 1988.
    .. [3] E P Simoncelli,  "Orthogonal sub-band image transforms", PhD Thesis, MIT Dept. of Elec.
       Eng. and Comp. Sci. May 1988. Also available as: MIT Media Laboratory Vision and Modeling
       Technical Report #100.
    .. [4] E P Simoncelli and E H Adelson, "Subband image coding", Subband Transforms, chapter 4,
       ed. John W Woods, Kluwer Academic Publishers,  Norwell, MA, 1990, pp 143--192.
    """

    def __init__(self, image, height='auto', filter_name='qmf9', edge_type='reflect1'):
        super().__init__(image=image, edge_type=edge_type)
        self.pyr_type = 'Wavelet'

        self.filters = {}
        self.filters['lo_filter'] = parse_filter(filter_name, normalize=False)
        self.filters["hi_filter"] = WaveletPyramid._modulate_flip(self.filters['lo_filter'])
        assert self.filters['lo_filter'].shape == self.filters['hi_filter'].shape

        # Stagger sampling if filter is odd-length
        self.stagger = (self.filters['lo_filter'].size + 1) % 2

        self._set_num_scales('lo_filter', height)

        # compute the number of channels per level
        if min(self.image.shape) == 1:
            self.num_orientations = 1
        else:
            self.num_orientations = 3

        self._build_pyr()

    def _modulate_flip(lo_filter):
        '''construct QMF/Wavelet highpass filter from lowpass filter

        modulate by (-1)^n, reverse order (and shift by one, which is handled by the convolution
        routines).  This is an extension of the original definition of QMF's (e.g., see
        Simoncelli90).

        Parameters
        ----------
        lo_filter : `array_like`
            one-dimensional array (or effectively 1d array) containing the lowpass filter to
            convert into the highpass filter.

        Returns
        -------
        hi_filter : `np.array`
            The highpass filter constructed from the lowpass filter, same shape as the lowpass
            filter.
        '''
        # check lo_filter is effectively 1D
        lo_filter_shape = lo_filter.shape
        assert lo_filter.size == max(lo_filter_shape)
        lo_filter = lo_filter.flatten()
        ind = np.arange(lo_filter.size, 0, -1) - (lo_filter.size + 1) // 2
        hi_filter = lo_filter[::-1] * (-1.0) ** ind

        return hi_filter.reshape(lo_filter_shape)

    def _build_next(self, image):
        """Build the next level fo the Wavelet pyramid

        Should not be called by users directly, this is a helper function to construct the pyramid.

        Parameters
        ----------
        image : `array_like`
            image to use to construct next level.

        Returns
        -------
        lolo : `array_like`
            This is the result of applying the lowpass filter once if `image` is 1d, twice if it's
            2d. It's downsampled by a factor of two from the original `image`.
        hi_tuple : `tuple`
            If `image` is 1d, this just contains `hihi`, the result of applying the highpass filter
            . If `image` is 2d, it is `(lohi, hilo, hihi)`, the result of applying the lowpass then
            the highpass, the highpass then the lowpass, and the highpass twice. All will be
            downsampled by a factor of two from the original `image`.
        """
        if image.shape[1] == 1:
            lolo = corrDn(image=image, filt=self.filters['lo_filter'], edge_type=self.edge_type, step=(2, 1), start=(self.stagger, 0))
            hihi = corrDn(image=image, filt=self.filters['hi_filter'], edge_type=self.edge_type, step=(2, 1), start=(1, 0))
            return lolo, (hihi, )
        elif image.shape[0] == 1:
            lolo = corrDn(image=image, filt=self.filters['lo_filter'].T, edge_type=self.edge_type, step=(1, 2), start=(0, self.stagger))
            hihi = corrDn(image=image, filt=self.filters['hi_filter'].T, edge_type=self.edge_type, step=(1, 2), start=(0, 1))
            return lolo, (hihi, )
        else:
            lo = corrDn(image=image, filt=self.filters['lo_filter'], edge_type=self.edge_type, step=(2, 1), start=(self.stagger, 0))
            hi = corrDn(image=image, filt=self.filters['hi_filter'], edge_type=self.edge_type, step=(2, 1), start=(1, 0))
            lolo = corrDn(image=lo, filt=self.filters['lo_filter'].T, edge_type=self.edge_type, step=(1, 2), start=(0, self.stagger))
            lohi = corrDn(image=hi, filt=self.filters['lo_filter'].T, edge_type=self.edge_type, step=(1, 2), start=(0, self.stagger))
            hilo = corrDn(image=lo, filt=self.filters['hi_filter'].T, edge_type=self.edge_type, step=(1, 2), start=(0, 1))
            hihi = corrDn(image=hi, filt=self.filters['hi_filter'].T, edge_type=self.edge_type, step=(1, 2), start=(0, 1))
            return lolo, (lohi, hilo, hihi)

    def _build_pyr(self):
        im = self.image
        for lev in range(self.num_scales):
            im, higher_bands = self._build_next(im)
            for j, band in enumerate(higher_bands):
                self.pyr_coeffs[(lev, j)] = band
                self.pyr_size[(lev, j)] = band.shape
        self.pyr_coeffs['residual_lowpass'] = im
        self.pyr_size['residual_lowpass'] = im.shape


    def _recon_prev(self, image, lev, recon_keys, output_size, lo_filter, hi_filter, edge_type,
                    stagger):
        """Reconstruct the previous level of the pyramid.

        Should not be called by users directly, this is a helper function for reconstructing the
        input image using pyramid coefficients.

        """
        if self.num_orientations == 1:
            if output_size[0] == 1:
                recon = upConv(image=image, filt=lo_filter.T, edge_type=edge_type, step=(1, 2), start=(0, stagger), stop=output_size)
                if (lev, 0) in recon_keys:
                    recon += upConv(image=self.pyr_coeffs[(lev, 0)], filt=hi_filter.T, edge_type=edge_type, step=(1, 2), start=(0, 1), stop=output_size)
            elif output_size[1] == 1:
                recon = upConv(image=image, filt=lo_filter, edge_type=edge_type, step=(2, 1), start=(stagger, 0), stop=output_size)
                if (lev, 0) in recon_keys:
                    recon += upConv(image=self.pyr_coeffs[(lev, 0)], filt=hi_filter, edge_type=edge_type, step=(2, 1), start=(1, 0), stop=output_size)
        else:
            lo_size = ([self.pyr_size[(lev, 1)][0], output_size[1]])
            hi_size = ([self.pyr_size[(lev, 0)][0], output_size[1]])

            tmp_recon = upConv(image=image, filt=lo_filter.T, edge_type=edge_type, step=(1, 2), start=(0, stagger), stop=lo_size)
            recon = upConv(image=tmp_recon, filt=lo_filter, edge_type=edge_type, step=(2, 1), start=(stagger, 0), stop=output_size)

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
                    tmp_recon = upConv(image=self.pyr_coeffs[(lev, band)], edge_type=edge_type, step=(1, 2), **bands_recon_dict[band][0])
                    recon += upConv(image=tmp_recon, edge_type=edge_type, step=(2, 1), stop=output_size, **bands_recon_dict[band][1])

        return recon

    def recon_pyr(self, filter_name=None, edge_type=None, levels='all', bands='all'):
        """Reconstruct the input image using pyramid coefficients.

        This function reconstructs the input image using pyramid coefficients.

        Parameters
        ----------
        filter_name : {None, 'binomN', 'haar', 'qmf8', 'qmf12', 'qmf16', 'daub2', 'daub3', 'daub4',
                       'qmf5', 'qmf9', 'qmf13'}
            name of filter to use for reconstruction. All scaled so L-2 norm is 1.0

            * None (default) - use `self.filter_name`, the filter used to construct the pyramid.
            * `'binomN'` - binomial coefficient filter of order N-1
            * `'haar'` - Haar wavelet
            * `'qmf8'`, `'qmf12'`, `'qmf16'` - Symmetric Quadrature Mirror Filters [1]_
            * `'daub2'`, `'daub3'`, `'daub4'` - Daubechies wavelet [2]_
            * `'qmf5'`, `'qmf9'`, `'qmf13'`   - Symmetric Quadrature Mirror Filters [3]_, [4]_
        edge_type : {None, 'circular', 'reflect1', 'reflect2', 'repeat', 'zero', 'extend',
                     'dont-compute'}
            Specifies how to handle edges. Options are:

            * None (default) - use `self.edge_type`, the edge_type used to construct the pyramid
            * `'circular'` - circular convolution
            * `'reflect1'` - reflect about the edge pixels
            * `'reflect2'` - reflect, doubling the edge pixels
            * `'repeat'` - repeat the edge pixels
            * `'zero'` - assume values of zero outside image boundary
            * `'extend'` - reflect and inverts
            * `'dont-compute'` - zero output when filter overhangs imput boundaries.
        levels : `list`, `int`,  or {`'all'`, `'residual_highpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_lowpass'`. If `'all'`, returned value will contain all
            valid levels. Otherwise, must be one of the valid levels.
        bands : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations.

        Returns
        -------
        recon : `np.array`
            The reconstructed image.
        """
        # Optional args

        if filter_name is None:
            lo_filter = self.filters['lo_filter']
            hi_filter = self.filters['hi_filter']
            stagger = self.stagger
        else:
            lo_filter = parse_filter(filter_name, normalize=False)
            hi_filter = WaveletPyramid._modulate_flip(lo_filter)
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
            recon = np.zeros_like(self.pyr_coeffs['residual_lowpass'])

        for lev in reversed(range(self.num_scales)):
            if self.num_orientations == 1:
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
