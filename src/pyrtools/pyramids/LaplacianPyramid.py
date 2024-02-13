import numpy as np
from .GaussianPyramid import GaussianPyramid
from .filters import parse_filter
from .c.wrapper import upConv


class LaplacianPyramid(GaussianPyramid):
    """Laplacian pyramid

    Parameters
    ----------
    image : `array_like`
        1d or 2d image upon which to construct to the pyramid.
    height : 'auto' or `int`.
        The height of the pyramid. If 'auto', will automatically determine based on the size of
        `image`.
    downsample_filter_name : {'binomN', 'haar', 'qmf8', 'qmf12', 'qmf16', 'daub2', 'daub3',
                              'daub4', 'qmf5', 'qmf9', 'qmf13'}
        name of filter to use for (separable) convolution to downsample the image. All scaled so
        L-2 norm is 1.0

        * `'binomN'` (default: 'binom5') - binomial coefficient filter of order N-1
        * `'haar'` - Haar wavelet
        * `'qmf8'`, `'qmf12'`, `'qmf16'` - Symmetric Quadrature Mirror Filters [1]_
        * `'daub2'`, `'daub3'`, `'daub4'` - Daubechies wavelet [2]_
        * `'qmf5'`, `'qmf9'`, `'qmf13'`   - Symmetric Quadrature Mirror Filters [3]_, [4]_
    upsample_filter_name : {None, 'binomN', 'haar', 'qmf8', 'qmf12', 'qmf16', 'daub2', 'daub3',
                            'daub4', 'qmf5', 'qmf9', 'qmf13'}
        name of filter to use as the "expansion" filter. All scaled so L-2 norm is 1.0

        * None (default) - same as `downsample_filter_name`
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
    def __init__(self, image, height='auto', downsample_filter_name='binom5',
                 upsample_filter_name=None, edge_type='reflect1'):
        self.pyr_type = 'Laplacian'
        if upsample_filter_name is None:
            upsample_filter_name = downsample_filter_name
        super().__init__(image, height, downsample_filter_name, edge_type, upsample_filter_name=upsample_filter_name)


    def _build_pyr(self):
        """build the pyramid

        This should not be called directly by users, it's a helper function for constructing the
        pyramid

        """
        im = self.image
        for lev in range(self.num_scales - 1):
            im_next = self._build_next(im)
            im_recon = self._recon_prev(im_next, output_size=im.shape)
            im_residual = im - im_recon
            self.pyr_coeffs[(lev, 0)] = im_residual.copy()
            self.pyr_size[(lev, 0)] = im_residual.shape
            im = im_next
        self.pyr_coeffs[(lev+1, 0)] = im.copy()
        self.pyr_size[(lev+1, 0)] = im.shape


    def _recon_prev(self, image, output_size, upsample_filter=None, edge_type=None):
        """Reconstruct the previous level of the pyramid.

        Should not be called by users directly, this is a helper function for reconstructing the
        input image using pyramid coefficients.

        """
        if upsample_filter is None:
            upsample_filter = self.filters['upsample_filter']
        else:
            upsample_filter = parse_filter(upsample_filter, normalize=False)

        if edge_type is None:
            edge_type = self.edge_type

        if image.shape[0] == 1:
            res = upConv(image=image, filt=upsample_filter.T, edge_type=edge_type, step=(1, 2), stop=(output_size[0], output_size[1]))
        elif image.shape[1] == 1:
            res = upConv(image=image, filt=upsample_filter, edge_type=edge_type, step=(2, 1), stop=(output_size[0], output_size[1]))
        else:
            tmp = upConv(image=image, filt=upsample_filter, edge_type=edge_type, step=(2, 1), stop=(output_size[0], image.shape[1]))
            res = upConv(image=tmp, filt=upsample_filter.T, edge_type=edge_type, step=(1, 2), stop=(output_size[0], output_size[1]))
        return res


    def recon_pyr(self, upsample_filter_name=None, edge_type=None, levels='all'):
        """Reconstruct the input image using pyramid coefficients

        Parameters
        ----------
        upsample_filter_name : {None, 'binomN', 'haar', 'qmf8', 'qmf12', 'qmf16', 'daub2', 'daub3',
                                'daub4', 'qmf5', 'qmf9', 'qmf13'}
            name of filter to use as "expansion" filter. All scaled so L-2 norm is 1.0

            * None (default) - use `self.upsample_filter_name`, the expansion filter set during
                               initialization.
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
            * `'extend'` - reflect and invert
            * `'dont-compute'` - zero output when filter overhangs imput boundaries.
        levels : `list`, `int`,  or {`'all'`, `'residual_highpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_lowpass'`. If `'all'`, returned value will contain all
            valid levels. Otherwise, must be one of the valid levels.

        Returns
        -------
        recon : `np.array`
            The reconstructed image.
        """
        recon_keys = self._recon_keys(levels, 'all')
        recon = np.zeros_like(self.pyr_coeffs[(self.num_scales-1, 0)])
        for lev in reversed(range(self.num_scales)):
            # upsample to generate higher reconolution image
            recon = self._recon_prev(recon, self.pyr_size[(lev, 0)], upsample_filter_name, edge_type)
            if (lev, 0) in recon_keys:
                recon += self.pyr_coeffs[(lev, 0)]
        return recon
