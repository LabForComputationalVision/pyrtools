from .pyramid import Pyramid
from .filters import parse_filter
from .c.wrapper import corrDn


class GaussianPyramid(Pyramid):
    """Gaussian pyramid

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

    def __init__(self, image, height='auto', filter_name='binom5', edge_type='reflect1', **kwargs):
        super().__init__(image=image, edge_type=edge_type)
        if self.pyr_type is None:
            self.pyr_type = 'Gaussian'
        self.num_orientations = 1

        self.filters = {'downsample_filter': parse_filter(filter_name, normalize=False)}
        upsamp_filt = kwargs.pop('upsample_filter_name', None)
        if upsamp_filt is not None:
            if self.pyr_type != 'Laplacian':
                raise Exception("upsample_filter should only be set for Laplacian pyramid!")
            self.filters['upsample_filter'] = parse_filter(upsamp_filt, normalize=False)
        self._set_num_scales('downsample_filter', height, 1)

        self._build_pyr()

    def _build_next(self, image):
        """build the next level of the pyramid

        This should not be called directly by users, it's a helper function for constructing the
        pyramid

        """
        if image.shape[0] == 1:
            res = corrDn(image=image, filt=self.filters['downsample_filter'].T, edge_type=self.edge_type, step=(1, 2))
        elif image.shape[1] == 1:
            res = corrDn(image=image, filt=self.filters['downsample_filter'], edge_type=self.edge_type, step=(2, 1))
        else:
            tmp = corrDn(image=image, filt=self.filters['downsample_filter'].T, edge_type=self.edge_type, step=(1, 2))
            res = corrDn(image=tmp, filt=self.filters['downsample_filter'], edge_type=self.edge_type, step=(2, 1))
        return res

    def _build_pyr(self):
        """build the pyramid

        This should not be called directly by users, it's a helper function for constructing the
        pyramid

        we do this in a separate method for a bit of class wizardry: by over-writing this method in
        the LaplacianPyramid class, which inherits the GaussianPyramid class, we can still
        correctly construct the LaplacianPyramid with a single call to the GaussianPyramid
        constructor
        """
        im = self.image
        self.pyr_coeffs[(0, 0)] = self.image.copy()
        self.pyr_size[(0, 0)] = self.image_size
        for lev in range(1, self.num_scales):
            im = self._build_next(im)
            self.pyr_coeffs[(lev, 0)] = im.copy()
            self.pyr_size[(lev, 0)] = im.shape

    def recon_pyr(self, *args):
        """Reconstruct the pyramid -- NOT NECESSARY FOR GAUSSIANS
        """
        raise Exception('Not necessary for Gaussian Pyramids')
