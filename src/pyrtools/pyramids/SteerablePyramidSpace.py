import numpy as np
from .pyramid import SteerablePyramidBase
from .filters import parse_filter
from .c.wrapper import corrDn, upConv


class SteerablePyramidSpace(SteerablePyramidBase):
    """Steerable pyramid (using spatial convolutions)

    Notes
    -----
    Transform described in [1]_, filter kernel design described in [2]_.

    Parameters
    ----------
    image : `array_like`
        2d image upon which to construct to the pyramid.
    height : 'auto' or `int`.
        The height of the pyramid. If 'auto', will automatically determine based on the size of
        `image`.
    order : {0, 1, 3, 5}.
        The Gaussian derivative order used for the steerable filters. If you want a different
        value, see SteerablePyramidFreq. Note that to achieve steerability the minimum number
        of orientation is `order` + 1, and is used here. To get more orientations at the same
        order, use the method `steer_coeffs`
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
    .. [1] E P Simoncelli and W T Freeman, "The Steerable Pyramid: A Flexible Architecture for
       Multi-Scale Derivative Computation," Second Int'l Conf on Image Processing, Washington, DC,
       Oct 1995.
    .. [2] A Karasaridis and E P Simoncelli, "A Filter Design Technique for Steerable Pyramid
       Image Transforms", ICASSP, Atlanta, GA, May 1996.
    """

    def __init__(self, image, height='auto', order=1, edge_type='reflect1'):
        super().__init__(image=image, edge_type=edge_type)

        self.order = order
        self.num_orientations = self.order + 1
        self.filters = parse_filter("sp{:d}_filters".format(self.num_orientations-1), normalize=False)
        self.pyr_type = 'SteerableSpace'
        self._set_num_scales('lofilt', height)

        hi0 = corrDn(image=self.image, filt=self.filters['hi0filt'], edge_type=self.edge_type)

        self.pyr_coeffs['residual_highpass'] = hi0
        self.pyr_size['residual_highpass'] = hi0.shape

        lo = corrDn(image=self.image, filt=self.filters['lo0filt'], edge_type=self.edge_type)
        for i in range(self.num_scales):
            # assume square filters  -- start of buildSpyrLevs
            bfiltsz = int(np.floor(np.sqrt(self.filters['bfilts'].shape[0])))

            for b in range(self.num_orientations):
                filt = self.filters['bfilts'][:, b].reshape(bfiltsz, bfiltsz).T
                band = corrDn(image=lo, filt=filt, edge_type=self.edge_type)
                self.pyr_coeffs[(i, b)] = np.asarray(band)
                self.pyr_size[(i, b)] = band.shape

            lo = corrDn(image=lo, filt=self.filters['lofilt'], edge_type=self.edge_type, step=(2, 2))

        self.pyr_coeffs['residual_lowpass'] = lo
        self.pyr_size['residual_lowpass'] = lo.shape

    def recon_pyr(self, order=None, edge_type=None, levels='all', bands='all'):
        """Reconstruct the image, optionally using subset of pyramid coefficients.

        Parameters
        ----------
        order : {None, 0, 1, 3, 5}.
            the Gaussian derivative order you want to use for the steerable pyramid filters used to
            reconstruct the pyramid. If None, uses the same order as that used to construct the
            pyramid.
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

        if order is None:
            filters = self.filters
            recon_keys = self._recon_keys(levels, bands)
        else:
            filters = parse_filter("sp{:d}_filters".format(order), normalize=False)
            recon_keys = self._recon_keys(levels, bands, order+1)

        # assume square filters  -- start of buildSpyrLevs
        bfiltsz = int(np.floor(np.sqrt(filters['bfilts'].shape[0])))

        if edge_type is None:
            edges = self.edge_type
        else:
            edges = edge_type


        # initialize reconstruction
        if 'residual_lowpass' in recon_keys:
            recon = self.pyr_coeffs['residual_lowpass']
        else:
            recon = np.zeros_like(self.pyr_coeffs['residual_lowpass'])

        for lev in reversed(range(self.num_scales)):
            # we need to upConv once per level, in order to up-sample
            # the image back to the right shape.
            recon = upConv(image=recon, filt=filters['lofilt'], edge_type=edges,
                           step=(2, 2), start=(0, 0), stop=self.pyr_size[(lev, 0)])
            # I think the most effective way to do this is to just
            # check every possible sub-band and then only add in the
            # ones we want (given that we have to loop through the
            # levels above in order to up-sample)
            for band in reversed(range(self.num_orientations)):
                if (lev, band) in recon_keys:
                    filt = filters['bfilts'][:, band].reshape(bfiltsz, bfiltsz, order='F')
                    recon += upConv(image=self.pyr_coeffs[(lev, band)], filt=filt, edge_type=edges,
                                    stop=self.pyr_size[(lev, band)])

        # apply lo0filt
        recon = upConv(image=recon, filt=filters['lo0filt'], edge_type=edges, stop=recon.shape)

        if 'residual_highpass' in recon_keys:
            recon += upConv(image=self.pyr_coeffs['residual_highpass'], filt=filters['hi0filt'],
                            edge_type=edges, start=(0, 0), step=(1, 1), stop=recon.shape)

        return recon
