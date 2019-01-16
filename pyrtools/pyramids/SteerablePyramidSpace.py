import numpy as np
from .pyramid import Pyramid
from .c.wrapper import corrDn, upConv


class SteerablePyramidSpace(Pyramid):

    def __init__(self, image, height='auto', num_orientations=2, edge_type='reflect1'):
        """Steerable pyramid.

        - `image` - a 2D numpy array

        - `height` - an integer denoting number of pyramid levels desired.  'auto' (default) uses
        max_pyr_height from pyr_utils.

        - num_orientations: {1, 2, 4, 6}. the number of orientations you want in the steerable
        - pyramid filters. If you want a different value, see SteerablePyramidFreq. Note that this
        - is the order of the pyramid plus one.

        - `edge_type` - specifies edge-handling.  Options are:
            * `'circular'` - circular convolution
            * `'reflect1'` - reflect about the edge pixels
            * `'reflect2'` - reflect, doubling the edge pixels
            * `'repeat'` - repeat the edge pixels
            * `'zero'` - assume values of zero outside image boundary
            * `'extend'` - reflect and invert
            * `'dont-compute'` - zero output when filter overhangs imput boundaries.

        """
        super().__init__(image=image, edge_type=edge_type)

        self.num_orientations = num_orientations
        self.filters = self._parse_filter("sp{:d}_filters".format(num_orientations-1))
        self.pyr_type = 'SteerableSpace'
        self._set_num_scales('lofilt', height)

        hi0 = corrDn(image=self.image, filt=self.filters['hi0filt'], edges=self.edge_type)

        self.pyr_coeffs['residual_highpass'] = hi0
        self.pyr_size['residual_highpass'] = hi0.shape

        lo = corrDn(image=self.image, filt=self.filters['lo0filt'], edges=self.edge_type)
        for i in range(self.num_scales):
            # assume square filters  -- start of buildSpyrLevs
            bfiltsz = int(np.floor(np.sqrt(self.filters['bfilts'].shape[0])))

            for b in range(self.num_orientations):
                filt = self.filters['bfilts'][:, b].reshape(bfiltsz, bfiltsz).T
                band = corrDn(image=lo, filt=filt, edges=self.edge_type)
                self.pyr_coeffs[(i, b)] = np.array(band)
                self.pyr_size[(i, b)] = band.shape

            lo = corrDn(image=lo, filt=self.filters['lofilt'], edges=self.edge_type, step=(2, 2))

        self.pyr_coeffs['residual_lowpass'] = lo
        self.pyr_size['residual_lowpass'] = lo.shape

    def recon_pyr(self, num_orientations=None, edge_type=None, levels='all', bands='all'):
        """Reconstruct the image, optionally using subset of pyramid coefficients.
        """

        if num_orientations is None:
            filters = self.filters
        else:
            filters = self._parse_filter("sp{:d}_filters".format(num_orientations-1))

        # assume square filters  -- start of buildSpyrLevs
        bfiltsz = int(np.floor(np.sqrt(filters['bfilts'].shape[0])))

        if edge_type is None:
            edges = self.edge_type
        else:
            edges = edge_type

        recon_keys = self._recon_keys(levels, bands, num_orientations)

        # initialize reconstruction
        if 'residual_lowpass' in recon_keys:
            recon = self.pyr_coeffs['residual_lowpass']
        else:
            recon = np.zeros_like(self.pyr_coeffs['residual_lowpass'])

        for lev in reversed(range(self.num_scales)):
            # we need to upConv once per level, in order to up-sample
            # the image back to the right shape.
            recon = upConv(image=recon, filt=filters['lofilt'], edges=edges,
                           step=(2, 2), start=(0, 0), stop=self.pyr_size[(lev, 0)])
            # I think the most effective way to do this is to just
            # check every possible sub-band and then only add in the
            # ones we want (given that we have to loop through the
            # levels above in order to up-sample)
            for band in reversed(range(self.num_orientations)):
                if (lev, band) in recon_keys:
                    filt = filters['bfilts'][:, band].reshape(bfiltsz, bfiltsz, order='F')
                    recon += upConv(image=self.pyr_coeffs[(lev, band)], filt=filt, edges=edges,
                                    stop=self.pyr_size[(lev, band)])

        # apply lo0filt
        recon = upConv(image=recon, filt=filters['lo0filt'], edges=edges, stop=recon.shape)

        if 'residual_highpass' in recon_keys:
            recon += upConv(image=self.pyr_coeffs['residual_highpass'], filt=filters['hi0filt'],
                            edges=edges, start=(0, 0), step=(1, 1), stop=recon.shape)

        return recon
