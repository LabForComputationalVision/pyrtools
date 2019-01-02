import numpy as np
from .pyramid import Pyramid
from .filters import steerable_filters
from .c.wrapper import corrDn, upConv


class SteerablePyramidSpace(Pyramid):

    def __init__(self, image, height='auto', filters='sp1Filters',
                 edge_type='reflect1'):
        """Steerable pyramid. image parameter is required, others are optional

        - `image` - a 2D numpy array

        - `height` - an integer denoting number of pyramid levels desired.  'auto' (default) uses
        maxPyrHt from pyPyrUtils.

        - `filters` - The name of one of the steerable pyramid filters:
        `'sp0Filters'`, `'sp1Filters'`, `'sp3Filters'`, `'sp5Filters'`.

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

        self.filters = steerable_filters(filters)
        self.pyr_type = 'Steerable'
        self.num_orientations = int(filters.replace('sp', '').replace('Filters', '')) + 1

        max_ht = self.maxPyrHt(self.image.shape, self.filters['lofilt'].shape)
        if height == 'auto':
            self.num_scales = max_ht
        elif height > max_ht:
            raise Exception("cannot build pyramid higher than %d levels." % (max_ht))
        else:
            self.num_scales = int(height)

        self.pyr_coeffs = {}
        self.pyr_size = {}

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
                self.pyr_size[(i, b)] = (band.shape[0], band.shape[1])

            lo = corrDn(image=lo, filt=self.filters['lofilt'], edges=self.edge_type, step=(2, 2))

        self.pyr_coeffs['residual_lowpass'] = np.array(lo)
        self.pyr_size['residual_lowpass'] = lo.shape

    def recon_pyr(self, filters=None, edge_type=None, levels='all', bands='all'):
        """Reconstruct the image, optionally using subset of pyramid coefficients.
        """
        # defaults

        if filters is None:
            filters = self.filters
        else:
            filters = steerable_filters(filters)

        # assume square filters  -- start of buildSpyrLevs
        bfiltsz = int(np.floor(np.sqrt(filters['bfilts'].shape[0])))

        if edge_type is None:
            edges = self.edge_type
        else:
            edges = edge_type

        if isinstance(levels, str) and levels == 'all':
            levels = ['residual_highpass'] + list(range(self.num_scales)) + ['residual_lowpass']
        else:
            levs_nums = np.array([int(i) for i in levels if isinstance(i, int) or i.isdigit()])
            assert (levs_nums >= 0).all(), "Level numbers must be non-negative."
            assert (levs_nums < self.num_scales).all(), "Level numbers must be in the range [0, %d]" % (self.num_scales-1)
            levs_tmp = list(np.sort(levs_nums))  # we want smallest first
            if 'residual_highpass' in levels:
                levs_tmp = ['residual_highpass'] + levs_tmp
            if 'residual_lowpass' in levels:
                levs_tmp = levs_tmp + ['residual_lowpass']
            levels = levs_tmp

        if isinstance(bands, str) and bands == "all":
            bands = np.arange(self.num_orientations)
        else:
            bands = np.array(bands)
            assert (bands >= 0).all(), "Error: band numbers must be larger than 0."
            assert (bands < self.num_orientations).all(), "Error: band numbers must be in the range [0, %d]" % (self.num_orientations - 1)

        # make a list of all the keys from pyr_coeffs to use in pyramid reconstruction
        recon_keys = []
        for level in levels:
            # residual highpass and lowpass
            if isinstance(level, str):
                recon_keys.append(level)
            # else we have to get each of the (specified) bands at
            # that level
            else:
                recon_keys.extend([(level, band) for band in bands])

        # initialize reconstruction
        if 'residual_lowpass' in recon_keys:
            recon = self.pyr_coeffs['residual_lowpass']
        else:
            recon = np.zeros(self.pyr_coeffs['residual_lowpass'].shape)

        # this just goes through the levels backwards, from top to
        # bottom
        for lev in range(self.num_scales)[::-1]:
            # we need to upConv once per level, in order to up-sample
            # the image back to the right shape.
            recon = upConv(image=recon, filt=filters['lofilt'], edges=edges,
                           step=(2, 2), start=(0, 0), stop=self.pyr_size[(lev, 0)])
            # I think the most effective way to do this is to just
            # check every possible sub-band and then only add in the
            # ones we want (given that we have to loop through the
            # levels above in order to up-sample)
            for band in range(self.num_orientations)[::-1]:
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
