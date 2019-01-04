import numpy as np
import warnings
from .pyr_utils import max_pyr_height
from .filters import namedFilter


class Pyramid:  # Pyramid base class

    def __init__(self, image, edge_type):
        ''' - `edge_type` - specifies edge-handling.  Options are:
            * `'circular'` - circular convolution
            * `'reflect1'` - reflect about the edge pixels
            * `'reflect2'` - reflect, doubling the edge pixels
            * `'repeat'` - repeat the edge pixels
            * `'zero'` - assume values of zero outside image boundary
            * `'extend'` - reflect and invert
            * `'dont-compute'` - zero output when filter overhangs imput boundaries.
            '''
        self.image = np.array(image).astype(np.float)
        if self.image.ndim == 1:
            self.image = self.image.reshape(-1, 1)
        assert self.image.ndim == 2, "Error: Input signal must be 1D or 2D."

        self.image_size = self.image.shape
        if not hasattr(self, 'pyr_type'):
            self.pyr_type = None
        self.edge_type = edge_type
        self.pyr_coeffs = {}
        self.pyr_size = {}
        self.is_complex = False

    def _set_num_scales(self, filter_name, height, extra_height=0):
        # the Gaussian and Laplacian pyramids can go one higher than the value returned here, so we
        # use the extra_height argument to allow for that
        max_ht = max_pyr_height(self.image.shape, self.filters[filter_name].shape) + extra_height
        if height == 'auto':
            self.num_scales = max_ht
        elif height > max_ht:
            raise Exception("cannot build pyramid higher than %d levels." % (max_ht))
        else:
            self.num_scales = int(height)

    def _parse_filter(self, filt):
        if isinstance(filt, str):
            filt = namedFilter(filt)

        # the steerable pyramid filters are returned as a dictionary and we don't need to do this
        # check for them
        if not isinstance(filt, dict):
            if filt.size > max(filt.shape):
                raise Exception("Error: filter should be 1D (i.e., a vector)")

            # when the first dimension of the image is 1, we need the filter to have shape (1, x)
            # instead of the normal (x, 1) or we get a segfault during corrDn / upConv. That's
            # because we need to match the filter to the image dimensions
            if filt.ndim == 1 or self.image.shape[0] == 1:
                filt = filt.reshape(1, -1)
        return filt

    def _recon_levels_check(self, levels):
        """when reconstructing pyramid, check whether levels arg is valid and return
        """
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
        # not all pyramids have residual highpass / lowpass, but it's easier to construct the list
        # including them, then remove them if necessary.
        if 'residual_lowpass' not in self.pyr_coeffs.keys() and 'residual_lowpass' in levels:
            levels.pop(-1)
        if 'residual_highpass' not in self.pyr_coeffs.keys() and 'residual_highpass' in levels:
            levels.pop(0)
        return levels

    def _recon_bands_check(self, bands):
        """when reconstructing pyramid, check whether bands arg is valid and return
        """
        if isinstance(bands, str) and bands == "all":
            bands = np.arange(self.num_orientations)
        else:
            bands = np.array(bands, ndmin=1)
            assert (bands >= 0).all(), "Error: band numbers must be larger than 0."
            assert (bands < self.num_orientations).all(), "Error: band numbers must be in the range [0, %d]" % (self.num_orientations - 1)
        return bands

    def _recon_keys(self, levels, bands, max_orientations=None):
        """make a list of all the keys from pyr_coeffs to use in pyramid reconstruction

        max_orientations: None or int. The maximum number of orientations we allow in the
        reconstruction. when we determine which ints are allowed for bands, we ignore all those
        greater than max_orientations.
        """
        levels = self._recon_levels_check(levels)
        bands = self._recon_bands_check(bands)
        if max_orientations is not None:
            for i in bands:
                if i >= max_orientations:
                    warnings.warn(("You wanted band %d in the reconstruction but max_orientation"
                                   " is %d, so we're ignoring that band" % (i, max_orientations)))
            bands = [i for i in bands if i < max_orientations]
        recon_keys = []
        for level in levels:
            # residual highpass and lowpass
            if isinstance(level, str):
                recon_keys.append(level)
            # else we have to get each of the (specified) bands at
            # that level
            else:
                recon_keys.extend([(level, band) for band in bands])
        return recon_keys
