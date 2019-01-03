import numpy as np
import functools
from operator import mul

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
        self.pyr_type = None
        self.edge_type = edge_type
        self.edgeType = edge_type
        self.pyr = []
        self.pyrSize = []
        self.is_complex = False

    # this is the base Pyramid class. each subclass should implement their own
    # functionalities, including:
    def initFilters(self):
        raise Exception('Error: Not implemented for base Pyramid class')

    def initHeight(self):
        raise Exception('Error: Not implemented for base Pyramid class')

    def buildNext(self):
        raise Exception('Error: Not implemented for base Pyramid class')

    def buildPyr(self):
        raise Exception('Error: Not implemented for base Pyramid class')

    def reconPrev(self):
        raise Exception('Error: Not implemented for base Pyramid class')

    def reconPyr(self):
        raise Exception('Error: Not implemented for base Pyramid class')

    # shared methods
    def nbands(self):
        return len(self.pyr)

    def band(self, bandNum):
        assert bandNum < len(self.pyr), 'band number is out of range'
        return np.array(self.pyr[bandNum])

    # return concatenation of all levels of 1d pyramid / not used?
    def concatBands(self):
        outarray = np.array([]).reshape((1,0))
        for i in range(self.nbands()):
            tmp = self.band(i).T
            outarray = np.concatenate((outarray, tmp), axis=1)
        return outarray

    def setValue(self, band, location, value):
        """set a pyramid value
        location must be a tuple, others are single numbers
        """
        self.pyr[band][location[0],location[1]] = value

    def maxPyrHt(self, imsz, filtsz):
        ''' Compute maximum pyramid height for given image and filter sizes.
            Specifically: the number of corrDn operations that can be sequentially
            performed when subsampling by a factor of 2. '''
        # check if inputs are one of int, tuple and have consistent type
        assert (isinstance(imsz, int) and isinstance(filtsz, int)) or (
                isinstance(imsz, tuple) and isinstance(filtsz, tuple))
        # 1D image case: reduce to the integer case
        if isinstance(imsz, tuple) and (len(imsz) == 1 or 1 in imsz):
            imsz = functools.reduce(mul, imsz)
            filtsz = functools.reduce(mul, filtsz)
        # integer case
        if isinstance(imsz, int):
            if imsz < filtsz:
                return 0
            else:
                return 1 + self.maxPyrHt( imsz // 2, filtsz )
        # 2D image case
        if isinstance(imsz, tuple):
            if min(imsz) < max(filtsz):
                return 0
            else:
                return 1 + self.maxPyrHt( (imsz[0] // 2, imsz[1] // 2), filtsz )

    def _parse_filter(self, filt):
        if isinstance(filt, str):
            filt = namedFilter(filt)
        filt = np.array(filt)

        if filt.size > max(filt.shape):
            raise Exception("Error: filter should be 1D (i.e., a vector)")

        # when the first dimension of the image is 1, we need the filter to have shape (1, x)
        # instead of the normal (x, 1) or we get a segfault during corrDn / upConv. That's because
        # we need to match the filter to the image dimensions
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
            bands = np.array(bands)
            assert (bands >= 0).all(), "Error: band numbers must be larger than 0."
            assert (bands < self.num_orientations).all(), "Error: band numbers must be in the range [0, %d]" % (self.num_orientations - 1)
        return bands

    def _recon_keys(self, levels, bands):
        """make a list of all the keys from pyr_coeffs to use in pyramid reconstruction
        """
        levels = self._recon_levels_check(levels)
        bands = self._recon_bands_check(bands)
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


# maxPyrHt
# showPyr
