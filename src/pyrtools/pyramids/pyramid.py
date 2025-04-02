import numpy as np
import warnings
from .pyr_utils import max_pyr_height
from .filters import named_filter
from .steer import steer


class Pyramid:
    """Base class for multiscale pyramids

    You should not instantiate this base class, it is instead inherited by the other classes found
    in this module.

    Parameters
    ----------
    image : `array_like`
        1d or 2d image upon which to construct to the pyramid.
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
    """

    def __init__(self, image, edge_type):

        self.image = np.asarray(image).astype(float)
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
        """Figure out the number of scales (height) of the pyramid

        The user should not call this directly. This is called during construction of a pyramid,
        and is based on the size of the filters (thus, should be called after instantiating the
        filters) and the input image, as well as the `extra_height` parameter (which corresponds to
        the residuals, which the Gaussian pyramid contains and others do not).

        This sets `self.num_scales` directly instead of returning something, so be careful.

        Parameters
        ----------
        filter_name : `str`
            Name of the filter in the `filters` dict that determines the height of the pyramid
        height : `'auto'` or `int`
            During construction, user can specify the number of scales (height) of the pyramid.
            The pyramid will have this number of scales unless that's greater than the maximum
            possible height.
        extra_height : `int`, optional
            The automatically calculated maximum number of scales is based on the size of the input
            image and filter size. The Gaussian pyramid also contains the final residuals and so we
            need to add one more to this number.

        Returns
        -------
        None
        """
        # the Gaussian and Laplacian pyramids can go one higher than the value returned here, so we
        # use the extra_height argument to allow for that
        max_ht = max_pyr_height(self.image.shape, self.filters[filter_name].shape) + extra_height
        if height == 'auto':
            self.num_scales = max_ht
        elif height > max_ht:
            raise ValueError("Cannot build pyramid higher than %d levels." % (max_ht))
        else:
            self.num_scales = int(height)

    def _recon_levels_check(self, levels):
        """Check whether levels arg is valid for reconstruction and return valid version

        When reconstructing the input image (i.e., when calling `recon_pyr()`), the user specifies
        which levels to include. This makes sure those levels are valid and gets them in the form
        we expect for the rest of the reconstruction. If the user passes `'all'`, this constructs
        the appropriate list (based on the values of `self.pyr_coeffs`).

        Parameters
        ----------
        levels : `list`, `int`,  or {`'all'`, `'residual_highpass'`, or `'residual_lowpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_highpass'` and `'residual_lowpass'` (if appropriate for the
            pyramid). If `'all'`, returned value will contain all valid levels. Otherwise, must be
            one of the valid levels.

        Returns
        -------
        levels : `list`
            List containing the valid levels for reconstruction.

        """
        if isinstance(levels, str) and levels == 'all':
            levels = ['residual_highpass'] + list(range(self.num_scales)) + ['residual_lowpass']
        else:
            if not hasattr(levels, '__iter__') or isinstance(levels, str):
                # then it's a single int or string
                levels = [levels]
            levs_nums = np.asarray([int(i) for i in levels if isinstance(i, int) or i.isdigit()])
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
        """Check whether bands arg is valid for reconstruction and return valid version

        When reconstructing the input image (i.e., when calling `recon_pyr()`), the user specifies
        which orientations to include. This makes sure those orientations are valid and gets them
        in the form we expect for the rest of the reconstruction. If the user passes `'all'`, this
        constructs the appropriate list (based on the values of `self.pyr_coeffs`).

        Parameters
        ----------
        bands : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations.

        Returns
        -------
        bands: `list`
            List containing the valid orientations for reconstruction.
        """
        if isinstance(bands, str) and bands == "all":
            bands = np.arange(self.num_orientations)
        else:
            bands = np.array(bands, ndmin=1)
            assert (bands >= 0).all(), "Error: band numbers must be larger than 0."
            assert (bands < self.num_orientations).all(), "Error: band numbers must be in the range [0, %d]" % (self.num_orientations - 1)
        return bands

    def _recon_keys(self, levels, bands, max_orientations=None):
        """Make a list of all the relevant keys from `pyr_coeffs` to use in pyramid reconstruction

        When reconstructing the input image (i.e., when calling `recon_pyr()`), the user specifies
        some subset of the pyramid coefficients to include in the reconstruction. This function
        takes in those specifications, checks that they're valid, and returns a list of tuples
        that are keys into the `pyr_coeffs` dictionary.

        Parameters
        ----------
        levels : `list`, `int`,  or {`'all'`, `'residual_highpass'`, `'residual_lowpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_highpass'` and `'residual_lowpass'` (if appropriate for the
            pyramid). If `'all'`, returned value will contain all valid levels. Otherwise, must be
            one of the valid levels.
        bands : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations.
        max_orientations: `None` or `int`.
            The maximum number of orientations we allow in the reconstruction. when we determine
            which ints are allowed for bands, we ignore all those greater than max_orientations.

        Returns
        -------
        recon_keys : `list`
            List of `tuples`, all of which are keys in `pyr_coeffs`. These are the coefficients to
            include in the reconstruction of the image.

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


class SteerablePyramidBase(Pyramid):
    """base class for steerable pyramid

    should not be called directly, we just use it so we can make both SteerablePyramidFreq and
    SteerablePyramidSpace inherit the steer_coeffs function

    """
    def __init__(self, image, edge_type):
        super().__init__(image=image, edge_type=edge_type)

    def steer_coeffs(self, angles, even_phase=True):
        """Steer pyramid coefficients to the specified angles

        This allows you to have filters that have the Gaussian derivative order specified in
        construction, but arbitrary angles or number of orientations.

        Parameters
        ----------
        angles : `list`
            list of angles (in radians) to steer the pyramid coefficients to
        even_phase : `bool`
            specifies whether the harmonics are cosine or sine phase aligned about those positions.

        Returns
        -------
        resteered_coeffs : `dict`
            dictionary of re-steered pyramid coefficients. will have the same number of scales as
            the original pyramid (though it will not contain the residual highpass or lowpass).
            like `self.pyr_coeffs`, keys are 2-tuples of ints indexing the scale and orientation,
            but now we're indexing `angles` instead of `self.num_orientations`.
        resteering_weights : `dict`
            dictionary of weights used to re-steer the pyramid coefficients. will have the same
            keys as `resteered_coeffs`.

        """
        resteered_coeffs = {}
        resteering_weights = {}
        for i in range(self.num_scales):
            basis = np.vstack([self.pyr_coeffs[(i, j)].flatten() for j in
                               range(self.num_orientations)]).T
            for j, a in enumerate(angles):
                res, steervect = steer(basis, a, return_weights=True, even_phase=even_phase)
                resteered_coeffs[(i, j)] = res.reshape(self.pyr_coeffs[(i, 0)].shape)
                resteering_weights[(i, j)] = steervect

        return resteered_coeffs, resteering_weights
