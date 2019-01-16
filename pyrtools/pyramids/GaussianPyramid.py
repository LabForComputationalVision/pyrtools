from .pyramid import Pyramid
from .c.wrapper import corrDn


class GaussianPyramid(Pyramid):

    def __init__(self, image, height='auto', filter_name='binom5',
                 edge_type='reflect1', **kwargs):
        """Gaussian pyramid
            - `image` - a 2D numpy array

            - `height` - an integer denoting number of pyramid levels desired. Defaults to
              1+`max_pyr_heightt`

            - `filter_name` - can be a string naming a standard filter (from
              named_filter()), or a numpy array which will be used for (separable) convolution.

            - `edge_type` - see class Pyramid.__init__()

        """
        super().__init__(image=image, edge_type=edge_type)
        if self.pyr_type is None:
            self.pyr_type = 'Gaussian'
        self.num_orientations = 1

        self.filters = {'downsample_filter': self._parse_filter(filter_name)}
        upsamp_filt = kwargs.pop('upsample_filter_name', None)
        if upsamp_filt is not None:
            if self.pyr_type != 'Laplacian':
                raise Exception("upsample_filter should only be set for Laplacian pyramid!")
            self.filters['upsample_filter'] = self._parse_filter(upsamp_filt)
        self._set_num_scales('downsample_filter', height, 1)

        self._build_pyr()

    def _build_next(self, image):
        if image.shape[0] == 1:
            res = corrDn(image=image, filt=self.filters['downsample_filter'], edges=self.edge_type,
                         step=(1, 2))
        elif image.shape[1] == 1:
            res = corrDn(image=image, filt=self.filters['downsample_filter'], edges=self.edge_type,
                         step=(2, 1))
        else:
            tmp = corrDn(image=image, filt=self.filters['downsample_filter'].T,
                         edges=self.edge_type, step=(1, 2))
            res = corrDn(image=tmp, filt=self.filters['downsample_filter'], edges=self.edge_type,
                         step=(2, 1))
        return res

    def _build_pyr(self):
        """build the pyramid

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
        raise Exception('Not necessary for Gaussian Pyramids')
