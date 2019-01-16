import numpy as np
from .GaussianPyramid import GaussianPyramid
from .c.wrapper import upConv


class LaplacianPyramid(GaussianPyramid):

    def __init__(self, image, height='auto', downsample_filter_name='binom5',
                 upsample_filter_name=None, edge_type='reflect1'):
        """Laplacian pyramid

            - `image` - a 2D numpy array

            - `height` - an integer denoting number of pyramid levels desired. Defaults to
              1+`max_pyr_height`

            - `downsample_filter_name` - can be a string namimg a standard filter (from
              named_filter()), or a numpy array which will be used for (separable)
              convolution to downsample the image. Default is 'binom5'.

            - `upsample_filter_name` - specifies the "expansion" filter. If None, then sets it to
              the same as downsample_filter_name

            - `edge_type` - see class Pyramid.__init__()

        """
        self.pyr_type = 'Laplacian'
        if upsample_filter_name is None:
            upsample_filter_name = downsample_filter_name
        super().__init__(image, height, downsample_filter_name, edge_type,
                         upsample_filter_name=upsample_filter_name)

    def _recon_prev(self, image, output_size, upsample_filter=None, edge_type=None):
        if upsample_filter is None:
            upsample_filter = self.filters['upsample_filter']
        else:
            upsample_filter = self._parse_filter(upsample_filter)

        if edge_type is None:
            edge_type = self.edge_type

        if image.shape[1] == 1:
            res = upConv(image=image, filt=upsample_filter.T, edges=edge_type,
                         step=(1, 2), stop=(output_size[1], output_size[0])).T
        elif image.shape[0] == 1:
            res = upConv(image=image, filt=upsample_filter.T, edges=edge_type,
                         step=(2, 1), stop=(output_size[1], output_size[0])).T
        else:
            tmp = upConv(image=image, filt=upsample_filter, edges=edge_type,
                         step=(2, 1), stop=(output_size[0], image.shape[1]))
            res = upConv(image=tmp, filt=upsample_filter.T, edges=edge_type,
                         step=(1, 2), stop=(output_size[0], output_size[1]))
        return res

    def _build_pyr(self):
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

    def recon_pyr(self, upsample_filter_name=None, edge_type=None, levels='all'):
        recon_keys = self._recon_keys(levels, 'all')
        recon = np.zeros_like(self.pyr_coeffs[(self.num_scales-1, 0)])
        for lev in reversed(range(self.num_scales)):
            # upsample to generate higher reconolution image
            recon = self._recon_prev(recon, self.pyr_size[(lev, 0)], upsample_filter_name,
                                     edge_type)
            if (lev, 0) in recon_keys:
                recon += self.pyr_coeffs[(lev, 0)]
        return recon
