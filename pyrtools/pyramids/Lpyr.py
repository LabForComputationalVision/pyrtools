import numpy as np
from .Gpyr import GaussianPyramid
from .c.wrapper import upConv

class LaplacianPyramid(GaussianPyramid):

    # constructor
    def __init__(self, image, height='auto', filter1='binom5', filter2=None,
                 edgeType='reflect1'):
        """Laplacian pyramid

            - `image` - a 2D numpy array
            - `height` - an integer denoting number of pyramid levels desired. Defaults to `maxPyrHt`
            - `filter1` - can be a string namimg a standard filter (from namedFilter()), or a
            numpy array which will be used for (separable) convolution. Default is 'binom5'.
            - `filter2` - specifies the "expansion" filter (default = filter1).
            - `edges` - see class Pyramid.__init__()
            """
        super().__init__(image=image, height=height, filter1=filter1, filter2=filter2,
                         edgeType=edgeType)
        self.pyrType = 'Laplacian'


    # methods
    # def buildNext():
    #     same as GaussianPyramid.buildNext()

    def reconPrev(self, image, out_size, filt=None, edges=None):
        if filt is None:
            filt = self.filter2
        else:
            filt = self.parseFilter(filt)
        if edges is None: edges = self.edgeType
        imsz = image.shape

        if len(imsz) == 1 or imsz[1] == 1:
            res = upConv(image=image, filt=filt.T, edges=edges,step=(1,2), stop=(out_size[1], out_size[0])).T
        elif imsz[0] == 1:
            res = upConv(image=image, filt=filt.T, edges=edges, step=(2,1), stop=(out_size[1], out_size[0])).T
        else:
            tmp = upConv(image=image, filt=filt, edges=edges, step=(2,1), stop=(out_size[0], imsz[1]))
            res = upConv(image=tmp, filt=filt.T, edges=edges, step=(1,2), stop=(out_size[0], out_size[1]))
        return res

    def buildPyr(self):
        img = self.image
        if len(img.shape) == 1:
            img = img.reshape(-1, 1)
        for h in range(1,self.height):
            img_next = self.buildNext(img)
            img_recon = self.reconPrev(img_next, out_size=img.shape)
            img_residual = img - img_recon
            self.pyr.append(img_residual.copy())
            self.pyrSize.append(img_residual.shape)
            img = img_next
        self.pyr.append(img.copy())
        self.pyrSize.append(img.shape)

    def reconPyr(self, levs='all', filter2=None, edgeType=None):
        if isinstance(levs, str) and levs == 'all':
            levs = np.arange(self.height)
        levs = np.array(levs)
        res = self.band(levs.max())
        for lev in range(levs.max()-1, -1, -1):
            # upsample to generate higher resolution image
            res = self.reconPrev(res, out_size=self.band(lev).shape, filt=filter2, edges=edgeType)
            if lev in levs:
                res += self.band(lev)
        return res
