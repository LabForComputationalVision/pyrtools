from .Lpyr import LaplacianPyramid

class GaussianPyramid(LaplacianPyramid):

    # constructor
    def __init__(self, image, height='auto', filter='binom5',
                 edgeType='reflect1', pyrType='Gaussian'):
        """Gaussian pyramid

            - `image` - a 2D numpy array
            - `height` - an integer denoting number of pyramid levels desired. Defaults to `maxPyrHt`
            - `filter` - can be a string namimg a standard filter (from namedFilter()), or a
            numpy array which will be used for (separable) convolution. Default is 'binom5'.
            - `edgeType` - see class Pyramid.__init__()
            """
        super().__init__(image=image, height=height, filter1=filter,
                         edgeType=edgeType, pyrType=pyrType)

    # methods

    def buildPyr(self):

        img = self.image
        if len(img.shape) == 1:
            img = img.reshape(-1, 1)

        self.pyr.append(img.copy())
        self.pyrSize.append(img.shape)

        for h in range(1,self.height):
            img = self.downSample(img)
            self.pyr.append(img.copy())
            self.pyrSize.append(img.shape)

    def reconPyr(self, *args):
        raise Exception('Error: undefined for Gaussian Pyramids...')
