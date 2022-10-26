import unittest
import tarfile
import tqdm
import math
import requests

import numpy as np
import matplotlib.pyplot as plt

import pyrtools as pt
from pyrtools.pyramids.pyramid import Pyramid

import scipy.io
import os
import os.path as op
matfiles_path = op.join(op.dirname(op.realpath(__file__)), 'matFiles')
test_data_path = op.join(op.dirname(op.realpath(__file__)), '..', 'DATA')

# TODO:
# - create class upConvTests(unittest.TestCase):
# - expand class corrDnTests(unittest.TestCase):
# - expand class blurDnTests(unittest.TestCase):
# - clean up histo function and then run histoTests and entropy2Tests

class corrDnTests(unittest.TestCase):
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'corrDn1.mat'))
        mres = matPyr['res']
        ramp = pt.synthetic_images.ramp(20)
        res = pt.corrDn(ramp, pt.named_filter('binom5'))
        self.assertTrue(pt.compareRecon(mres, res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'corrDn2.mat'))
        mres = matPyr['res']
        ramp = pt.synthetic_images.ramp(20)
        res = pt.corrDn(ramp, pt.named_filter('qmf13'))
        self.assertTrue(pt.compareRecon(mres, res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'corrDn3.mat'))
        mres = matPyr['res']
        ramp = pt.synthetic_images.ramp(20)
        res = pt.corrDn(ramp, pt.named_filter('qmf16'))
        self.assertTrue(pt.compareRecon(mres, res))

class blurTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blur0.mat'))
        ramp = pt.synthetic_images.ramp(20)
        res = pt.blur(ramp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blur1.mat'))
        ramp = pt.synthetic_images.ramp(20)
        res = pt.blur(ramp, n_levels=3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blur2.mat'))
        ramp = pt.synthetic_images.ramp(20)
        res = pt.blur(ramp, n_levels=3, filt=pt.named_filter('qmf5'))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blur3.mat'))
        ramp = pt.synthetic_images.ramp((20,30))
        res = pt.blur(ramp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blur4.mat'))
        ramp = pt.synthetic_images.ramp((20,30))
        res = pt.blur(ramp, n_levels=3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blur5.mat'))
        ramp = pt.synthetic_images.ramp((20,30))
        res = pt.blur(ramp, n_levels=3, filt=pt.named_filter('qmf5'))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))


class blurDnTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn0.mat'))
        pyRamp = pt.synthetic_images.ramp((20,20))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn1.mat'))
        pyRamp = pt.synthetic_images.ramp((256,256))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn2.mat'))
        pyRamp = pt.synthetic_images.ramp((256,128))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn3.mat'))
        pyRamp = pt.synthetic_images.ramp((128,256))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn4.mat'))
        pyRamp = pt.synthetic_images.ramp((200, 100))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn5.mat'))
        pyRamp = pt.synthetic_images.ramp((100, 200))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn6.mat'))
        pyRamp = pt.synthetic_images.ramp((1, 256)).T
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn7.mat'))
        pyRamp = pt.synthetic_images.ramp((1, 256))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    # def test8(self):#  need a 2D filter
    #    matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn8.mat'))
    #    pyRamp = pt.synthetic_images.ramp((256, 256))
    #    res = pt.blurDn(pyRamp, filt=2dfilt)
    #    self.assertTrue(pt.compareRecon(matPyr['res'], res))

class upBlurTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur0.mat'))
        mres = matPyr['res']
        im = pt.synthetic_images.ramp((1,20))
        res = pt.upBlur(im)
        self.assertTrue(pt.compareRecon(mres, res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur1.mat'))
        mres = matPyr['res']
        im = pt.synthetic_images.ramp((1,20))
        res = pt.upBlur(im.T)
        self.assertTrue(pt.compareRecon(mres, res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur2.mat'))
        mres = matPyr['res']
        im = pt.synthetic_images.ramp(20)
        res = pt.upBlur(im)
        self.assertTrue(pt.compareRecon(mres, res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur3.mat'))
        mres = matPyr['res']
        im = pt.synthetic_images.ramp((1,20))
        res = pt.upBlur(im, n_levels=3)
        self.assertTrue(pt.compareRecon(mres, res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur4.mat'))
        mres = matPyr['res']
        im = pt.synthetic_images.ramp((1,20))
        res = pt.upBlur(im.T, n_levels=3)
        self.assertTrue(pt.compareRecon(mres, res))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur5.mat'))
        mres = matPyr['res']
        im = pt.synthetic_images.ramp(20)
        res = pt.upBlur(im, n_levels=3)
        self.assertTrue(pt.compareRecon(mres, res))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur6.mat'))
        mres = matPyr['res']
        im = pt.synthetic_images.ramp((1,20))
        filt = pt.named_filter('qmf9')
        res = pt.upBlur(im, n_levels=3, filt=filt)
        self.assertTrue(pt.compareRecon(mres, res))
    #def test7(self):   # fails in matlab and python because of dim mismatch
    #    matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur7.mat'))
    #    mres = matPyr['res']
    #    im = pt.synthetic_images.ramp((1,20))
    #    filt = pt.named_filter('qmf9')
    #    res = pt.upBlur(im, 3, filt.T)
    #    self.assertTrue(pt.compareRecon(mres, res))
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur8.mat'))
        mres = matPyr['res']
        im = pt.synthetic_images.ramp((1,20))
        filt = pt.named_filter('qmf9')
        res = pt.upBlur(im.T, n_levels=3, filt=filt)
        self.assertTrue(pt.compareRecon(mres, res))
    #def test9(self):  # fails in matlab and python because of dim mismatch
    #    matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur6.mat'))
    #    mres = matPyr['res']
    #    im = pt.synthetic_images.ramp((1,20))
    #    filt = pt.named_filter('qmf9')
    #    res = pt.upBlur(im.T, 3, filt.T)
    #    self.assertTrue(pt.compareRecon(mres, res))
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur10.mat'))
        mres = matPyr['res']
        im = pt.synthetic_images.ramp(20)
        filt = pt.synthetic_images.disk(3)
        res = pt.upBlur(im, n_levels=3, filt=filt)
        self.assertTrue(pt.compareRecon(mres, res))
    def test11(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur11.mat'))
        mres = matPyr['res']
        im = pt.synthetic_images.ramp((20,10))
        filt = pt.synthetic_images.disk((5,3))
        res = pt.upBlur(im, n_levels=3, filt=filt)
        self.assertTrue(pt.compareRecon(mres, res))
    def test12(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur12.mat'))
        mres = matPyr['res']
        im = pt.synthetic_images.ramp((10,20))
        filt = pt.synthetic_images.disk((3,5))
        res = pt.upBlur(im, n_levels=3, filt=filt)
        self.assertTrue(pt.compareRecon(mres, res))

class pointOpTests(unittest.TestCase):
    def test1(self):
        matImg = scipy.io.loadmat(op.join(matfiles_path, 'pointOp1.mat'))
        img = pt.synthetic_images.ramp((200,200))
        filt = np.array([0.2, 0.5, 1.0, 0.4, 0.1]);
        #foo = pointOp(200, 200, img, 5, filt, 0, 1, 0);
        foo = pt.pointOp(img, filt, 0, 1);
        foo = np.reshape(foo,(200,200))
        self.assertTrue((matImg['foo'] == foo).all())

class maxPyrHeightTests(unittest.TestCase):
    def test1(self):
        self.assertTrue(pt.pyramids.max_pyr_height((1,10),(3,4)) == 0)
    def test2(self):
        self.assertTrue(pt.pyramids.max_pyr_height((10,1),(3,4)) == 0)
    def test3(self):
        self.assertTrue(pt.pyramids.max_pyr_height((10,10),(1,4)) == 2)
    def test4(self):
        self.assertTrue(pt.pyramids.max_pyr_height((10,10),(3,1)) == 2)
    def test5(self):
        self.assertTrue(pt.pyramids.max_pyr_height((10,10),(3,4)) == 2)
    def test6(self):
        self.assertTrue(pt.pyramids.max_pyr_height((20,10),(5,1)) == 2)
    def test7(self):
        self.assertTrue(pt.pyramids.max_pyr_height((10,20),(5,1)) == 2)
    def test8(self):
        self.assertTrue(pt.pyramids.max_pyr_height((20,10),(1,5)) == 2)
    def test9(self):
        self.assertTrue(pt.pyramids.max_pyr_height((10,20),(1,5)) == 2)
    def test10(self):
        self.assertTrue(pt.pyramids.max_pyr_height((256,1),(1,5)) == 6)
    def test11(self):
        self.assertTrue(pt.pyramids.max_pyr_height((256,1),(5,1)) == 6)
    def test12(self):
        self.assertTrue(pt.pyramids.max_pyr_height((1,256),(1,5)) == 6)
    def test13(self):
        self.assertTrue(pt.pyramids.max_pyr_height((1,256),(5,1)) == 6)

class binomialFilterTests(unittest.TestCase):
    def test1(self):
        target = np.array([[0.5],[0.5]])
        #target = target / np.sqrt(np.sum(target ** 2))
        self.assertTrue((pt.binomial_filter(2) == target).all() )
    def test2(self):
        target = np.array([[0.25], [0.5], [0.25]])
        #target = target / np.sqrt(np.sum(target ** 2))
        self.assertTrue((pt.binomial_filter(3) == target).all())
    def test3(self):
        target = np.array([[0.0625], [0.25], [0.3750], [0.25], [0.0625]])
        #target = target / np.sqrt(np.sum(target ** 2))
        self.assertTrue((pt.binomial_filter(5) == target).all())

class GpyrTests(unittest.TestCase):
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildGpyr1.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.GaussianPyramid(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildGpyr2row.mat'))
        img = np.array(list(range(256))).astype(float)
        img = img.reshape(1, 256)
        pyPyr = pt.pyramids.GaussianPyramid(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildGpyr2col.mat'))
        img = np.array(list(range(256))).astype(float)
        img = img.reshape(256, 1)
        pyPyr = pt.pyramids.GaussianPyramid(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildGpyr3.mat'))
        img = pt.synthetic_images.ramp(10)
        pyPyr = pt.pyramids.GaussianPyramid(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildGpyr4.mat'))
        img = pt.synthetic_images.ramp((10,20))
        pyPyr = pt.pyramids.GaussianPyramid(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildGpyr5.mat'))
        img = pt.synthetic_images.ramp((20, 10))
        pyPyr = pt.pyramids.GaussianPyramid(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))

class LpyrTests(unittest.TestCase):
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr1.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.LaplacianPyramid(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr2 = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr2.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr2['pyr'], pyPyr))
    def test3(self):
        matPyr2 = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr3.mat'))
        pyRamp = pt.synthetic_images.ramp((100,200))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr2['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr4.mat'))
        pyRamp = pt.synthetic_images.ramp(200,100)
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr5.mat'))
        pyRamp = np.array(list(range(200))).reshape(1, 200)
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test5bis(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr5.mat'))
        pyRamp = np.array(list(range(200)))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr6.mat'))
        pyRamp = np.array(list(range(200)))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr7.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.LaplacianPyramid(img)
        recon = pyPyr.recon_pyr()
        self.assertTrue((matPyr['recon'] == recon).all())
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr8.mat'))
        pyRamp = pt.synthetic_images.ramp(200)
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue((matPyr['recon'] == recon).all())
    def test9(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr9.mat'))
        pyRamp = pt.synthetic_images.ramp((200,100))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue((matPyr['recon'] == recon).all())
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr10.mat'))
        pyRamp = pt.synthetic_images.ramp((100,200))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue((matPyr['recon'] == recon).all())
    def test11(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr11.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        recon = pyPyr.recon_pyr(levels=[1])
        self.assertTrue((matPyr['recon'] == recon).all())
    def test12(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr12.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        recon = pyPyr.recon_pyr(levels=[0, 2, 4])
        self.assertTrue((matPyr['recon'] == recon).all())

class WpyrTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr0.mat'))
        pyRamp = pt.synthetic_images.ramp((20,20))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr1.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.WaveletPyramid(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr2.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr3.mat'))
        pyRamp = pt.synthetic_images.ramp((100,200))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr4.mat'))
        pyRamp = pt.synthetic_images.ramp((200,100))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr5.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.WaveletPyramid(img)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr6.mat'))
        pyRamp = pt.synthetic_images.ramp((256,128))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr7.mat'))
        pyRamp = pt.synthetic_images.ramp((128,256))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr8.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr9.mat'))
        pyRamp = pt.synthetic_images.ramp((200,100))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr10.mat'))
        pyRamp = pt.synthetic_images.ramp((100,200))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr11.mat'))
        pyRamp = pt.synthetic_images.ramp((256,256))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.recon_pyr('qmf9', 'reflect1', [0])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr12.mat'))
        pyRamp = pt.synthetic_images.ramp((256,256))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.recon_pyr('qmf9', 'reflect1', [0,2,4])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr13.mat'))
        pyRamp = pt.synthetic_images.ramp((256,256))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.recon_pyr('qmf9', 'reflect1', [0,2,4], [1])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test14(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr14.mat'))
        pyRamp = pt.synthetic_images.ramp((256,256))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.recon_pyr('qmf8')
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test15(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr15.mat'))
        pyRamp = pt.synthetic_images.ramp((256,128))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.recon_pyr('qmf8')
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test16(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr16.mat'))
        pyRamp = pt.synthetic_images.ramp((128,256))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.recon_pyr('qmf8')
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test17(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr17.mat'))
        pyRamp = pt.synthetic_images.ramp((1,200))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test18(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr18.mat'))
        pyRamp = pt.synthetic_images.ramp((1,200)).T
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def testRecon1(self):
        np.random.seed(0)
        im = pt.synthetic_images.pink_noise((1, 64))
        pyr = pt.pyramids.WaveletPyramid(im)
        res = pyr.recon_pyr()
        self.assertTrue(np.allclose(res, im, atol=5e-3))
    def testRecon2(self):
        np.random.seed(0)
        im = pt.synthetic_images.pink_noise((64, 1))
        pyr = pt.pyramids.WaveletPyramid(im)
        res = pyr.recon_pyr()
        self.assertTrue(np.allclose(res, im, atol=5e-3))
    def testRecon3(self):
        im = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.WaveletPyramid(im)
        res = pyr.recon_pyr()
        # print("\n%s\n" % np.max(res-im))
        self.assertTrue(np.allclose(res, im, atol=1))
    def testRecon4(self):
        im = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.WaveletPyramid(im, edge_type='circular')
        res = pyr.recon_pyr()
        # print("\n%s\n" % np.max(res-im))
        self.assertTrue(np.allclose(res, im, atol=1))
    def testRecon5(self):
        im = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.WaveletPyramid(im, filter_name='daub2', edge_type='circular')
        res = pyr.recon_pyr()
        # print("\n%s\n" % np.max(res-im))
        self.assertTrue(np.allclose(res, im, atol=5e-3))
    def testRecon6(self):
        im = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.WaveletPyramid(im, filter_name='qmf13', edge_type='circular')
        res = pyr.recon_pyr()
        # print("\n%s\n" % np.max(res-im))
        self.assertTrue(np.allclose(res, im, atol=2))
    def testRecon7(self):
        im = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.WaveletPyramid(im, filter_name='haar', edge_type='circular')
        res = pyr.recon_pyr()
        self.assertTrue(np.allclose(res, im))

class spFilterTests(unittest.TestCase):
    def test1(self):
        matFilt0 = scipy.io.loadmat(op.join(matfiles_path, 'sp0Filters.mat'))
        pySP0filt = pt.steerable_filters('sp0_filters')
        tmpKeys = []
        for key in list(matFilt0.keys()):
            if "_" not in key:
                tmpKeys.append(key)
        self.assertTrue(set(tmpKeys) == set(pySP0filt.keys()))
        for key in tmpKeys:
            self.assertTrue((matFilt0[key] == pySP0filt[key]).all())

    def test2(self):
        matFilt1 = scipy.io.loadmat(op.join(matfiles_path, 'sp1Filters.mat'))
        pySP1filt = pt.steerable_filters('sp1_filters')
        tmpKeys = []
        for key in list(matFilt1.keys()):
            if "_" not in key:
                tmpKeys.append(key)
        self.assertTrue(set(tmpKeys) == set(pySP1filt.keys()))
        for key in tmpKeys:
            self.assertTrue((matFilt1[key] == pySP1filt[key]).all())

    def test3(self):
        matFilt3 = scipy.io.loadmat(op.join(matfiles_path, 'sp3Filters.mat'))
        pySP3filt = pt.steerable_filters('sp3_filters')
        tmpKeys = []
        for key in list(matFilt3.keys()):
            if "_" not in key:
                tmpKeys.append(key)
        self.assertTrue(set(tmpKeys) == set(pySP3filt.keys()))
        for key in tmpKeys:
            self.assertTrue((matFilt3[key] == pySP3filt[key]).all())

    def test4(self):
        matFilt5 = scipy.io.loadmat(op.join(matfiles_path, 'sp5Filters.mat'))
        pySP5filt = pt.steerable_filters('sp5_filters')
        tmpKeys = []
        for key in list(matFilt5.keys()):
            if "_" not in key:
                tmpKeys.append(key)
        self.assertTrue(set(tmpKeys) == set(pySP5filt.keys()))
        for key in tmpKeys:
            self.assertTrue((matFilt5[key] == pySP5filt[key]).all())

class SteerablePyramidSpaceTests(unittest.TestCase):
    def test00(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr00.mat'))
        pyRamp = pt.synthetic_images.ramp((20,20))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr1.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.SteerablePyramidSpace(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr2.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr3.mat'))
        pyRamp = pt.synthetic_images.ramp((100,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr4.mat'))
        pyRamp = pt.synthetic_images.ramp((200,100))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr0.mat'))
        pyRamp = pt.synthetic_images.ramp(20)
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr5.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.SteerablePyramidSpace(img)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr6.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr7.mat'))
        pyRamp = pt.synthetic_images.ramp((256,128))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr8.mat'))
        pyRamp = pt.synthetic_images.ramp((128,256))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr9.mat'))
        pyRamp = pt.synthetic_images.ramp((200,100))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr10.mat'))
        pyRamp = pt.synthetic_images.ramp((100,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr11.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr(1, 'reflect1', [0])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr12.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr(1, 'reflect1',
                                ['residual_highpass', 1, 3])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr13.mat'))
        pyRamp = pt.synthetic_images.ramp((20,20))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=1, order=0)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test14(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr14.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=3, order=0)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test15(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr15.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=1, order=1)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test16(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr16.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=3, order=1)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test17(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr17.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=1, order=3)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test18(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr18.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=3, order=3)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test19(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr19.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=1, order=5)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test20(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr20.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=3, order=5)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test21(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr21.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:256,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, order=0)
        recon = pyPyr.recon_pyr(0);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test22(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr22.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:256,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, order=1)
        recon = pyPyr.recon_pyr(1);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test23(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr23.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:256,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, order=3)
        recon = pyPyr.recon_pyr(3);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test24(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr24.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:256,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, order=5)
        recon = pyPyr.recon_pyr(5);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test25(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr25.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:256,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, order=5)
        recon = pyPyr.recon_pyr(5,'reflect1', ['residual_highpass', 0, 1], [0]);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test26(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr26.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:128,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, order=0)
        recon = pyPyr.recon_pyr(0,'reflect1', ['residual_highpass', 0, 1], [0]);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test27(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr27.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:128,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, order=1)
        recon = pyPyr.recon_pyr(1,'reflect1', ['residual_highpass', 0, 1], [0]);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test28(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr28.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:128,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, order=3)
        recon = pyPyr.recon_pyr(3,'reflect1', ['residual_highpass', 0, 1], [0]);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test29(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr29.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:128,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, order=5)
        recon = pyPyr.recon_pyr(5,'reflect1', ['residual_highpass', 0, 1], [0]);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))

class SteerablePyramidFreqpyrTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr0.mat'))
        pyRamp = pt.synthetic_images.ramp((20,20))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr1.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.SteerablePyramidFreq(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr2.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr3.mat'))
        pyRamp = pt.synthetic_images.ramp((100,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr4.mat'))
        pyRamp = pt.synthetic_images.ramp((200,100))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr5.mat'))
        pyRamp = pt.synthetic_images.ramp((20,20))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr6.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.SteerablePyramidFreq(img)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr7.mat'))
        pyRamp = pt.synthetic_images.ramp((256,128))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr8.mat'))
        pyRamp = pt.synthetic_images.ramp((128,256))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr9.mat'))
        pyRamp = pt.synthetic_images.ramp((200,100))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr10.mat'))
        pyRamp = pt.synthetic_images.ramp((100,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr11.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.recon_pyr(['residual_highpass'])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr12.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.recon_pyr(['residual_highpass', 1, 3])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr13.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.recon_pyr(['residual_highpass', 1, 3], [1])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def testRecon1(self):
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.SteerablePyramidFreq(img)
        recon = pyr.recon_pyr()
        self.assertTrue(np.allclose(img, recon, atol=5e-3))
    # we initially had a bug with odd orientations, since fixed
    def testRecon2(self):
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.SteerablePyramidFreq(img, order=2)
        recon = pyr.recon_pyr()
        self.assertTrue(np.allclose(img, recon, atol=5e-3))

class SteerablePyramidComplexTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr0.mat'))
        pyRamp = pt.synthetic_images.ramp((20,20))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr1.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.SteerablePyramidFreq(img, is_complex=True)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr2.mat'))
        pyRamp = pt.synthetic_images.ramp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr3.mat'))
        pyRamp = pt.synthetic_images.ramp((100,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr4.mat'))
        pyRamp = pt.synthetic_images.ramp((200,100))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr5.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.SteerablePyramidFreq(img, is_complex=True)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr6.mat'))
        pyRamp = pt.synthetic_images.ramp((256,128))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr7.mat'))
        pyRamp = pt.synthetic_images.ramp((128,256))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr10.mat'))
        pyRamp = pt.synthetic_images.ramp((256,256))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        recon = pyPyr.recon_pyr(['residual_highpass'])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr11.mat'))
        pyRamp = pt.synthetic_images.ramp((256,256))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        recon = pyPyr.recon_pyr(['residual_highpass', 1, 3])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr12.mat'))
        pyRamp = pt.synthetic_images.ramp((256,256))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        recon = pyPyr.recon_pyr(['residual_highpass', 1, 3], [1])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def testRecon1(self):
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.SteerablePyramidFreq(img, is_complex=True)
        recon = pyr.recon_pyr()
        self.assertTrue(np.allclose(img, recon, atol=5e-3))
    # we initially had a bug with odd orientations, since fixed
    def testRecon2(self):
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.SteerablePyramidFreq(img, order=2, is_complex=True)
        recon = pyr.recon_pyr()
        self.assertTrue(np.allclose(img, recon, atol=5e-3))

class mkAngularSineTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkAngularSine0.mat'))
        res = pt.synthetic_images.angular_sine(20)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkAngularSine1.mat'))
        res = pt.synthetic_images.angular_sine(20, 5)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkAngularSine2.mat'))
        res = pt.synthetic_images.angular_sine(20, 5, 3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkAngularSine3.mat'))
        res = pt.synthetic_images.angular_sine(20, 5, 3, 2)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkAngularSine4.mat'))
        res = pt.synthetic_images.angular_sine(20, 5, 3, 2, (2,2))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

class mkGaussianTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkGaussian0.mat'))
        res = pt.synthetic_images.gaussian(20)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkGaussian1.mat'))
        res = pt.synthetic_images.gaussian(20, (2,3))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkGaussian2.mat'))
        res = pt.synthetic_images.gaussian(20, [[-1, 0], [0, 1]])
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkGaussian3.mat'))
        res = pt.synthetic_images.gaussian(10, [[-1, 0], [0, 1]])
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkGaussian4.mat'))
        res = pt.synthetic_images.gaussian(20, [[2, 0], [0, 1]])
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

class mkDiscTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkDisc0.mat'))
        res = pt.synthetic_images.disk(20)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkDisc1.mat'))
        res = pt.synthetic_images.disk(20, 8)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkDisc2.mat'))
        res = pt.synthetic_images.disk(20, 8, (0,0))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkDisc3.mat'))
        res = pt.synthetic_images.disk(20, 8, (0,0), 5)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkDisc4.mat'))
        res = pt.synthetic_images.disk(20, 8, (0,0), 5, (0.75, 0.25))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

class mkSineTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine0.mat'))
        res = pt.synthetic_images.sine(20, period=5.5)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine1.mat'))
        res = pt.synthetic_images.sine(20, period=5.5, direction=2)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine2.mat'))
        res = pt.synthetic_images.sine(20, period=5.5, direction=2, amplitude=3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine3.mat'))
        res = pt.synthetic_images.sine(20, period=5.5, direction=2, amplitude=3, phase=5)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine4.mat'))
        res = pt.synthetic_images.sine(20, period=5.5, direction=2, amplitude=3, phase=5, origin=[4,5])
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine5.mat'))
        res = pt.synthetic_images.sine(20, frequency=[1,2])
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine6.mat'))
        res = pt.synthetic_images.sine(20, frequency=[1,2], amplitude=3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine7.mat'))
        res = pt.synthetic_images.sine(20, frequency=[1,2], amplitude=3, phase=2)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine8.mat'))
        res = pt.synthetic_images.sine(20, frequency=[1,2], amplitude=3, phase=2, origin=[5,4])
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

class mkZonePlateTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkZonePlate0.mat'))
        res = pt.synthetic_images.zone_plate(20)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkZonePlate1.mat'))
        res = pt.synthetic_images.zone_plate(20, 4)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkZonePlate2.mat'))
        res = pt.synthetic_images.zone_plate(20, 4, 3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

class mkSquareTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare0.mat'))
        res = pt.synthetic_images.square_wave(20, period=5.5)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare1.mat'))
        res = pt.synthetic_images.square_wave(20, period=5.5, direction=3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare2.mat'))
        res = pt.synthetic_images.square_wave(20, period=5.5, direction=3, amplitude=5.1)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare3.mat'))
        res = pt.synthetic_images.square_wave(20, period=5.5, direction=3, amplitude=5.1, phase=-1)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare4.mat'))
        res = pt.synthetic_images.square_wave(20, period=5.5, direction=3, amplitude=5.1, phase=-1, origin=(2,3))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare5.mat'))
        res = pt.synthetic_images.square_wave(20, period=5.5, direction=3, amplitude=5.1, phase=-1, origin=(2,3), twidth=.25)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare6.mat'))
        res = pt.synthetic_images.square_wave(20, frequency=(1,2))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare7.mat'))
        res = pt.synthetic_images.square_wave(20, frequency=(1,2), amplitude=3.2)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare8.mat'))
        res = pt.synthetic_images.square_wave(20, frequency=(1,2), amplitude=3.2, phase=-2)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test9(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare9.mat'))
        res = pt.synthetic_images.square_wave(20, frequency=(1,2), amplitude=3.2, phase=-2, origin=(2,3))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare10.mat'))
        res = pt.synthetic_images.square_wave(20, frequency=(1,2), amplitude=3.2, phase=-2, origin=(2,3), twidth=.55)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

# TODO

# python version of histo
# adding 0.7 to ramp to nullify rounding differences between Python and Matlab

# class histoTests(unittest.TestCase):
#     def test0(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'histo0.mat'))
#         (N,X) = pt.matlab_histo(pt.synthetic_images.ramp(10) + 0.7)
#         # X will not be the same because matlab returns centers and
#         #   python return edges
#         #self.assertTrue(pt.compareRecon(matPyr['X'], X))
#         #self.assertTrue(pt.compareRecon(matPyr['N'], N))
#         self.assertTrue((matPyr['N'] == N).all())

# FIX: why does matlab version return N+1 bins??
#    def test1(self):
#        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'histo1.mat'))
#        (N,X) = pt.histo(pt.synthetic_images.ramp(10) + 0.7, 26)
#        # X will not be the same because matlab returns centers and
#        #   python return edges
#        #self.assertTrue(pt.compareRecon(matPyr['X'], X))
#        #self.assertTrue(pt.compareRecon(matPyr['N'], N))
#        self.assertTrue(matPyr['N'] == N)
#    def test2(self):
#        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'histo2.mat'))
#        (N,X) = pt.histo(pt.synthetic_images.ramp(10) + 0.7, -1.15)
#        # X will not be the same because matlab returns centers and
#        #   python return edges
#        #self.assertTrue(pt.compareRecon(matPyr['X'], X))
#        self.assertTrue(pt.compareRecon(matPyr['N'], N))
#    def test3(self):
#        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'histo3.mat'))
#        (N,X) = pt.histo(pt.synthetic_images.ramp(10) + 0.7, 26, 3)
#        # X will not be the same because matlab returns centers and
#        #   python return edges
#        #self.assertTrue(pt.compareRecon(matPyr['X'], X))
#        self.assertTrue(pt.compareRecon(matPyr['N'], N))

# class entropy2Tests(unittest.TestCase):
#    # def test0(self):
#    #     matPyr = scipy.io.loadmat(op.join(matfiles_path, 'entropy2_0.mat'))
#    #     H = pt.entropy(pt.synthetic_images.ramp(10))
#    #     self.assertTrue(matPyr['H'] == H)
#
#    def test1(self):
#        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'entropy2_1.mat'))
#        H = pt.entropy(pt.synthetic_images.ramp(10), 1)
#        self.assertTrue(matPyr['H'] == H)

class ImageGradientTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'imGradient0.mat'))
        ramp = pt.synthetic_images.ramp(10)
        [dx,dy] = pt.image_gradient(ramp)
        dx = np.array(dx)
        dy = np.array(dy)
        self.assertTrue(pt.compareRecon(matPyr['res'][:,:,0], dx))
        self.assertTrue(pt.compareRecon(matPyr['res'][:,:,1], dy))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'imGradient1.mat'))
        ramp = pt.synthetic_images.ramp(10)
        [dx,dy] = pt.image_gradient(ramp, 'reflect1')
        dx = np.array(dx)
        dy = np.array(dy)
        self.assertTrue(pt.compareRecon(matPyr['res'][:,:,0], dx))
        self.assertTrue(pt.compareRecon(matPyr['res'][:,:,1], dy))

class skewTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'skew2_0.mat'))
        # not sure why matPyr is [[ans]]???
        mres = matPyr['res'][0][0]
        disc = pt.synthetic_images.disk(10)
        res = pt.skew(disc)
        self.assertTrue(np.absolute(res - mres) <= np.power(10.0,-11))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'skew2_1.mat'))
        # not sure why matPyr is [[ans]]???
        mres = matPyr['res'][0][0]
        disc = pt.synthetic_images.disk(10)
        # using incorrect mean for better test
        mn = disc.mean() + 0.1
        res = pt.skew(disc, mn)
        self.assertTrue(np.absolute(res - mres) <= np.power(10.0,-11))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'skew2_2.mat'))
        # not sure why matPyr is [[ans]]???
        mres = matPyr['res'][0][0]
        disc = pt.synthetic_images.disk(10)
        # using incorrect mean for better test
        mn = disc.mean() + 0.1
        v = pt.var(disc) + 0.1
        res = pt.skew(disc, mn, v)
        self.assertTrue(np.absolute(res - mres) <= np.power(10.0,-11))


class ProjectPolarTests(unittest.TestCase):
    def test0(self):
        # check that it runs, even though here the output will be nonsensical
        img = pt.synthetic_images.disk(256)
        proj_img = pt.project_polar_to_cartesian(img)
    def test1(self):
        # currently only works for square images
        img = pt.synthetic_images.disk((256, 512))
        with self.assertRaises(Exception):
            pt.project_polar_to_cartesian(img)


# class cconv2Tests(unittest.TestCase):
#     def test0(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_0.mat'))
#         res = pt.cconv2(pt.synthetic_images.ramp(20), pt.synthetic_images.ramp(10))
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test1(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_1.mat'))
#         res = pt.cconv2(pt.synthetic_images.ramp(10), pt.synthetic_images.ramp(20))
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test2(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_2.mat'))
#         res = pt.cconv2(pt.synthetic_images.ramp(20), pt.synthetic_images.ramp(10), 3)
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test3(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_3.mat'))
#         res = pt.cconv2(pt.synthetic_images.ramp(10), pt.synthetic_images.ramp(20), 3)
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test4(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_4.mat'))
#         res = pt.cconv2(pt.synthetic_images.ramp((20,30)), pt.synthetic_images.ramp((10,20)))
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test5(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_5.mat'))
#         res = pt.cconv2(pt.synthetic_images.ramp((10,20)), pt.synthetic_images.ramp((20,30)))
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test6(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_6.mat'))
#         res = pt.cconv2(pt.synthetic_images.ramp((20,30)), pt.synthetic_images.ramp((10,20)), 5)
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test7(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_7.mat'))
#         res = pt.cconv2(pt.synthetic_images.ramp((10,20)), pt.synthetic_images.ramp((20,30)), 5)
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))

# class zconv2Tests(unittest.TestCase):
#     def test0(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'zconv2_0.mat'))
#         mres = matPyr['res']
#         ramp = pt.synthetic_images.ramp(10)
#         disc = pt.synthetic_images.disk(5)
#         res = pt.zconv2(ramp, disc)
#         self.assertTrue(pt.compareRecon(mres, res))
#     def test1(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'zconv2_1.mat'))
#         mres = matPyr['res']
#         ramp = pt.synthetic_images.ramp((10,20))
#         disc = pt.synthetic_images.disk((5,10))
#         res = pt.zconv2(ramp, disc)
#         self.assertTrue(pt.compareRecon(mres, res))
#     def test2(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'zconv2_2.mat'))
#         mres = matPyr['res']
#         ramp = pt.synthetic_images.ramp((20,10))
#         disc = pt.synthetic_images.disk((10,5))
#         res = pt.zconv2(ramp, disc)
#         self.assertTrue(pt.compareRecon(mres, res))
#     def test3(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'zconv2_3.mat'))
#         mres = matPyr['res']
#         ramp = pt.synthetic_images.ramp(10)
#         disc = pt.synthetic_images.disk(5)
#         res = pt.zconv2(ramp, disc, 3)
#         self.assertTrue(pt.compareRecon(mres, res))
#     def test4(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'zconv2_4.mat'))
#         mres = matPyr['res']
#         ramp = pt.synthetic_images.ramp((10,20))
#         disc = pt.synthetic_images.disk((5,10))
#         res = pt.zconv2(ramp, disc, 3)
#         self.assertTrue(pt.compareRecon(mres, res))
#     def test5(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'zconv2_5.mat'))
#         mres = matPyr['res']
#         ramp = pt.synthetic_images.ramp((20,10))
#         disc = pt.synthetic_images.disk((10,5))
#         res = pt.zconv2(ramp, disc, 3)
#         self.assertTrue(pt.compareRecon(mres, res))

class TestImshow(unittest.TestCase):

    def test_imshow0(self):
        im = np.random.rand(3, 10, 10)
        fig = pt.imshow([i for i in im])
        assert len(fig.axes) == 3, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow0_fail(self):
        im = np.random.rand(3, 10, 10)
        with self.assertRaises(Exception):
            # because we now need to pass a list of arrays for multiple images
            fig = pt.imshow(im)

    def test_imshow1(self):
        im = np.random.rand(3, 10, 10, 4)
        fig = pt.imshow([i for i in im])
        assert len(fig.axes) == 3, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow1_fails0(self):
        im = np.random.rand(3, 10, 10, 4)
        with self.assertRaises(Exception):
            # because we now need to pass a list of arrays for multiple images
            fig = pt.imshow(im)

    def test_imshow2(self):
        im = np.random.rand(3, 10, 10, 4)
        fig = pt.imshow(im[0])
        assert len(fig.axes) == 1, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow3(self):
        im = np.random.rand(3, 10, 10, 4)
        fig = pt.imshow([im[0, 0], im[0, 1]])
        assert len(fig.axes) == 2, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow4(self):
        im = np.random.rand(10, 10, 4)
        im2 = np.random.rand(5, 5, 4)
        fig = pt.imshow([im, im2])
        assert len(fig.axes) == 2, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow5(self):
        im = np.random.rand(2, 10, 10, 4)
        fig = pt.imshow([im[0], im[1]])
        assert len(fig.axes) == 2, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow6(self):
        im = np.random.rand(10, 10, 4)
        fig = pt.imshow([i for i in im])
        assert len(fig.axes) == 10, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow7(self):
        im = np.random.rand(10, 10)
        im2 = np.random.rand(10, 10, 4)
        fig = pt.imshow([im, im2])
        assert len(fig.axes) == 2, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow8(self):
        im = np.random.rand(2, 10, 10, 4)
        fig = pt.imshow([im[0,...,0], im[0,...,1]])
        assert len(fig.axes) == 2, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow9(self):
        im = np.random.rand(2, 10, 10, 4)
        fig = pt.imshow([im[0,...,0], im[0,...,1]])
        assert len(fig.axes) == 2, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow10(self):
        im = np.random.rand(10, 10)
        im2 = np.random.rand(5, 5, 4)
        fig = pt.imshow([im, im2])
        assert len(fig.axes) == 2, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow11(self):
        im = np.random.rand(5, 5)
        im2 = np.random.rand(10, 10, 4)
        fig = pt.imshow([im, im2])
        assert len(fig.axes) == 2, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow12(self):
        im = np.random.rand(3, 10, 10, 4)
        with self.assertRaises(Exception):
            # no longer support 4d arrays
            fig = pt.imshow(im)

    def test_imshow13(self):
        im = np.random.rand(10, 10, 3)
        im2 = np.random.rand(10, 10, 4)
        fig = pt.imshow([im, im2])
        assert len(fig.axes) == 2, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow14(self):
        im = np.random.rand(5, 5, 3)
        im2 = np.random.rand(10, 10, 4)
        fig = pt.imshow([im, im2])
        assert len(fig.axes) == 2, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_imshow15(self):
        # don't support 1d arrays
        im = np.random.rand(10)
        with self.assertRaises(Exception):
            fig = pt.imshow(im)

    def test_imshow16(self):
        # must be an array or list of them
        im = 10
        with self.assertRaises(TypeError):
            fig = pt.imshow(im)

    def test_imshow17(self):
        im = np.random.rand(256, 256) + 1j * np.random.rand(256, 256)
        fig = pt.imshow(im)
        assert len(fig.axes) == 2, "Created wrong number of axes!"

    def test_imshow18(self):
        im = np.random.rand(2, 256, 256, 3) + 1j * np.random.rand(2, 256, 256, 3)
        for c in ['rectangular', 'polar', 'logpolar']:
            fig = pt.imshow([i for i in im], plot_complex=c)
            assert len(fig.axes) == 4, "Created wrong number of axes!"

    def test_imshow19(self):
        im = np.random.randn(2, 32, 32)
        for rng in ['auto', 'indep']:
            for i in range(4):
                vrange = rng + str(i)
                pt.imshow(list(im), vrange=vrange)

class TestAnimshow(unittest.TestCase):

    def test_animshow0(self):
        vid = np.random.randn(3, 10, 10, 10)
        fig = pt.animshow([i for i in vid], as_html5=False)._fig
        assert len(fig.axes) == 3, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_animshow1(self):
        vid = np.random.rand(3, 10, 10, 10, 4)
        fig = pt.animshow(vid[0], as_html5=False)._fig
        assert len(fig.axes) == 1, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_animshow2(self):
        vid1 = np.random.rand(10, 10, 10)
        vid2 = np.random.rand(10, 5, 5)
        fig = pt.animshow([vid1, vid2], as_html5=False)._fig
        assert len(fig.axes) == 2, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_animshow3(self):
        vid1 = np.random.rand(3, 10, 10, 10)
        vid2 = np.random.rand(2, 10, 5, 5)
        fig = pt.animshow([v for v in vid1] + [v for v in vid2], as_html5=False)._fig
        assert len(fig.axes) == 5, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_animshow4(self):
        vid1 = np.random.rand(10, 10, 10, 4)
        vid2 = np.random.rand(10, 5, 5, 4)
        fig = pt.animshow([vid1, vid2], as_html5=False)._fig
        assert len(fig.axes) == 2, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_animshow5(self):
        vid = np.random.rand(6, 10, 10, 10, 4)
        fig = pt.animshow([v for v in vid], col_wrap=3, as_html5=False)._fig
        assert len(fig.axes) == 6, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_animshow6(self):
        vid = np.random.randn(2, 10, 10, 10) +\
              1j * np.random.randn(2, 10, 10, 10)
        fig = pt.animshow([v for v in vid], plot_complex='polar',
                          col_wrap=2, as_html5=False)._fig
        assert len(fig.axes) == 4, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_animshow7(self):
        vid = np.random.randn(2, 32, 32, 32, 4) +\
              1j * np.random.randn(2, 32, 32, 32, 4)
        fig = pt.animshow([v for v in vid], framerate=24,
                          plot_complex='logpolar', col_wrap=2, zoom=8,
                          title=[None, 'hello'], as_html5=False)._fig
        assert len(fig.axes) == 4, "Created wrong number of axes! Probably plotting color as grayscale or vice versa"

    def test_animshow_fail_n_frames(self):
        # raises exception because two videos have different numbers of frames
        vid1 = np.random.rand(10, 10, 10)
        vid2 = np.random.rand(5, 10, 10)
        with self.assertRaises(Exception):
            fig = pt.animshow([vid1, vid2], as_html5=False)._fig

def main():
    unittest.main()

if __name__ == '__main__':
    if not os.path.exists(matfiles_path):
        print("matfiles required for testing not found, downloading now...")
        # Streaming, so we can iterate over the response.
        r = requests.get("https://osf.io/cbux8/download", stream=True)

        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024*1024
        wrote = 0
        with open(matfiles_path + ".tar.gz", 'wb') as f:
            for data in tqdm.tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size),
                                  unit='MB', unit_scale=True):
                wrote = wrote  + len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            raise Exception("Error downloading test matfiles!")
        with tarfile.open(matfiles_path + ".tar.gz") as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, os.path.dirname(matfiles_path))
        os.remove(matfiles_path + ".tar.gz")
    main()
