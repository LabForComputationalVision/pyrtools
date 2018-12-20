import unittest
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os.path as op
import pyrtools as pt
from pyrtools.pyramids.pyramid import Pyramid

matfiles_path = op.join(op.dirname(op.realpath(__file__)), 'matFiles')
test_data_path = op.join(op.dirname(op.realpath(__file__)), '..', 'DATA')

class maxPyrHtTests(unittest.TestCase):
    def test1(self):
        pyr = Pyramid(image=[0.0], edge_type=None)
        self.assertTrue(pyr.maxPyrHt((1,10),(3,4)) == 0)
    def test2(self):
        pyr = Pyramid(image=[0.0], edge_type=None)
        self.assertTrue(pyr.maxPyrHt((10,1),(3,4)) == 0)
    def test3(self):
        pyr = Pyramid(image=[0.0], edge_type=None)
        self.assertTrue(pyr.maxPyrHt((10,10),(1,4)) == 2)
    def test4(self):
        pyr = Pyramid(image=[0.0], edge_type=None)
        self.assertTrue(pyr.maxPyrHt((10,10),(3,1)) == 2)
    def test5(self):
        pyr = Pyramid(image=[0.0], edge_type=None)
        self.assertTrue(pyr.maxPyrHt((10,10),(3,4)) == 2)
    def test6(self):
        pyr = Pyramid(image=[0.0], edge_type=None)
        self.assertTrue(pyr.maxPyrHt((20,10),(5,1)) == 2)
    def test7(self):
        pyr = Pyramid(image=[0.0], edge_type=None)
        self.assertTrue(pyr.maxPyrHt((10,20),(5,1)) == 2)
    def test8(self):
        pyr = Pyramid(image=[0.0], edge_type=None)
        self.assertTrue(pyr.maxPyrHt((20,10),(1,5)) == 2)
    def test9(self):
        pyr = Pyramid(image=[0.0], edge_type=None)
        self.assertTrue(pyr.maxPyrHt((10,20),(1,5)) == 2)
    def test10(self):
        pyr = Pyramid(image=[0.0], edge_type=None)
        self.assertTrue(pyr.maxPyrHt((256,1),(1,5)) == 6)
    def test11(self):
        pyr = Pyramid(image=[0.0], edge_type=None)
        self.assertTrue(pyr.maxPyrHt((256,1),(5,1)) == 6)
    def test12(self):
        pyr = Pyramid(image=[0.0], edge_type=None)
        self.assertTrue(pyr.maxPyrHt((1,256),(1,5)) == 6)
    def test13(self):
        pyr = Pyramid(image=[0.0], edge_type=None)
        self.assertTrue(pyr.maxPyrHt((1,256),(5,1)) == 6)

class binomialFilterTests(unittest.TestCase):
    def test1(self):
        target = np.array([[0.5],[0.5]])
        #target = target / np.sqrt(np.sum(target ** 2))
        self.assertTrue((pt.binomialFilter(2) == target).all() )
    def test2(self):
        target = np.array([[0.25], [0.5], [0.25]])
        #target = target / np.sqrt(np.sum(target ** 2))
        self.assertTrue((pt.binomialFilter(3) == target).all())
    def test3(self):
        target = np.array([[0.0625], [0.25], [0.3750], [0.25], [0.0625]])
        #target = target / np.sqrt(np.sum(target ** 2))
        self.assertTrue((pt.binomialFilter(5) == target).all())

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
        img = pt.mkRamp(10)
        pyPyr = pt.pyramids.GaussianPyramid(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildGpyr4.mat'))
        img = pt.mkRamp((10,20))
        pyPyr = pt.pyramids.GaussianPyramid(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildGpyr5.mat'))
        img = pt.mkRamp((20, 10))
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
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr2['pyr'], pyPyr))
    def test3(self):
        matPyr2 = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr3.mat'))
        pyRamp = pt.mkRamp((100,200))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr2['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr4.mat'))
        pyRamp = pt.mkRamp(200,100)
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr5.mat'))
        pyRamp = np.array(list(range(200))).reshape(1, 200)
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test_segfault(self):
        # we used to have a segfault happening in this situation; if the filters have the correct
        # shape, that won't happen (if they're (5,1) instead, it will)
        pySig = np.zeros((1, 36))
        pyr = pt.pyramids.LaplacianPyramid(pySig)
        self.assertTrue(pyr.filter1.shape == (1, 5))
        self.assertTrue(pyr.filter2.shape == (1, 5))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr6.mat'))
        pyRamp = np.array(list(range(200)))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr7.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.LaplacianPyramid(img)
        recon = pyPyr.reconPyr()
        self.assertTrue((matPyr['recon'] == recon).all())
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr8.mat'))
        pyRamp = pt.mkRamp(200)
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        recon = pyPyr.reconPyr()
        self.assertTrue((matPyr['recon'] == recon).all())
    def test9(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr9.mat'))
        pyRamp = pt.mkRamp((200,100))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        recon = pyPyr.reconPyr()
        self.assertTrue((matPyr['recon'] == recon).all())
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr10.mat'))
        pyRamp = pt.mkRamp((100,200))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        recon = pyPyr.reconPyr()
        self.assertTrue((matPyr['recon'] == recon).all())
    def test11(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr11.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        recon = pyPyr.reconPyr([1])
        self.assertTrue((matPyr['recon'] == recon).all())
    def test12(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildLpyr12.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.LaplacianPyramid(pyRamp)
        recon = pyPyr.reconPyr([0,2,4])
        self.assertTrue((matPyr['recon'] == recon).all())

class spFilterTests(unittest.TestCase):
    def test1(self):
        matFilt0 = scipy.io.loadmat(op.join(matfiles_path, 'sp0Filters.mat'))
        pySP0filt = pt.steerable_filters('sp0Filters')
        tmpKeys = []
        for key in list(matFilt0.keys()):
            if "_" not in key:
                tmpKeys.append(key)
        self.assertTrue(set(tmpKeys) == set(pySP0filt.keys()))
        for key in tmpKeys:
            self.assertTrue((matFilt0[key] == pySP0filt[key]).all())

    def test2(self):
        matFilt1 = scipy.io.loadmat(op.join(matfiles_path, 'sp1Filters.mat'))
        pySP1filt = pt.steerable_filters('sp1Filters')
        tmpKeys = []
        for key in list(matFilt1.keys()):
            if "_" not in key:
                tmpKeys.append(key)
        self.assertTrue(set(tmpKeys) == set(pySP1filt.keys()))
        for key in tmpKeys:
            self.assertTrue((matFilt1[key] == pySP1filt[key]).all())

    def test3(self):
        matFilt3 = scipy.io.loadmat(op.join(matfiles_path, 'sp3Filters.mat'))
        pySP3filt = pt.steerable_filters('sp3Filters')
        tmpKeys = []
        for key in list(matFilt3.keys()):
            if "_" not in key:
                tmpKeys.append(key)
        self.assertTrue(set(tmpKeys) == set(pySP3filt.keys()))
        for key in tmpKeys:
            self.assertTrue((matFilt3[key] == pySP3filt[key]).all())

    def test4(self):
        matFilt5 = scipy.io.loadmat(op.join(matfiles_path, 'sp5Filters.mat'))
        pySP5filt = pt.steerable_filters('sp5Filters')
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
        pyRamp = pt.mkRamp((20,20))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr1.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.SteerablePyramidSpace(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr2.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr3.mat'))
        pyRamp = pt.mkRamp((100,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr4.mat'))
        pyRamp = pt.mkRamp((200,100))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr0.mat'))
        pyRamp = pt.mkRamp(20)
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
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr7.mat'))
        pyRamp = pt.mkRamp((256,128))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr8.mat'))
        pyRamp = pt.mkRamp((128,256))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr9.mat'))
        pyRamp = pt.mkRamp((200,100))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr10.mat'))
        pyRamp = pt.mkRamp((100,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr11.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr('sp1Filters', 'reflect1', [0])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr12.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp)
        recon = pyPyr.recon_pyr('sp1Filters', 'reflect1',
                                ['residual_highpass', 1, 3])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr13.mat'))
        pyRamp = pt.mkRamp((20,20))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=1, filters='sp0Filters')
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test14(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr14.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=3, filters='sp0Filters')
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test15(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr15.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=1, filters='sp1Filters')
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test16(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr16.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=3, filters='sp1Filters')
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test17(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr17.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=1, filters='sp3Filters')
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test18(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr18.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=3, filters='sp3Filters')
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test19(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr19.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=1, filters='sp5Filters')
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test20(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr20.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidSpace(pyRamp, height=3, filters='sp5Filters')
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test21(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr21.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:256,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, filters='sp0Filters')
        recon = pyPyr.recon_pyr('sp0Filters');
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test22(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr22.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:256,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, filters='sp1Filters')
        recon = pyPyr.recon_pyr('sp1Filters');
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test23(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr23.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:256,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, filters='sp3Filters')
        recon = pyPyr.recon_pyr('sp3Filters');
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test24(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr24.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:256,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, filters='sp5Filters')
        recon = pyPyr.recon_pyr('sp5Filters');
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test25(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr25.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:256,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, filters='sp5Filters')
        recon = pyPyr.recon_pyr('sp5Filters','reflect1', ['residual_highpass', 0, 1], [0]);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test26(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr26.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:128,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, filters='sp0Filters')
        recon = pyPyr.recon_pyr('sp0Filters','reflect1', ['residual_highpass', 0, 1], [0]);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test27(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr27.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:128,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, filters='sp1Filters')
        recon = pyPyr.recon_pyr('sp1Filters','reflect1', ['residual_highpass', 0, 1], [0]);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test28(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr28.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:128,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, filters='sp3Filters')
        recon = pyPyr.recon_pyr('sp3Filters','reflect1', ['residual_highpass', 0, 1], [0]);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test29(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSpyr29.mat'))
        texture = scipy.io.loadmat(op.join(matfiles_path, 'im04-1.mat'))['res'][0:128,0:256]
        pyPyr = pt.pyramids.SteerablePyramidSpace(texture, height=3, filters='sp5Filters')
        recon = pyPyr.recon_pyr('sp5Filters','reflect1', ['residual_highpass', 0, 1], [0]);
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))




class pointOpTests(unittest.TestCase):
    def test1(self):
        matImg = scipy.io.loadmat(op.join(matfiles_path, 'pointOp1.mat'))
        img = pt.mkRamp((200,200))
        filt = np.array([0.2, 0.5, 1.0, 0.4, 0.1]);
        #foo = pointOp(200, 200, img, 5, filt, 0, 1, 0);
        foo = pt.pointOp(img, filt, 0, 1, 0);
        foo = np.reshape(foo,(200,200))
        self.assertTrue((matImg['foo'] == foo).all())

class SteerablePyramidFreqpyrTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr0.mat'))
        pyRamp = pt.mkRamp((20,20))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr1.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.SteerablePyramidFreq(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr2.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr3.mat'))
        pyRamp = pt.mkRamp((100,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr4.mat'))
        pyRamp = pt.mkRamp((200,100))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr5.mat'))
        pyRamp = pt.mkRamp((20,20))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr6.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.SteerablePyramidFreq(img)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr7.mat'))
        pyRamp = pt.mkRamp((256,128))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr8.mat'))
        pyRamp = pt.mkRamp((128,256))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr9.mat'))
        pyRamp = pt.mkRamp((200,100))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr10.mat'))
        pyRamp = pt.mkRamp((100,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr11.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.reconPyr([0])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr12.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.reconPyr([0,2,4])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSFpyr13.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp)
        recon = pyPyr.reconPyr([0,2,4], [1])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))

class SteerablePyramidComplexTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr0.mat'))
        pyRamp = pt.mkRamp((20,20))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr1.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.SteerablePyramidFreq(img, is_complex=True)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr2.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr3.mat'))
        pyRamp = pt.mkRamp((100,200))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr4.mat'))
        pyRamp = pt.mkRamp((200,100))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr5.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.SteerablePyramidFreq(img, is_complex=True)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr6.mat'))
        pyRamp = pt.mkRamp((256,128))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr7.mat'))
        pyRamp = pt.mkRamp((128,256))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    #def test8(self):  # fails in matlab version
    #    matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr8.mat'))
    #    pyRamp = pt.mkRamp((200,100))
    #    pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
    #    recon = pyPyr.reconSteerablePyramidFreq(, is_complex=True)
    #    self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    #def test9(self):
    #    matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr9.mat'))
    #    pyRamp = pt.mkRamp((100,200))
    #    pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
    #    recon = pyPyr.reconSteerablePyramidFreq(, is_complex=True)
    #    self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr10.mat'))
        pyRamp = pt.mkRamp((256,256))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        recon = pyPyr.reconPyr([0])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr11.mat'))
        pyRamp = pt.mkRamp((256,256))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        recon = pyPyr.reconPyr([0,2,4])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildSCFpyr12.mat'))
        pyRamp = pt.mkRamp((256,256))
        pyPyr = pt.pyramids.SteerablePyramidFreq(pyRamp, is_complex=True)
        recon = pyPyr.reconPyr([0,2,4], [1])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))

class WpyrTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr0.mat'))
        pyRamp = pt.mkRamp((20,20))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr1.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.WaveletPyramid(img)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr2.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr3.mat'))
        pyRamp = pt.mkRamp((100,200))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr4.mat'))
        pyRamp = pt.mkRamp((200,100))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr5.mat'))
        img = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyPyr = pt.pyramids.WaveletPyramid(img)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr6.mat'))
        pyRamp = pt.mkRamp((256,128))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr7.mat'))
        pyRamp = pt.mkRamp((128,256))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr8.mat'))
        pyRamp = pt.mkRamp((200,200))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr9.mat'))
        pyRamp = pt.mkRamp((200,100))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr10.mat'))
        pyRamp = pt.mkRamp((100,200))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.reconPyr()
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr11.mat'))
        pyRamp = pt.mkRamp((256,256))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.reconPyr('qmf9', 'reflect1', [0])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr12.mat'))
        pyRamp = pt.mkRamp((256,256))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.reconPyr('qmf9', 'reflect1', [0,2,4])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr13.mat'))
        pyRamp = pt.mkRamp((256,256))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.reconPyr('qmf9', 'reflect1', [0,2,4], [1])
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test14(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr14.mat'))
        pyRamp = pt.mkRamp((256,256))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.reconPyr('qmf8')
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test15(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr15.mat'))
        pyRamp = pt.mkRamp((256,128))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.reconPyr('qmf8')
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test16(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr16.mat'))
        pyRamp = pt.mkRamp((128,256))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        recon = pyPyr.reconPyr('qmf8')
        self.assertTrue(pt.compareRecon(matPyr['recon'], recon))
    def test17(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr17.mat'))
        pyRamp = pt.mkRamp((1,200))
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def test18(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'buildWpyr18.mat'))
        pyRamp = pt.mkRamp((1,200)).T
        pyPyr = pt.pyramids.WaveletPyramid(pyRamp)
        self.assertTrue(pt.comparePyr(matPyr['pyr'], pyPyr))
    def testRecon1(self):
        im = pt.mkFract((1, 64))
        pyr = pt.pyramids.WaveletPyramid(im)
        res = pyr.reconPyr()
        self.assertTrue(np.allclose(res, im, atol=5e-3))
    def testRecon2(self):
        im = pt.mkFract((64, 1))
        pyr = pt.pyramids.WaveletPyramid(im)
        res = pyr.reconPyr()
        self.assertTrue(np.allclose(res, im, atol=5e-3))
    def testRecon3(self):
        im = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.WaveletPyramid(im)
        res = pyr.reconPyr()
        print("\n%s\n" % np.max(res-im))
        self.assertTrue(np.allclose(res, im, atol=5e-3))
    def testRecon4(self):
        im = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.WaveletPyramid(im, edge_type='circular')
        res = pyr.reconPyr()
        print("\n%s\n" % np.max(res-im))
        self.assertTrue(np.allclose(res, im, atol=5e-3))
    def testRecon5(self):
        im = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.WaveletPyramid(im, filt='daub2', edge_type='circular')
        res = pyr.reconPyr()
        print("\n%s\n" % np.max(res-im))
        self.assertTrue(np.allclose(res, im, atol=5e-3))
    def testRecon6(self):
        im = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.WaveletPyramid(im, filt='qmf13', edge_type='circular')
        res = pyr.reconPyr()
        print("\n%s\n" % np.max(res-im))
        self.assertTrue(np.allclose(res, im, atol=5e-3))
    def testRecon7(self):
        im = plt.imread(op.join(test_data_path, 'lenna-256x256.tif'))
        pyr = pt.pyramids.WaveletPyramid(im, filt='haar', edge_type='circular')
        res = pyr.reconPyr()
        self.assertTrue(np.allclose(res, im))

class blurDnTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn0.mat'))
        pyRamp = pt.mkRamp((20,20))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn1.mat'))
        pyRamp = pt.mkRamp((256,256))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn2.mat'))
        pyRamp = pt.mkRamp((256,128))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn3.mat'))
        pyRamp = pt.mkRamp((128,256))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn4.mat'))
        pyRamp = pt.mkRamp((200, 100))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn5.mat'))
        pyRamp = pt.mkRamp((100, 200))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn6.mat'))
        pyRamp = pt.mkRamp((1, 256)).T
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn7.mat'))
        pyRamp = pt.mkRamp((1, 256))
        res = pt.blurDn(pyRamp)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    #def test8(self):  need a 2D filter
    #    matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blurDn8.mat'))
    #    pyRamp = pt.mkRamp((256, 256))
    #    res = pt.blurDn(pyRamp, 2dfilt)
    #    self.assertTrue(pt.compareRecon(matPyr['res'], res))

class mkAngularSineTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkAngularSine0.mat'))
        res = pt.mkAngularSine(20)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkAngularSine1.mat'))
        res = pt.mkAngularSine(20, 5)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkAngularSine2.mat'))
        res = pt.mkAngularSine(20, 5, 3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkAngularSine3.mat'))
        res = pt.mkAngularSine(20, 5, 3, 2)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkAngularSine4.mat'))
        res = pt.mkAngularSine(20, 5, 3, 2, (2,2))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

class mkGaussianTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkGaussian0.mat'))
        res = pt.mkGaussian(20)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkGaussian1.mat'))
        res = pt.mkGaussian(20, (2,3))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkGaussian2.mat'))
        res = pt.mkGaussian(20, [[-1, 0], [0, 1]])
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkGaussian3.mat'))
        res = pt.mkGaussian(10, [[-1, 0], [0, 1]])
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkGaussian4.mat'))
        res = pt.mkGaussian(20, [[2, 0], [0, 1]])
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

class mkDiscTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkDisc0.mat'))
        res = pt.mkDisc(20)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkDisc1.mat'))
        res = pt.mkDisc(20, 8)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkDisc2.mat'))
        res = pt.mkDisc(20, 8, (0,0))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkDisc3.mat'))
        res = pt.mkDisc(20, 8, (0,0), 5)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkDisc4.mat'))
        res = pt.mkDisc(20, 8, (0,0), 5, (0.75, 0.25))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

class mkSineTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine0.mat'))
        res = pt.mkSine(20, period=5.5)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine1.mat'))
        res = pt.mkSine(20, period=5.5, direction=2)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine2.mat'))
        res = pt.mkSine(20, period=5.5, direction=2, amplitude=3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine3.mat'))
        res = pt.mkSine(20, period=5.5, direction=2, amplitude=3, phase=5)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine4.mat'))
        res = pt.mkSine(20, period=5.5, direction=2, amplitude=3, phase=5, origin=[4,5])
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine5.mat'))
        res = pt.mkSine(20, frequency=[1,2])
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine6.mat'))
        res = pt.mkSine(20, frequency=[1,2], amplitude=3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine7.mat'))
        res = pt.mkSine(20, frequency=[1,2], amplitude=3, phase=2)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSine8.mat'))
        res = pt.mkSine(20, frequency=[1,2], amplitude=3, phase=2, origin=[5,4])
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

class mkZonePlateTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkZonePlate0.mat'))
        res = pt.mkZonePlate(20)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkZonePlate1.mat'))
        res = pt.mkZonePlate(20, 4)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkZonePlate2.mat'))
        res = pt.mkZonePlate(20, 4, 3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

class mkSquareTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare0.mat'))
        res = pt.mkSquare(20, period=5.5)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare1.mat'))
        res = pt.mkSquare(20, period=5.5, direction=3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare2.mat'))
        res = pt.mkSquare(20, period=5.5, direction=3, amplitude=5.1)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare3.mat'))
        res = pt.mkSquare(20, period=5.5, direction=3, amplitude=5.1, phase=-1)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare4.mat'))
        res = pt.mkSquare(20, period=5.5, direction=3, amplitude=5.1, phase=-1, origin=(2,3))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare5.mat'))
        res = pt.mkSquare(20, period=5.5, direction=3, amplitude=5.1, phase=-1, origin=(2,3), twidth=.25)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare6.mat'))
        res = pt.mkSquare(20, frequency=(1,2))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare7.mat'))
        res = pt.mkSquare(20, frequency=(1,2), amplitude=3.2)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare8.mat'))
        res = pt.mkSquare(20, frequency=(1,2), amplitude=3.2, phase=-2)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test9(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare9.mat'))
        res = pt.mkSquare(20, frequency=(1,2), amplitude=3.2, phase=-2, origin=(2,3))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'mkSquare10.mat'))
        res = pt.mkSquare(20, frequency=(1,2), amplitude=3.2, phase=-2, origin=(2,3), twidth=.55)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

class blurTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blur0.mat'))
        res = pt.blur(pt.mkRamp(20))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blur1.mat'))
        res = pt.blur(pt.mkRamp(20), 3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blur2.mat'))
        res = pt.blur(pt.mkRamp(20), 3, pt.namedFilter('qmf5'))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blur3.mat'))
        res = pt.blur(pt.mkRamp((20,30)))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blur4.mat'))
        res = pt.blur(pt.mkRamp((20,30)), 3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'blur5.mat'))
        res = pt.blur(pt.mkRamp((20,30)), 3, pt.namedFilter('qmf5'))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

# class cconv2Tests(unittest.TestCase):
#     def test0(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_0.mat'))
#         res = pt.cconv2(pt.mkRamp(20), pt.mkRamp(10))
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test1(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_1.mat'))
#         res = pt.cconv2(pt.mkRamp(10), pt.mkRamp(20))
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test2(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_2.mat'))
#         res = pt.cconv2(pt.mkRamp(20), pt.mkRamp(10), 3)
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test3(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_3.mat'))
#         res = pt.cconv2(pt.mkRamp(10), pt.mkRamp(20), 3)
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test4(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_4.mat'))
#         res = pt.cconv2(pt.mkRamp((20,30)), pt.mkRamp((10,20)))
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test5(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_5.mat'))
#         res = pt.cconv2(pt.mkRamp((10,20)), pt.mkRamp((20,30)))
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test6(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_6.mat'))
#         res = pt.cconv2(pt.mkRamp((20,30)), pt.mkRamp((10,20)), 5)
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))
#     def test7(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'cconv2_7.mat'))
#         res = pt.cconv2(pt.mkRamp((10,20)), pt.mkRamp((20,30)), 5)
#         self.assertTrue(pt.compareRecon(matPyr['res'], res))

class clipTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'clip0.mat'))
        res = pt.clip(pt.mkRamp(20) / pt.mkRamp(20).max())
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'clip1.mat'))
        res = pt.clip(pt.mkRamp(20) / pt.mkRamp(20).max(), 0.3)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'clip2.mat'))
        res = pt.clip(pt.mkRamp(20) / pt.mkRamp(20).max(), (0.3,0.7))
        self.assertTrue(pt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'clip3.mat'))
        res = pt.clip(pt.mkRamp(20) / pt.mkRamp(20).max(), 0.3, 0.7)
        self.assertTrue(pt.compareRecon(matPyr['res'], res))

# python version of histo
# adding 0.7 to ramp to nullify rounding differences between Python and Matlab
class histoTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'histo0.mat'))
        (N,X) = pt.matlab_histo(pt.mkRamp(10) + 0.7)
        # X will not be the same because matlab returns centers and
        #   python return edges
        #self.assertTrue(pt.compareRecon(matPyr['X'], X))
        #self.assertTrue(pt.compareRecon(matPyr['N'], N))
        self.assertTrue((matPyr['N'] == N).all())
# FIX: why does matlab version return N+1 bins??
#    def test1(self):
#        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'histo1.mat'))
#        (N,X) = pt.histo(pt.mkRamp(10) + 0.7, 26)
#        # X will not be the same because matlab returns centers and
#        #   python return edges
#        #self.assertTrue(pt.compareRecon(matPyr['X'], X))
#        #self.assertTrue(pt.compareRecon(matPyr['N'], N))
#        self.assertTrue(matPyr['N'] == N)
#    def test2(self):
#        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'histo2.mat'))
#        (N,X) = pt.histo(pt.mkRamp(10) + 0.7, -1.15)
#        # X will not be the same because matlab returns centers and
#        #   python return edges
#        #self.assertTrue(pt.compareRecon(matPyr['X'], X))
#        self.assertTrue(pt.compareRecon(matPyr['N'], N))
#    def test3(self):
#        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'histo3.mat'))
#        (N,X) = pt.histo(pt.mkRamp(10) + 0.7, 26, 3)
#        # X will not be the same because matlab returns centers and
#        #   python return edges
#        #self.assertTrue(pt.compareRecon(matPyr['X'], X))
#        self.assertTrue(pt.compareRecon(matPyr['N'], N))

#class entropy2Tests(unittest.TestCase):
#    def test0(self):
#        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'entropy2_0.mat'))
#        H = pt.entropy2(pt.mkRamp(10))
#        self.assertTrue(matPyr['H'] == H)
#    def test1(self):
#        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'entropy2_1.mat'))
#        H = pt.entropy2(pt.mkRamp(10), 1)
#        self.assertTrue(matPyr['H'] == H)

# class factorialTests(unittest.TestCase):
#     def test0(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'factorial0.mat'))
#         res = pt.factorial([[1,2],[3,4]])
#         self.assertTrue((matPyr['res'] == res).all())
#     def test1(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'factorial1.mat'))
#         res = pt.factorial(4)
#         self.assertTrue(matPyr['res'] == res)

class histoMatchTests(unittest.TestCase):
   def test0(self):
       matPyr = scipy.io.loadmat(op.join(matfiles_path, 'histoMatch0.mat'))
       # adding 0.7 to get the bins to line up between matlab and python
       # answers between matlab and python may be different,
       #   but not necessarily incorrect.
       # similar to histo above
       # TODO - why?
       ramp = pt.mkRamp(10) + 0.7
       disc = pt.mkDisc(10) + 0.7
       (rN,rX) = pt.matlab_histo(ramp)
       res = pt.histoMatch(disc, rN, rX, 'edges')
       self.assertTrue(pt.compareRecon(matPyr['res'], res))

class imGradientTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'imGradient0.mat'))
        ramp = pt.mkRamp(10)
        [dx,dy] = pt.imGradient(ramp)
        dx = np.array(dx)
        dy = np.array(dy)
        self.assertTrue(pt.compareRecon(matPyr['res'][:,:,0], dx))
        self.assertTrue(pt.compareRecon(matPyr['res'][:,:,1], dy))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'imGradient1.mat'))
        ramp = pt.mkRamp(10)
        [dx,dy] = pt.imGradient(ramp, 'reflect1')
        dx = np.array(dx)
        dy = np.array(dy)
        self.assertTrue(pt.compareRecon(matPyr['res'][:,:,0], dx))
        self.assertTrue(pt.compareRecon(matPyr['res'][:,:,1], dy))

class skew2Tests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'skew2_0.mat'))
        # not sure why matPyr is [[ans]]???
        mres = matPyr['res'][0][0]
        disc = pt.mkDisc(10)
        res = pt.skew2(disc)
        self.assertTrue(np.absolute(res - mres) <= np.power(10.0,-11))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'skew2_1.mat'))
        # not sure why matPyr is [[ans]]???
        mres = matPyr['res'][0][0]
        disc = pt.mkDisc(10)
        # using incorrect mean for better test
        mn = disc.mean() + 0.1
        res = pt.skew2(disc, mn)
        self.assertTrue(np.absolute(res - mres) <= np.power(10.0,-11))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'skew2_2.mat'))
        # not sure why matPyr is [[ans]]???
        mres = matPyr['res'][0][0]
        disc = pt.mkDisc(10)
        # using incorrect mean for better test
        mn = disc.mean() + 0.1
        v = pt.var2(disc) + 0.1
        res = pt.skew2(disc, mn, v)
        self.assertTrue(np.absolute(res - mres) <= np.power(10.0,-11))

class upBlurTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur0.mat'))
        mres = matPyr['res']
        im = pt.mkRamp((1,20))
        res = pt.upBlur(im)
        self.assertTrue(pt.compareRecon(mres, res))
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur1.mat'))
        mres = matPyr['res']
        im = pt.mkRamp((1,20))
        res = pt.upBlur(im.T)
        self.assertTrue(pt.compareRecon(mres, res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur2.mat'))
        mres = matPyr['res']
        im = pt.mkRamp(20)
        res = pt.upBlur(im)
        self.assertTrue(pt.compareRecon(mres, res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur3.mat'))
        mres = matPyr['res']
        im = pt.mkRamp((1,20))
        res = pt.upBlur(im, 3)
        self.assertTrue(pt.compareRecon(mres, res))
    def test4(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur4.mat'))
        mres = matPyr['res']
        im = pt.mkRamp((1,20))
        res = pt.upBlur(im.T, 3)
        self.assertTrue(pt.compareRecon(mres, res))
    def test5(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur5.mat'))
        mres = matPyr['res']
        im = pt.mkRamp(20)
        res = pt.upBlur(im, 3)
        self.assertTrue(pt.compareRecon(mres, res))
    def test6(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur6.mat'))
        mres = matPyr['res']
        im = pt.mkRamp((1,20))
        filt = pt.namedFilter('qmf9')
        res = pt.upBlur(im, 3, filt)
        self.assertTrue(pt.compareRecon(mres, res))
    #def test7(self):   # fails in matlab and python because of dim mismatch
    #    matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur7.mat'))
    #    mres = matPyr['res']
    #    im = pt.mkRamp((1,20))
    #    filt = pt.namedFilter('qmf9')
    #    res = pt.upBlur(im, 3, filt.T)
    #    self.assertTrue(pt.compareRecon(mres, res))
    def test8(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur8.mat'))
        mres = matPyr['res']
        im = pt.mkRamp((1,20))
        filt = pt.namedFilter('qmf9')
        res = pt.upBlur(im.T, 3, filt)
        self.assertTrue(pt.compareRecon(mres, res))
    #def test9(self):  # fails in matlab and python because of dim mismatch
    #    matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur6.mat'))
    #    mres = matPyr['res']
    #    im = pt.mkRamp((1,20))
    #    filt = pt.namedFilter('qmf9')
    #    res = pt.upBlur(im.T, 3, filt.T)
    #    self.assertTrue(pt.compareRecon(mres, res))
    def test10(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur10.mat'))
        mres = matPyr['res']
        im = pt.mkRamp(20)
        filt = pt.mkDisc(3)
        res = pt.upBlur(im, 3, filt)
        self.assertTrue(pt.compareRecon(mres, res))
    def test11(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur11.mat'))
        mres = matPyr['res']
        im = pt.mkRamp((20,10))
        filt = pt.mkDisc((5,3))
        res = pt.upBlur(im, 3, filt)
        self.assertTrue(pt.compareRecon(mres, res))
    def test12(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'upBlur12.mat'))
        mres = matPyr['res']
        im = pt.mkRamp((10,20))
        filt = pt.mkDisc((3,5))
        res = pt.upBlur(im, 3, filt)
        self.assertTrue(pt.compareRecon(mres, res))

# class zconv2Tests(unittest.TestCase):
#     def test0(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'zconv2_0.mat'))
#         mres = matPyr['res']
#         ramp = pt.mkRamp(10)
#         disc = pt.mkDisc(5)
#         res = pt.zconv2(ramp, disc)
#         self.assertTrue(pt.compareRecon(mres, res))
#     def test1(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'zconv2_1.mat'))
#         mres = matPyr['res']
#         ramp = pt.mkRamp((10,20))
#         disc = pt.mkDisc((5,10))
#         res = pt.zconv2(ramp, disc)
#         self.assertTrue(pt.compareRecon(mres, res))
#     def test2(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'zconv2_2.mat'))
#         mres = matPyr['res']
#         ramp = pt.mkRamp((20,10))
#         disc = pt.mkDisc((10,5))
#         res = pt.zconv2(ramp, disc)
#         self.assertTrue(pt.compareRecon(mres, res))
#     def test3(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'zconv2_3.mat'))
#         mres = matPyr['res']
#         ramp = pt.mkRamp(10)
#         disc = pt.mkDisc(5)
#         res = pt.zconv2(ramp, disc, 3)
#         self.assertTrue(pt.compareRecon(mres, res))
#     def test4(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'zconv2_4.mat'))
#         mres = matPyr['res']
#         ramp = pt.mkRamp((10,20))
#         disc = pt.mkDisc((5,10))
#         res = pt.zconv2(ramp, disc, 3)
#         self.assertTrue(pt.compareRecon(mres, res))
#     def test5(self):
#         matPyr = scipy.io.loadmat(op.join(matfiles_path, 'zconv2_5.mat'))
#         mres = matPyr['res']
#         ramp = pt.mkRamp((20,10))
#         disc = pt.mkDisc((10,5))
#         res = pt.zconv2(ramp, disc, 3)
#         self.assertTrue(pt.compareRecon(mres, res))

class corrDnTests(unittest.TestCase):
    def test1(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'corrDn1.mat'))
        mres = matPyr['res']
        ramp = pt.mkRamp(20)
        res = pt.corrDn(ramp, pt.namedFilter('binom5'))
        self.assertTrue(pt.compareRecon(mres, res))
    def test2(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'corrDn2.mat'))
        mres = matPyr['res']
        ramp = pt.mkRamp(20)
        res = pt.corrDn(ramp, pt.namedFilter('qmf13'))
        self.assertTrue(pt.compareRecon(mres, res))
    def test3(self):
        matPyr = scipy.io.loadmat(op.join(matfiles_path, 'corrDn3.mat'))
        mres = matPyr['res']
        ramp = pt.mkRamp(20)
        res = pt.corrDn(ramp, pt.namedFilter('qmf16'))
        self.assertTrue(pt.compareRecon(mres, res))

def main():
    unittest.main()

if __name__ == '__main__':
    main()
