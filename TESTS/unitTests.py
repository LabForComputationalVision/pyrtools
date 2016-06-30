#!/users-local/ryoung/anaconda/bin/python
import unittest
import math
import numpy as np
import scipy.io
import Image
from operator import mul
import sys
sys.path.append('/users/ryoung/programming/python')
import pyPyrTools as ppt

class maxPyrHtTests(unittest.TestCase):
    def test1(self):
        self.failUnless(ppt.maxPyrHt((1,10),(3,4)) == 0)
    def test2(self):
        self.failUnless(ppt.maxPyrHt((10,1),(3,4)) == 0)
    def test3(self):
        self.failUnless(ppt.maxPyrHt((10,10),(1,4)) == 4)
    def test4(self):
        self.failUnless(ppt.maxPyrHt((10,10),(3,1)) == 2)
    def test5(self):
        self.failUnless(ppt.maxPyrHt((10,10),(3,4)) == 2)
    def test6(self):
        self.failUnless(ppt.maxPyrHt((20,10),(5,1)) == 2)
    def test7(self):
        self.failUnless(ppt.maxPyrHt((10,20),(5,1)) == 2)
    def test8(self):
        self.failUnless(ppt.maxPyrHt((20,10),(1,5)) == 5)
    def test9(self):
        self.failUnless(ppt.maxPyrHt((10,20),(1,5)) == 5)
    def test10(self):
        self.failUnless(ppt.maxPyrHt((256,1),(1,5)) == 6)
    def test11(self):
        self.failUnless(ppt.maxPyrHt((256,1),(5,1)) == 6)
    def test12(self):
        self.failUnless(ppt.maxPyrHt((1,256),(1,5)) == 6)
    def test13(self):
        self.failUnless(ppt.maxPyrHt((1,256),(5,1)) == 6)

class binomialFilterTests(unittest.TestCase):
    def test1(self):
        self.failUnless((ppt.binomialFilter(2) == np.array([[0.5],
                                                            [0.5]])).all() )
    def test2(self):
        self.failUnless((ppt.binomialFilter(3) == np.array([[0.25], [0.5],
                                                            [0.25]])).all())
    def test3(self):
        self.failUnless((ppt.binomialFilter(5) == np.array([[0.0625], [0.25],
                                                            [0.3750], [0.25],
                                                            [0.0625]])).all())

class GpyrTests(unittest.TestCase):
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/buildGpyr1.mat')
        img = Image.open('../lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/buildGpyr2row.mat')
        img = np.array(range(256)).astype(float)
        img = img.reshape(1, 256)
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/buildGpyr2col.mat')
        img = np.array(range(256)).astype(float)
        img = img.reshape(256, 1)
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/buildGpyr3.mat')
        img = ppt.mkRamp(10)
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat('../matFiles/buildGpyr4.mat')
        img = ppt.mkRamp((10,20))
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test6(self):
        matPyr = scipy.io.loadmat('../matFiles/buildGpyr5.mat')
        img = ppt.mkRamp((20, 10))
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))

class LpyrTests(unittest.TestCase):
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/buildLpyr1.mat')
        img = Image.open('../lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.Lpyr(img)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self): 
        matPyr2 = scipy.io.loadmat('../matFiles/buildLpyr2.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr2['pyr'], pyPyr))
    def test3(self):
        matPyr2 = scipy.io.loadmat('../matFiles/buildLpyr3.mat')
        pyRamp = ppt.mkRamp((100,200))
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr2['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/buildLpyr4.mat')
        pyRamp = ppt.mkRamp(200,100)
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):   
        matPyr = scipy.io.loadmat('../matFiles/buildLpyr5.mat')
        pyRamp = np.array(range(200)).reshape(1, 200)
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test6(self):
        matPyr = scipy.io.loadmat('../matFiles/buildLpyr6.mat')
        pyRamp = np.array(range(200))
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test7(self):
        matPyr = scipy.io.loadmat('../matFiles/buildLpyr7.mat')
        img = Image.open('../lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.Lpyr(img)
        recon = pyPyr.reconPyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test8(self): 
        matPyr = scipy.io.loadmat('../matFiles/buildLpyr8.mat')
        pyRamp = ppt.mkRamp(200)
        pyPyr = ppt.Lpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test9(self): 
        matPyr = scipy.io.loadmat('../matFiles/buildLpyr9.mat')
        pyRamp = ppt.mkRamp((200,100))
        pyPyr = ppt.Lpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test10(self): 
        matPyr = scipy.io.loadmat('../matFiles/buildLpyr10.mat')
        pyRamp = ppt.mkRamp((100,200))
        pyPyr = ppt.Lpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test11(self): 
        matPyr = scipy.io.loadmat('../matFiles/buildLpyr11.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Lpyr(pyRamp)
        recon = pyPyr.reconPyr([1])
        self.failUnless((matPyr['recon'] == recon).all())
    def test12(self): 
        matPyr = scipy.io.loadmat('../matFiles/buildLpyr12.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Lpyr(pyRamp)
        recon = pyPyr.reconPyr([0,2,4])
        self.failUnless((matPyr['recon'] == recon).all())

class spFilterTests(unittest.TestCase):
    def test1(self):
        matFilt0 = scipy.io.loadmat('../matFiles/sp0Filters.mat')
        pySP0filt = ppt.sp0Filters()
        tmpKeys = []
        for key in matFilt0.keys():
            if "_" not in key:
                tmpKeys.append(key)
        self.failUnless(tmpKeys == pySP0filt.keys())
        for key in tmpKeys:
            self.failUnless((matFilt0[key] == pySP0filt[key]).all())

    def test2(self):
        matFilt1 = scipy.io.loadmat('../matFiles/sp1Filters.mat')
        pySP1filt = ppt.sp1Filters()
        tmpKeys = []
        for key in matFilt1.keys():
            if "_" not in key:
                tmpKeys.append(key)
        self.failUnless(tmpKeys == pySP1filt.keys())
        for key in tmpKeys:
            self.failUnless((matFilt1[key] == pySP1filt[key]).all())

    def test3(self):
        matFilt3 = scipy.io.loadmat('../matFiles/sp3Filters.mat')
        pySP3filt = ppt.sp3Filters()
        tmpKeys = []
        for key in matFilt3.keys():
            if "_" not in key:
                tmpKeys.append(key)
        self.failUnless(tmpKeys == pySP3filt.keys())
        for key in tmpKeys:
            self.failUnless((matFilt3[key] == pySP3filt[key]).all())

    def test4(self):
        matFilt5 = scipy.io.loadmat('../matFiles/sp5Filters.mat')
        pySP5filt = ppt.sp5Filters()
        tmpKeys = []
        for key in matFilt5.keys():
            if "_" not in key:
                tmpKeys.append(key)
        self.failUnless(tmpKeys == pySP5filt.keys())
        for key in tmpKeys:
            self.failUnless((matFilt5[key] == pySP5filt[key]).all())

class SpyrTests(unittest.TestCase):
    def test00(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr00.mat')
        pyRamp = ppt.mkRamp((20,20))
        pyPyr = ppt.Spyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr1.mat')
        img = np.array(Image.open('../lenna-256x256.tif')).astype(float)
        pyPyr = ppt.Spyr(img)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr2.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr3.mat')
        pyRamp = ppt.mkRamp((100,200))
        pyPyr = ppt.Spyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr4.mat')
        pyRamp = ppt.mkRamp((200,100))
        pyPyr = ppt.Spyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr0.mat')
        pyRamp = ppt.mkRamp(20)
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test5(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr5.mat')
        img = np.array(Image.open('../lenna-256x256.tif')).astype(float)
        pyPyr = ppt.Spyr(img)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr6.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr7.mat')
        pyRamp = ppt.mkRamp((256,128))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr8.mat')
        pyRamp = ppt.mkRamp((128,256))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr9.mat')
        pyRamp = ppt.mkRamp((200,100))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr10.mat')
        pyRamp = ppt.mkRamp((100,200))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr11.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr('sp1Filters', 'reflect1', [1])
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr12.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr('sp1Filters', 'reflect1', [0,2,4])
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr13.mat')
        pyRamp = ppt.mkRamp((20,20))
        pyPyr = ppt.Spyr(pyRamp, 1, 'sp0Filters')
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test14(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr14.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 3, 'sp0Filters')
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test15(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr15.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 1, 'sp1Filters')
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test16(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr16.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 3, 'sp1Filters')
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test17(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr17.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 1, 'sp3Filters')
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test18(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr18.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 3, 'sp3Filters')
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test19(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr19.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 1, 'sp5Filters')
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test20(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr20.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 3, 'sp5Filters')
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test21(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr21.mat')
        texture = scipy.io.loadmat('../matFiles/im04-1.mat')['res'][0:256,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp0Filters')
        recon = pyPyr.reconPyr('sp0Filters');
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test22(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr22.mat')
        texture = scipy.io.loadmat('../matFiles/im04-1.mat')['res'][0:256,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp1Filters')
        recon = pyPyr.reconPyr('sp1Filters');
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test23(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr23.mat')
        texture = scipy.io.loadmat('../matFiles/im04-1.mat')['res'][0:256,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp3Filters')
        recon = pyPyr.reconPyr('sp3Filters');
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test24(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr24.mat')
        texture = scipy.io.loadmat('../matFiles/im04-1.mat')['res'][0:256,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp5Filters')
        recon = pyPyr.reconPyr('sp5Filters');
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test25(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr25.mat')
        texture = scipy.io.loadmat('../matFiles/im04-1.mat')['res'][0:256,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp5Filters')
        recon = pyPyr.reconPyr('sp5Filters','reflect1',[0,1,2], [0]);
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test26(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr26.mat')
        texture = scipy.io.loadmat('../matFiles/im04-1.mat')['res'][0:128,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp0Filters')
        recon = pyPyr.reconPyr('sp0Filters','reflect1',[0,1,2], [0]);
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test27(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr27.mat')
        texture = scipy.io.loadmat('../matFiles/im04-1.mat')['res'][0:128,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp1Filters')
        recon = pyPyr.reconPyr('sp1Filters','reflect1',[0,1,2], [0]);
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test28(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr28.mat')
        texture = scipy.io.loadmat('../matFiles/im04-1.mat')['res'][0:128,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp3Filters')
        recon = pyPyr.reconPyr('sp3Filters','reflect1',[0,1,2], [0]);
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test29(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSpyr29.mat')
        texture = scipy.io.loadmat('../matFiles/im04-1.mat')['res'][0:128,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp5Filters')
        recon = pyPyr.reconPyr('sp5Filters','reflect1',[0,1,2], [0]);
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
        
        
        

class pointOpTests(unittest.TestCase):
    def test1(self):
        matImg = scipy.io.loadmat('../matFiles/pointOp1.mat')
        img = ppt.mkRamp((200,200))
        filt = np.array([0.2, 0.5, 1.0, 0.4, 0.1]);
        #foo = pointOp(200, 200, img, 5, filt, 0, 1, 0);
        foo = ppt.pointOp(img, filt, 0, 1, 0);
        foo = np.reshape(foo,(200,200))
        self.failUnless((matImg['foo'] == foo).all())

class SFpyrTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr0.mat')
        pyRamp = ppt.mkRamp((20,20))
        pyPyr = ppt.SFpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr1.mat')
        img = Image.open('../lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.SFpyr(img)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr2.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.SFpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr3.mat')
        pyRamp = ppt.mkRamp((100,200))
        pyPyr = ppt.SFpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr4.mat')
        pyRamp = ppt.mkRamp((200,100))
        pyPyr = ppt.SFpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr5.mat')
        pyRamp = ppt.mkRamp((20,20))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr6.mat')
        img = Image.open('../lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.SFpyr(img)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr7.mat')
        pyRamp = ppt.mkRamp((256,128))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr8.mat')
        pyRamp = ppt.mkRamp((128,256))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr9.mat')
        pyRamp = ppt.mkRamp((200,100))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr10.mat')
        pyRamp = ppt.mkRamp((100,200))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr11.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr([0])
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr12.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr([0,2,4])
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSFpyr13.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr([0,2,4], [1])
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))

class SCFpyrTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSCFpyr0.mat')
        pyRamp = ppt.mkRamp((20,20))
        pyPyr = ppt.SCFpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSCFpyr1.mat')
        img = Image.open('../lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.SCFpyr(img)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSCFpyr2.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.SCFpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSCFpyr3.mat')
        pyRamp = ppt.mkRamp((100,200))
        pyPyr = ppt.SCFpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSCFpyr4.mat')
        pyRamp = ppt.mkRamp((200,100))
        pyPyr = ppt.SCFpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSCFpyr5.mat')
        img = Image.open('../lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.SCFpyr(img)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSCFpyr6.mat')
        pyRamp = ppt.mkRamp((256,128))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSCFpyr7.mat')
        pyRamp = ppt.mkRamp((128,256))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    #def test8(self):  # fails in matlab version
    #    matPyr = scipy.io.loadmat('../matFiles/buildSCFpyr8.mat')
    #    pyRamp = ppt.mkRamp((200,100))
    #    pyPyr = ppt.SCFpyr(pyRamp)
    #    recon = pyPyr.reconSCFpyr()
    #    self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    #def test9(self):
    #    matPyr = scipy.io.loadmat('../matFiles/buildSCFpyr9.mat')
    #    pyRamp = ppt.mkRamp((100,200))
    #    pyPyr = ppt.SCFpyr(pyRamp)
    #    recon = pyPyr.reconSCFpyr()
    #    self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSCFpyr10.mat')
        pyRamp = ppt.mkRamp((256,256))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconPyr([0])
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSCFpyr11.mat')
        pyRamp = ppt.mkRamp((256,256))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconPyr([0,2,4])
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat('../matFiles/buildSCFpyr12.mat')
        pyRamp = ppt.mkRamp((256,256))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconPyr([0,2,4], [1])
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))

class WpyrTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr0.mat')
        pyRamp = ppt.mkRamp((20,20))
        pyPyr = ppt.Wpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr1.mat')
        img = np.array(Image.open('../lenna-256x256.tif')).astype(float)
        pyPyr = ppt.Wpyr(img)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr2.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Wpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr3.mat')
        pyRamp = ppt.mkRamp((100,200))
        pyPyr = ppt.Wpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr4.mat')
        pyRamp = ppt.mkRamp((200,100))
        pyPyr = ppt.Wpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr5.mat')
        img = np.array(Image.open('../lenna-256x256.tif')).astype(float)
        pyPyr = ppt.Wpyr(img)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr6.mat')
        pyRamp = ppt.mkRamp((256,128))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr7.mat')
        pyRamp = ppt.mkRamp((128,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr8.mat')
        pyRamp = ppt.mkRamp((200,200))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr9.mat')
        pyRamp = ppt.mkRamp((200,100))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr10.mat')
        pyRamp = ppt.mkRamp((100,200))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr11.mat')
        pyRamp = ppt.mkRamp((256,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr('qmf9', 'reflect1', [0])
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr12.mat')
        pyRamp = ppt.mkRamp((256,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr('qmf9', 'reflect1', [0,2,4])
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr13.mat')
        pyRamp = ppt.mkRamp((256,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr('qmf9', 'reflect1', [0,2,4], [1])
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test14(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr14.mat')
        pyRamp = ppt.mkRamp((256,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr('qmf8')
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test15(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr15.mat')
        pyRamp = ppt.mkRamp((256,128))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr('qmf8')
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test16(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr16.mat')
        pyRamp = ppt.mkRamp((128,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr('qmf8')
        self.failUnless(ppt.compareRecon(matPyr['recon'], recon))
    def test17(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr17.mat')
        pyRamp = ppt.mkRamp((1,200))
        pyPyr = ppt.Wpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))
    def test18(self):
        matPyr = scipy.io.loadmat('../matFiles/buildWpyr18.mat')
        pyRamp = ppt.mkRamp((1,200)).T
        pyPyr = ppt.Wpyr(pyRamp)
        self.failUnless(ppt.comparePyr(matPyr['pyr'], pyPyr))

class blurDnTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/blurDn0.mat')
        pyRamp = ppt.mkRamp((20,20))
        res = ppt.blurDn(pyRamp)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/blurDn1.mat')
        pyRamp = ppt.mkRamp((256,256))
        res = ppt.blurDn(pyRamp)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/blurDn2.mat')
        pyRamp = ppt.mkRamp((256,128))
        res = ppt.blurDn(pyRamp)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/blurDn3.mat')
        pyRamp = ppt.mkRamp((128,256))
        res = ppt.blurDn(pyRamp)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/blurDn4.mat')
        pyRamp = ppt.mkRamp((200, 100))
        res = ppt.blurDn(pyRamp)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat('../matFiles/blurDn5.mat')
        pyRamp = ppt.mkRamp((100, 200))
        res = ppt.blurDn(pyRamp)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat('../matFiles/blurDn6.mat')
        pyRamp = ppt.mkRamp((1, 256)).T
        res = ppt.blurDn(pyRamp)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat('../matFiles/blurDn7.mat')
        pyRamp = ppt.mkRamp((1, 256))
        res = ppt.blurDn(pyRamp)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    #def test8(self):  need a 2D filter
    #    matPyr = scipy.io.loadmat('../matFiles/blurDn8.mat')
    #    pyRamp = ppt.mkRamp((256, 256))
    #    res = ppt.blurDn(pyRamp, 2dfilt)
    #    self.failUnless(ppt.compareRecon(matPyr['res'], res))

class mkAngularSineTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/mkAngularSine0.mat')
        res = ppt.mkAngularSine(20)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/mkAngularSine1.mat')
        res = ppt.mkAngularSine(20, 5)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/mkAngularSine2.mat')
        res = ppt.mkAngularSine(20, 5, 3)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/mkAngularSine3.mat')
        res = ppt.mkAngularSine(20, 5, 3, 2)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/mkAngularSine4.mat')
        res = ppt.mkAngularSine(20, 5, 3, 2, (2,2))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))

class mkGaussianTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/mkGaussian0.mat')
        res = ppt.mkGaussian(20)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/mkGaussian1.mat')
        res = ppt.mkGaussian(20, (2,3))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/mkGaussian2.mat')
        res = ppt.mkGaussian(20, [[-1, 0], [0, 1]])
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/mkGaussian3.mat')
        res = ppt.mkGaussian(10, [[-1, 0], [0, 1]])
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/mkGaussian4.mat')
        res = ppt.mkGaussian(20, [[2, 0], [0, 1]])
        self.failUnless(ppt.compareRecon(matPyr['res'], res))

class mkDiscTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/mkDisc0.mat')
        res = ppt.mkDisc(20)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/mkDisc1.mat')
        res = ppt.mkDisc(20, 8)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/mkDisc2.mat')
        res = ppt.mkDisc(20, 8, (0,0))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/mkDisc3.mat')
        res = ppt.mkDisc(20, 8, (0,0), 5)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/mkDisc4.mat')
        res = ppt.mkDisc(20, 8, (0,0), 5, (0.75, 0.25))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))

class mkSineTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSine0.mat')
        res = ppt.mkSine(20, 5.5)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSine1.mat')
        res = ppt.mkSine(20, 5.5, 2)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSine2.mat')
        res = ppt.mkSine(20, 5.5, 2, 3)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSine3.mat')
        res = ppt.mkSine(20, 5.5, 2, 3, 5)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSine4.mat')
        res = ppt.mkSine(20, 5.5, 2, 3, 5, [4,5])
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSine5.mat')
        res = ppt.mkSine(20, [1,2])
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSine6.mat')
        res = ppt.mkSine(20, [1,2], 3)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSine7.mat')
        res = ppt.mkSine(20, [1,2], 3, 2)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test8(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSine8.mat')
        res = ppt.mkSine(20, [1,2], 3, 2, [5,4])
        self.failUnless(ppt.compareRecon(matPyr['res'], res))

class mkZonePlateTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/mkZonePlate0.mat')
        res = ppt.mkZonePlate(20)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/mkZonePlate1.mat')
        res = ppt.mkZonePlate(20, 4)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/mkZonePlate2.mat')
        res = ppt.mkZonePlate(20, 4, 3)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))

class mkSquareTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSquare0.mat')
        res = ppt.mkSquare(20, 5.5)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSquare1.mat')
        res = ppt.mkSquare(20, 5.5, 3)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSquare2.mat')
        res = ppt.mkSquare(20, 5.5, 3, 5.1)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSquare3.mat')
        res = ppt.mkSquare(20, 5.5, 3, 5.1, -1)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSquare4.mat')
        res = ppt.mkSquare(20, 5.5, 3, 5.1, -1, (2,3))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSquare5.mat')
        res = ppt.mkSquare(20, 5.5, 3, 5.1, -1, (2,3), 0.25)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSquare6.mat')
        res = ppt.mkSquare(20, (1,2))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSquare7.mat')
        res = ppt.mkSquare(20, (1,2), 3.2)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test8(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSquare8.mat')
        res = ppt.mkSquare(20, (1,2), 3.2, -2)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test9(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSquare9.mat')
        res = ppt.mkSquare(20, (1,2), 3.2, -2, (2,3))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test10(self):
        matPyr = scipy.io.loadmat('../matFiles/mkSquare10.mat')
        res = ppt.mkSquare(20, (1,2), 3.2, -2, (2,3), 0.55)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))

class blurTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/blur0.mat')
        res = ppt.blur(ppt.mkRamp(20))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/blur1.mat')
        res = ppt.blur(ppt.mkRamp(20), 3)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/blur2.mat')
        res = ppt.blur(ppt.mkRamp(20), 3, ppt.namedFilter('qmf5'))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/blur3.mat')
        res = ppt.blur(ppt.mkRamp((20,30)))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/blur4.mat')
        res = ppt.blur(ppt.mkRamp((20,30)), 3)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat('../matFiles/blur5.mat')
        res = ppt.blur(ppt.mkRamp((20,30)), 3, ppt.namedFilter('qmf5'))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))

class cconv2Tests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/cconv2_0.mat')
        res = ppt.cconv2(ppt.mkRamp(20), ppt.mkRamp(10))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/cconv2_1.mat')
        res = ppt.cconv2(ppt.mkRamp(10), ppt.mkRamp(20))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/cconv2_2.mat')
        res = ppt.cconv2(ppt.mkRamp(20), ppt.mkRamp(10), 3)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/cconv2_3.mat')
        res = ppt.cconv2(ppt.mkRamp(10), ppt.mkRamp(20), 3)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/cconv2_4.mat')
        res = ppt.cconv2(ppt.mkRamp((20,30)), ppt.mkRamp((10,20)))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat('../matFiles/cconv2_5.mat')
        res = ppt.cconv2(ppt.mkRamp((10,20)), ppt.mkRamp((20,30)))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat('../matFiles/cconv2_6.mat')
        res = ppt.cconv2(ppt.mkRamp((20,30)), ppt.mkRamp((10,20)), 5)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat('../matFiles/cconv2_7.mat')
        res = ppt.cconv2(ppt.mkRamp((10,20)), ppt.mkRamp((20,30)), 5)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))

class clipTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/clip0.mat')
        res = ppt.clip(ppt.mkRamp(20) / ppt.mkRamp(20).max())
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/clip1.mat')
        res = ppt.clip(ppt.mkRamp(20) / ppt.mkRamp(20).max(), 0.3)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/clip2.mat')
        res = ppt.clip(ppt.mkRamp(20) / ppt.mkRamp(20).max(), (0.3,0.7))
        self.failUnless(ppt.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/clip3.mat')
        res = ppt.clip(ppt.mkRamp(20) / ppt.mkRamp(20).max(), 0.3, 0.7)
        self.failUnless(ppt.compareRecon(matPyr['res'], res))

# python version of histo
# adding 0.7 to ramp to nullify rounding differences between Python and Matlab
class histoTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/histo0.mat')
        (N,X) = ppt.histo(ppt.mkRamp(10) + 0.7)
        # X will not be the same because matlab returns centers and 
        #   python return edges
        #self.failUnless(ppt.compareRecon(matPyr['X'], X))
        #self.failUnless(ppt.compareRecon(matPyr['N'], N))
        self.failUnless((matPyr['N'] == N).all())
# FIX: why does matlab version return N+1 bins??
#    def test1(self):
#        matPyr = scipy.io.loadmat('../matFiles/histo1.mat')
#        (N,X) = ppt.histo(ppt.mkRamp(10) + 0.7, 26)
#        # X will not be the same because matlab returns centers and 
#        #   python return edges
#        #self.failUnless(ppt.compareRecon(matPyr['X'], X))
#        #self.failUnless(ppt.compareRecon(matPyr['N'], N))
#        self.failUnless(matPyr['N'] == N)
#    def test2(self):
#        matPyr = scipy.io.loadmat('../matFiles/histo2.mat')
#        (N,X) = ppt.histo(ppt.mkRamp(10) + 0.7, -1.15)
#        # X will not be the same because matlab returns centers and 
#        #   python return edges
#        #self.failUnless(ppt.compareRecon(matPyr['X'], X))
#        self.failUnless(ppt.compareRecon(matPyr['N'], N))
#    def test3(self):
#        matPyr = scipy.io.loadmat('../matFiles/histo3.mat')
#        (N,X) = ppt.histo(ppt.mkRamp(10) + 0.7, 26, 3)
#        # X will not be the same because matlab returns centers and 
#        #   python return edges
#        #self.failUnless(ppt.compareRecon(matPyr['X'], X))
#        self.failUnless(ppt.compareRecon(matPyr['N'], N))

#class entropy2Tests(unittest.TestCase):
#    def test0(self):
#        matPyr = scipy.io.loadmat('../matFiles/entropy2_0.mat')
#        H = ppt.entropy2(ppt.mkRamp(10))
#        self.failUnless(matPyr['H'] == H)
#    def test1(self):
#        matPyr = scipy.io.loadmat('../matFiles/entropy2_1.mat')
#        H = ppt.entropy2(ppt.mkRamp(10), 1)
#        self.failUnless(matPyr['H'] == H)

class factorialTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/factorial0.mat')
        res = ppt.factorial([[1,2],[3,4]])
        self.failUnless((matPyr['res'] == res).all())
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/factorial1.mat')
        res = ppt.factorial(4)
        self.failUnless(matPyr['res'] == res)

#class histoMatchTests(unittest.TestCase):
#    def test0(self):
#        matPyr = scipy.io.loadmat('../matFiles/histoMatch0.mat')
#        # adding 0.7 to get the bins to line up between matlab and python
#        # answers between matlab and python may be different, 
#        #   but not necessarily incorrect.
#        # similar to histo above
#        ramp = ppt.mkRamp(10) + 0.7
#        disc = ppt.mkDisc(10) + 0.7
#        (rN,rX) = ppt.histo(ramp)
#        res = ppt.histoMatch(disc, rN, rX, 'edges')
#        self.failUnless(ppt.compareRecon(matPyr['res'], res))

class imGradientTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/imGradient0.mat')
        ramp = ppt.mkRamp(10)
        [dx,dy] = ppt.imGradient(ramp)
        dx = np.array(dx)
        dy = np.array(dy)
        self.failUnless(ppt.compareRecon(matPyr['res'][:,:,0], dx))
        self.failUnless(ppt.compareRecon(matPyr['res'][:,:,1], dy))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/imGradient1.mat')
        ramp = ppt.mkRamp(10)
        [dx,dy] = ppt.imGradient(ramp, 'reflect1')
        dx = np.array(dx)
        dy = np.array(dy)
        self.failUnless(ppt.compareRecon(matPyr['res'][:,:,0], dx))
        self.failUnless(ppt.compareRecon(matPyr['res'][:,:,1], dy))
 
class skew2Tests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/skew2_0.mat')
        # not sure why matPyr is [[ans]]???
        mres = matPyr['res'][0][0]
        disc = ppt.mkDisc(10)
        res = ppt.skew2(disc)
        self.failUnless(np.absolute(res - mres) <= math.pow(10,-11))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/skew2_1.mat')
        # not sure why matPyr is [[ans]]???
        mres = matPyr['res'][0][0]
        disc = ppt.mkDisc(10)
        # using incorrect mean for better test
        mn = disc.mean() + 0.1
        res = ppt.skew2(disc, mn)
        self.failUnless(np.absolute(res - mres) <= math.pow(10,-11))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/skew2_2.mat')
        # not sure why matPyr is [[ans]]???
        mres = matPyr['res'][0][0]
        disc = ppt.mkDisc(10)
        # using incorrect mean for better test
        mn = disc.mean() + 0.1
        v = ppt.var2(disc) + 0.1
        res = ppt.skew2(disc, mn, v)
        self.failUnless(np.absolute(res - mres) <= math.pow(10,-11))
       
class upBlurTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/upBlur0.mat')
        mres = matPyr['res']
        im = ppt.mkRamp((1,20))
        res = ppt.upBlur(im)
        self.failUnless(ppt.compareRecon(mres, res))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/upBlur1.mat')
        mres = matPyr['res']
        im = ppt.mkRamp((1,20))
        res = ppt.upBlur(im.T)
        self.failUnless(ppt.compareRecon(mres, res))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/upBlur2.mat')
        mres = matPyr['res']
        im = ppt.mkRamp(20)
        res = ppt.upBlur(im)
        self.failUnless(ppt.compareRecon(mres, res))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/upBlur3.mat')
        mres = matPyr['res']
        im = ppt.mkRamp((1,20))
        res = ppt.upBlur(im, 3)
        self.failUnless(ppt.compareRecon(mres, res))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/upBlur4.mat')
        mres = matPyr['res']
        im = ppt.mkRamp((1,20))
        res = ppt.upBlur(im.T, 3)
        self.failUnless(ppt.compareRecon(mres, res))
    def test5(self):
        matPyr = scipy.io.loadmat('../matFiles/upBlur5.mat')
        mres = matPyr['res']
        im = ppt.mkRamp(20)
        res = ppt.upBlur(im, 3)
        self.failUnless(ppt.compareRecon(mres, res))
    def test6(self):
        matPyr = scipy.io.loadmat('../matFiles/upBlur6.mat')
        mres = matPyr['res']
        im = ppt.mkRamp((1,20))
        filt = ppt.namedFilter('qmf9')
        res = ppt.upBlur(im, 3, filt)
        self.failUnless(ppt.compareRecon(mres, res))
    #def test7(self):   # fails in matlab and python because of dim mismatch
    #    matPyr = scipy.io.loadmat('../matFiles/upBlur7.mat')
    #    mres = matPyr['res']
    #    im = ppt.mkRamp((1,20))
    #    filt = ppt.namedFilter('qmf9')
    #    res = ppt.upBlur(im, 3, filt.T)
    #    self.failUnless(ppt.compareRecon(mres, res))
    def test8(self):
        matPyr = scipy.io.loadmat('../matFiles/upBlur8.mat')
        mres = matPyr['res']
        im = ppt.mkRamp((1,20))
        filt = ppt.namedFilter('qmf9')
        res = ppt.upBlur(im.T, 3, filt)
        self.failUnless(ppt.compareRecon(mres, res))
    #def test9(self):  # fails in matlab and python because of dim mismatch
    #    matPyr = scipy.io.loadmat('../matFiles/upBlur6.mat')
    #    mres = matPyr['res']
    #    im = ppt.mkRamp((1,20))
    #    filt = ppt.namedFilter('qmf9')
    #    res = ppt.upBlur(im.T, 3, filt.T)
    #    self.failUnless(ppt.compareRecon(mres, res))
    def test10(self):
        matPyr = scipy.io.loadmat('../matFiles/upBlur10.mat')
        mres = matPyr['res']
        im = ppt.mkRamp(20)
        filt = ppt.mkDisc(3)
        res = ppt.upBlur(im, 3, filt)
        self.failUnless(ppt.compareRecon(mres, res))
    def test11(self):
        matPyr = scipy.io.loadmat('../matFiles/upBlur11.mat')
        mres = matPyr['res']
        im = ppt.mkRamp((20,10))
        filt = ppt.mkDisc((5,3))
        res = ppt.upBlur(im, 3, filt)
        self.failUnless(ppt.compareRecon(mres, res))
    def test12(self):
        matPyr = scipy.io.loadmat('../matFiles/upBlur12.mat')
        mres = matPyr['res']
        im = ppt.mkRamp((10,20))
        filt = ppt.mkDisc((3,5))
        res = ppt.upBlur(im, 3, filt)
        self.failUnless(ppt.compareRecon(mres, res))
    
class zconv2Tests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('../matFiles/zconv2_0.mat')
        mres = matPyr['res']
        ramp = ppt.mkRamp(10)
        disc = ppt.mkDisc(5)
        res = ppt.zconv2(ramp, disc)
        self.failUnless(ppt.compareRecon(mres, res))
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/zconv2_1.mat')
        mres = matPyr['res']
        ramp = ppt.mkRamp((10,20))
        disc = ppt.mkDisc((5,10))
        res = ppt.zconv2(ramp, disc)
        self.failUnless(ppt.compareRecon(mres, res))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/zconv2_2.mat')
        mres = matPyr['res']
        ramp = ppt.mkRamp((20,10))
        disc = ppt.mkDisc((10,5))
        res = ppt.zconv2(ramp, disc)
        self.failUnless(ppt.compareRecon(mres, res))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/zconv2_3.mat')
        mres = matPyr['res']
        ramp = ppt.mkRamp(10)
        disc = ppt.mkDisc(5)
        res = ppt.zconv2(ramp, disc, 3)
        self.failUnless(ppt.compareRecon(mres, res))
    def test4(self):
        matPyr = scipy.io.loadmat('../matFiles/zconv2_4.mat')
        mres = matPyr['res']
        ramp = ppt.mkRamp((10,20))
        disc = ppt.mkDisc((5,10))
        res = ppt.zconv2(ramp, disc, 3)
        self.failUnless(ppt.compareRecon(mres, res))
    def test5(self):
        matPyr = scipy.io.loadmat('../matFiles/zconv2_5.mat')
        mres = matPyr['res']
        ramp = ppt.mkRamp((20,10))
        disc = ppt.mkDisc((10,5))
        res = ppt.zconv2(ramp, disc, 3)
        self.failUnless(ppt.compareRecon(mres, res))

class corrDnTests(unittest.TestCase):
    def test1(self):
        matPyr = scipy.io.loadmat('../matFiles/corrDn1.mat')
        mres = matPyr['res']
        ramp = ppt.mkRamp(20)
        res = ppt.corrDn(ramp, ppt.namedFilter('binom5'))
        self.failUnless(ppt.compareRecon(mres, res))
    def test2(self):
        matPyr = scipy.io.loadmat('../matFiles/corrDn2.mat')
        mres = matPyr['res']
        ramp = ppt.mkRamp(20)
        res = ppt.corrDn(ramp, ppt.namedFilter('qmf13'))
        self.failUnless(ppt.compareRecon(mres, res))
    def test3(self):
        matPyr = scipy.io.loadmat('../matFiles/corrDn3.mat')
        mres = matPyr['res']
        ramp = ppt.mkRamp(20)
        res = ppt.corrDn(ramp, ppt.namedFilter('qmf16'))
        self.failUnless(ppt.compareRecon(mres, res))

def main():
    unittest.main()

if __name__ == '__main__':
    main()


