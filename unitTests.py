#!/users-local/ryoung/anaconda/bin/python
import unittest
import pyPyrUtils as ppu
import pyPyrTools as ppt
import math
import numpy as np
import scipy.io
import Image
from operator import mul
#from pyPyrCcode import *

class maxPyrHtTests(unittest.TestCase):
    def test1(self):
        self.failUnless(ppu.maxPyrHt((1,10),(3,4)) == 0)
    def test2(self):
        self.failUnless(ppu.maxPyrHt((10,1),(3,4)) == 0)
    def test3(self):
        self.failUnless(ppu.maxPyrHt((10,10),(1,4)) == 4)
    def test4(self):
        self.failUnless(ppu.maxPyrHt((10,10),(3,1)) == 2)
    def test5(self):
        self.failUnless(ppu.maxPyrHt((10,10),(3,4)) == 2)
    def test6(self):
        self.failUnless(ppu.maxPyrHt((20,10),(5,1)) == 2)
    def test7(self):
        self.failUnless(ppu.maxPyrHt((10,20),(5,1)) == 2)
    def test8(self):
        self.failUnless(ppu.maxPyrHt((20,10),(1,5)) == 5)
    def test9(self):
        self.failUnless(ppu.maxPyrHt((10,20),(1,5)) == 5)
    def test10(self):
        self.failUnless(ppu.maxPyrHt((256,1),(1,5)) == 6)
    def test11(self):
        self.failUnless(ppu.maxPyrHt((256,1),(5,1)) == 6)
    def test12(self):
        self.failUnless(ppu.maxPyrHt((1,256),(1,5)) == 6)
    def test13(self):
        self.failUnless(ppu.maxPyrHt((1,256),(5,1)) == 6)

class binomialFilterTests(unittest.TestCase):
    def test1(self):
        self.failUnless((ppu.binomialFilter(2) == np.array([[0.5], [0.5]])).all() )
    def test2(self):
        self.failUnless((ppu.binomialFilter(3) == np.array([[0.25], [0.5], [0.25]])).all())
    def test3(self):
        self.failUnless((ppu.binomialFilter(5) == np.array([[0.0625], [0.25], [0.3750], [0.25], [0.0625]])).all())

class GpyrTests(unittest.TestCase):
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/buildGpyr1.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/buildGpyr2row.mat')
        img = np.array(range(256)).astype(float)
        img = img.reshape(1, 256)
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/buildGpyr2col.mat')
        img = np.array(range(256)).astype(float)
        img = img.reshape(256, 1)
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/buildGpyr3.mat')
        img = ppu.mkRamp(10)
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat('matFiles/buildGpyr4.mat')
        img = ppu.mkRamp((10,20))
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/buildGpyr5.mat')
        img = ppu.mkRamp((20, 10))
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))

class LpyrTests(unittest.TestCase):
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/buildLpyr1.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.Lpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self): 
        matPyr2 = scipy.io.loadmat('matFiles/buildLpyr2.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr2['pyr'], pyPyr))
    def test3(self):
        matPyr2 = scipy.io.loadmat('matFiles/buildLpyr3.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr2['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/buildLpyr4.mat')
        pyRamp = ppu.mkRamp(200,100)
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):   
        matPyr = scipy.io.loadmat('matFiles/buildLpyr5.mat')
        pyRamp = np.array(range(200)).reshape(1, 200)
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/buildLpyr6.mat')
        pyRamp = np.array(range(200))
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/buildLpyr7.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.Lpyr(img)
        #recon = pyPyr.reconLpyr()
        recon = pyPyr.reconPyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test8(self): 
        matPyr = scipy.io.loadmat('matFiles/buildLpyr8.mat')
        pyRamp = ppu.mkRamp(200)
        pyPyr = ppt.Lpyr(pyRamp)
        #recon = pyPyr.reconLpyr()
        recon = pyPyr.reconPyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test9(self): 
        matPyr = scipy.io.loadmat('matFiles/buildLpyr9.mat')
        pyRamp = ppu.mkRamp((200,100))
        pyPyr = ppt.Lpyr(pyRamp)
        #recon = pyPyr.reconLpyr()
        recon = pyPyr.reconPyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test10(self): 
        matPyr = scipy.io.loadmat('matFiles/buildLpyr10.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.Lpyr(pyRamp)
        #recon = pyPyr.reconLpyr()
        recon = pyPyr.reconPyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test11(self): 
        matPyr = scipy.io.loadmat('matFiles/buildLpyr11.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Lpyr(pyRamp)
        #recon = pyPyr.reconLpyr([1])
        recon = pyPyr.reconPyr([1])
        self.failUnless((matPyr['recon'] == recon).all())
    def test12(self): 
        matPyr = scipy.io.loadmat('matFiles/buildLpyr12.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Lpyr(pyRamp)
        #recon = pyPyr.reconLpyr([0,2,4])
        recon = pyPyr.reconPyr([0,2,4])
        self.failUnless((matPyr['recon'] == recon).all())

class spFilterTests(unittest.TestCase):
    def test1(self):
        matFilt0 = scipy.io.loadmat('matFiles/sp0Filters.mat')
        pySP0filt = ppu.sp0Filters()
        tmpKeys = []
        for key in matFilt0.keys():
            if "_" not in key:
                tmpKeys.append(key)
        self.failUnless(tmpKeys == pySP0filt.keys())
        for key in tmpKeys:
            self.failUnless((matFilt0[key] == pySP0filt[key]).all())

    def test2(self):
        matFilt1 = scipy.io.loadmat('matFiles/sp1Filters.mat')
        pySP1filt = ppu.sp1Filters()
        tmpKeys = []
        for key in matFilt1.keys():
            if "_" not in key:
                tmpKeys.append(key)
        self.failUnless(tmpKeys == pySP1filt.keys())
        for key in tmpKeys:
            self.failUnless((matFilt1[key] == pySP1filt[key]).all())

    def test3(self):
        matFilt3 = scipy.io.loadmat('matFiles/sp3Filters.mat')
        pySP3filt = ppu.sp3Filters()
        tmpKeys = []
        for key in matFilt3.keys():
            if "_" not in key:
                tmpKeys.append(key)
        self.failUnless(tmpKeys == pySP3filt.keys())
        for key in tmpKeys:
            self.failUnless((matFilt3[key] == pySP3filt[key]).all())

    def test4(self):
        matFilt5 = scipy.io.loadmat('matFiles/sp5Filters.mat')
        pySP5filt = ppu.sp5Filters()
        tmpKeys = []
        for key in matFilt5.keys():
            if "_" not in key:
                tmpKeys.append(key)
        self.failUnless(tmpKeys == pySP5filt.keys())
        for key in tmpKeys:
            self.failUnless((matFilt5[key] == pySP5filt[key]).all())

class SpyrTests(unittest.TestCase):
    def test00(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr00.mat')
        pyRamp = ppu.mkRamp((20,20))
        pyPyr = ppt.Spyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr1.mat')
        img = np.array(Image.open('lenna-256x256.tif')).astype(float)
        pyPyr = ppt.Spyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr2.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr3.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.Spyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr4.mat')
        pyRamp = ppu.mkRamp((200,100))
        pyPyr = ppt.Spyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr0.mat')
        pyRamp = ppu.mkRamp(20)
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test5(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr5.mat')
        img = np.array(Image.open('lenna-256x256.tif')).astype(float)
        pyPyr = ppt.Spyr(img)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr6.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr7.mat')
        pyRamp = ppu.mkRamp((256,128))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr8.mat')
        pyRamp = ppu.mkRamp((128,256))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr9.mat')
        pyRamp = ppu.mkRamp((200,100))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr10.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr11.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr('sp1Filters', 'reflect1', [1])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr12.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconPyr('sp1Filters', 'reflect1', [0,2,4])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr13.mat')
        pyRamp = ppu.mkRamp((20,20))
        pyPyr = ppt.Spyr(pyRamp, 1, 'sp0Filters')
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test14(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr14.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 3, 'sp0Filters')
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test15(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr15.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 1, 'sp1Filters')
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test16(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr16.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 3, 'sp1Filters')
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test17(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr17.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 1, 'sp3Filters')
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test18(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr18.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 3, 'sp3Filters')
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test19(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr19.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 1, 'sp5Filters')
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test20(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr20.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp, 3, 'sp5Filters')
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test21(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr21.mat')
        texture = scipy.io.loadmat('matFiles/im04-1.mat')['res'][0:256,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp0Filters')
        recon = pyPyr.reconPyr('sp0Filters');
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test22(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr22.mat')
        texture = scipy.io.loadmat('matFiles/im04-1.mat')['res'][0:256,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp1Filters')
        recon = pyPyr.reconPyr('sp1Filters');
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test23(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr23.mat')
        texture = scipy.io.loadmat('matFiles/im04-1.mat')['res'][0:256,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp3Filters')
        recon = pyPyr.reconPyr('sp3Filters');
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test24(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr24.mat')
        texture = scipy.io.loadmat('matFiles/im04-1.mat')['res'][0:256,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp5Filters')
        recon = pyPyr.reconPyr('sp5Filters');
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test25(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr25.mat')
        texture = scipy.io.loadmat('matFiles/im04-1.mat')['res'][0:256,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp5Filters')
        recon = pyPyr.reconPyr('sp5Filters','reflect1',[0,1,2], [0]);
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test26(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr26.mat')
        texture = scipy.io.loadmat('matFiles/im04-1.mat')['res'][0:128,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp0Filters')
        recon = pyPyr.reconPyr('sp0Filters','reflect1',[0,1,2], [0]);
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test27(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr27.mat')
        texture = scipy.io.loadmat('matFiles/im04-1.mat')['res'][0:128,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp1Filters')
        recon = pyPyr.reconPyr('sp1Filters','reflect1',[0,1,2], [0]);
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test28(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr28.mat')
        texture = scipy.io.loadmat('matFiles/im04-1.mat')['res'][0:128,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp3Filters')
        recon = pyPyr.reconPyr('sp3Filters','reflect1',[0,1,2], [0]);
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test29(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr29.mat')
        texture = scipy.io.loadmat('matFiles/im04-1.mat')['res'][0:128,0:256]
        pyPyr = ppt.Spyr(texture, 3, 'sp5Filters')
        recon = pyPyr.reconPyr('sp5Filters','reflect1',[0,1,2], [0]);
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
        
        
        

class pointOpTests(unittest.TestCase):
    def test1(self):
        matImg = scipy.io.loadmat('matFiles/pointOp1.mat')
        img = ppu.mkRamp((200,200))
        filt = np.array([0.2, 0.5, 1.0, 0.4, 0.1]);
        #foo = pointOp(200, 200, img, 5, filt, 0, 1, 0);
        foo = ppu.pointOp(img, filt, 0, 1, 0);
        foo = np.reshape(foo,(200,200))
        self.failUnless((matImg['foo'] == foo).all())

class SFpyrTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr0.mat')
        pyRamp = ppu.mkRamp((20,20))
        pyPyr = ppt.SFpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr1.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.SFpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr2.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.SFpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr3.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.SFpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr4.mat')
        pyRamp = ppu.mkRamp((200,100))
        pyPyr = ppt.SFpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr5.mat')
        pyRamp = ppu.mkRamp((20,20))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr6.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.SFpyr(img)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr7.mat')
        pyRamp = ppu.mkRamp((256,128))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr8.mat')
        pyRamp = ppu.mkRamp((128,256))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr9.mat')
        pyRamp = ppu.mkRamp((200,100))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr10.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr11.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr([0])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr12.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr([0,2,4])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr13.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconPyr([0,2,4], [1])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))

class SCFpyrTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr0.mat')
        pyRamp = ppu.mkRamp((20,20))
        pyPyr = ppt.SCFpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr1.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.SCFpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr2.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.SCFpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr3.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.SCFpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr4.mat')
        pyRamp = ppu.mkRamp((200,100))
        pyPyr = ppt.SCFpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr5.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.SCFpyr(img)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr6.mat')
        pyRamp = ppu.mkRamp((256,128))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr7.mat')
        pyRamp = ppu.mkRamp((128,256))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    #def test8(self):  # fails in matlab version
    #    matPyr = scipy.io.loadmat('matFiles/buildSCFpyr8.mat')
    #    pyRamp = ppu.mkRamp((200,100))
    #    pyPyr = ppt.SCFpyr(pyRamp)
    #    recon = pyPyr.reconSCFpyr()
    #    self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    #def test9(self):
    #    matPyr = scipy.io.loadmat('matFiles/buildSCFpyr9.mat')
    #    pyRamp = ppu.mkRamp((100,200))
    #    pyPyr = ppt.SCFpyr(pyRamp)
    #    recon = pyPyr.reconSCFpyr()
    #    self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr10.mat')
        pyRamp = ppu.mkRamp((256,256))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconPyr([0])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr11.mat')
        pyRamp = ppu.mkRamp((256,256))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconPyr([0,2,4])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr12.mat')
        pyRamp = ppu.mkRamp((256,256))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconPyr([0,2,4], [1])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))

class WpyrTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr0.mat')
        pyRamp = ppu.mkRamp((20,20))
        pyPyr = ppt.Wpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr1.mat')
        img = np.array(Image.open('lenna-256x256.tif')).astype(float)
        pyPyr = ppt.Wpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr2.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Wpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr3.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.Wpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr4.mat')
        pyRamp = ppu.mkRamp((200,100))
        pyPyr = ppt.Wpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr5.mat')
        img = np.array(Image.open('lenna-256x256.tif')).astype(float)
        pyPyr = ppt.Wpyr(img)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr6.mat')
        pyRamp = ppu.mkRamp((256,128))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr7.mat')
        pyRamp = ppu.mkRamp((128,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr8.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr9.mat')
        pyRamp = ppu.mkRamp((200,100))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr10.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr11.mat')
        pyRamp = ppu.mkRamp((256,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr('qmf9', 'reflect1', [0])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr12.mat')
        pyRamp = ppu.mkRamp((256,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr('qmf9', 'reflect1', [0,2,4])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr13.mat')
        pyRamp = ppu.mkRamp((256,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr('qmf9', 'reflect1', [0,2,4], [1])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test14(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr14.mat')
        pyRamp = ppu.mkRamp((256,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr('qmf8')
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test15(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr15.mat')
        pyRamp = ppu.mkRamp((256,128))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr('qmf8')
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test16(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr16.mat')
        pyRamp = ppu.mkRamp((128,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconPyr('qmf8')
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test17(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr17.mat')
        pyRamp = ppu.mkRamp((1,200))
        pyPyr = ppt.Wpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test18(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr18.mat')
        pyRamp = ppu.mkRamp((1,200)).T
        pyPyr = ppt.Wpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))

class blurDnTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/blurDn0.mat')
        pyRamp = ppu.mkRamp((20,20))
        res = ppu.blurDn(pyRamp)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/blurDn1.mat')
        pyRamp = ppu.mkRamp((256,256))
        res = ppu.blurDn(pyRamp)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/blurDn2.mat')
        pyRamp = ppu.mkRamp((256,128))
        res = ppu.blurDn(pyRamp)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/blurDn3.mat')
        pyRamp = ppu.mkRamp((128,256))
        res = ppu.blurDn(pyRamp)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/blurDn4.mat')
        pyRamp = ppu.mkRamp((200, 100))
        res = ppu.blurDn(pyRamp)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat('matFiles/blurDn5.mat')
        pyRamp = ppu.mkRamp((100, 200))
        res = ppu.blurDn(pyRamp)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/blurDn6.mat')
        pyRamp = ppu.mkRamp((1, 256)).T
        res = ppu.blurDn(pyRamp)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/blurDn7.mat')
        pyRamp = ppu.mkRamp((1, 256))
        res = ppu.blurDn(pyRamp)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    #def test8(self):  need a 2D filter
    #    matPyr = scipy.io.loadmat('matFiles/blurDn8.mat')
    #    pyRamp = ppu.mkRamp((256, 256))
    #    res = ppu.blurDn(pyRamp, 2dfilt)
    #    self.failUnless(ppu.compareRecon(matPyr['res'], res))

class mkAngularSineTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/mkAngularSine0.mat')
        res = ppu.mkAngularSine(20)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/mkAngularSine1.mat')
        res = ppu.mkAngularSine(20, 5)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/mkAngularSine2.mat')
        res = ppu.mkAngularSine(20, 5, 3)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/mkAngularSine3.mat')
        res = ppu.mkAngularSine(20, 5, 3, 2)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/mkAngularSine4.mat')
        res = ppu.mkAngularSine(20, 5, 3, 2, (2,2))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))

class mkGaussianTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/mkGaussian0.mat')
        res = ppu.mkGaussian(20)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/mkGaussian1.mat')
        res = ppu.mkGaussian(20, (2,3))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/mkGaussian2.mat')
        res = ppu.mkGaussian(20, [[-1, 0], [0, 1]])
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/mkGaussian3.mat')
        res = ppu.mkGaussian(10, [[-1, 0], [0, 1]])
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/mkGaussian4.mat')
        res = ppu.mkGaussian(20, [[2, 0], [0, 1]])
        self.failUnless(ppu.compareRecon(matPyr['res'], res))

class mkDiscTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/mkDisc0.mat')
        res = ppu.mkDisc(20)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/mkDisc1.mat')
        res = ppu.mkDisc(20, 8)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/mkDisc2.mat')
        res = ppu.mkDisc(20, 8, (0,0))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/mkDisc3.mat')
        res = ppu.mkDisc(20, 8, (0,0), 5)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/mkDisc4.mat')
        res = ppu.mkDisc(20, 8, (0,0), 5, (0.75, 0.25))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))

class mkSineTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/mkSine0.mat')
        res = ppu.mkSine(20, 5.5)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/mkSine1.mat')
        res = ppu.mkSine(20, 5.5, 2)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/mkSine2.mat')
        res = ppu.mkSine(20, 5.5, 2, 3)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/mkSine3.mat')
        res = ppu.mkSine(20, 5.5, 2, 3, 5)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/mkSine4.mat')
        res = ppu.mkSine(20, 5.5, 2, 3, 5, [4,5])
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat('matFiles/mkSine5.mat')
        res = ppu.mkSine(20, [1,2])
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/mkSine6.mat')
        res = ppu.mkSine(20, [1,2], 3)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/mkSine7.mat')
        res = ppu.mkSine(20, [1,2], 3, 2)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test8(self):
        matPyr = scipy.io.loadmat('matFiles/mkSine8.mat')
        res = ppu.mkSine(20, [1,2], 3, 2, [5,4])
        self.failUnless(ppu.compareRecon(matPyr['res'], res))

class mkZonePlateTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/mkZonePlate0.mat')
        res = ppu.mkZonePlate(20)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/mkZonePlate1.mat')
        res = ppu.mkZonePlate(20, 4)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/mkZonePlate2.mat')
        res = ppu.mkZonePlate(20, 4, 3)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))

class mkSquareTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/mkSquare0.mat')
        res = ppu.mkSquare(20, 5.5)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/mkSquare1.mat')
        res = ppu.mkSquare(20, 5.5, 3)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/mkSquare2.mat')
        res = ppu.mkSquare(20, 5.5, 3, 5.1)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/mkSquare3.mat')
        res = ppu.mkSquare(20, 5.5, 3, 5.1, -1)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/mkSquare4.mat')
        res = ppu.mkSquare(20, 5.5, 3, 5.1, -1, (2,3))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat('matFiles/mkSquare5.mat')
        res = ppu.mkSquare(20, 5.5, 3, 5.1, -1, (2,3), 0.25)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/mkSquare6.mat')
        res = ppu.mkSquare(20, (1,2))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/mkSquare7.mat')
        res = ppu.mkSquare(20, (1,2), 3.2)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test8(self):
        matPyr = scipy.io.loadmat('matFiles/mkSquare8.mat')
        res = ppu.mkSquare(20, (1,2), 3.2, -2)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test9(self):
        matPyr = scipy.io.loadmat('matFiles/mkSquare9.mat')
        res = ppu.mkSquare(20, (1,2), 3.2, -2, (2,3))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test10(self):
        matPyr = scipy.io.loadmat('matFiles/mkSquare10.mat')
        res = ppu.mkSquare(20, (1,2), 3.2, -2, (2,3), 0.55)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))

class blurTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/blur0.mat')
        res = ppu.blur(ppu.mkRamp(20))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/blur1.mat')
        res = ppu.blur(ppu.mkRamp(20), 3)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/blur2.mat')
        res = ppu.blur(ppu.mkRamp(20), 3, ppu.namedFilter('qmf5'))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/blur3.mat')
        res = ppu.blur(ppu.mkRamp((20,30)))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/blur4.mat')
        res = ppu.blur(ppu.mkRamp((20,30)), 3)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat('matFiles/blur5.mat')
        res = ppu.blur(ppu.mkRamp((20,30)), 3, ppu.namedFilter('qmf5'))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))

class cconv2Tests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/cconv2_0.mat')
        res = ppu.cconv2(ppu.mkRamp(20), ppu.mkRamp(10))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/cconv2_1.mat')
        res = ppu.cconv2(ppu.mkRamp(10), ppu.mkRamp(20))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/cconv2_2.mat')
        res = ppu.cconv2(ppu.mkRamp(20), ppu.mkRamp(10), 3)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/cconv2_3.mat')
        res = ppu.cconv2(ppu.mkRamp(10), ppu.mkRamp(20), 3)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/cconv2_4.mat')
        res = ppu.cconv2(ppu.mkRamp((20,30)), ppu.mkRamp((10,20)))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test5(self):
        matPyr = scipy.io.loadmat('matFiles/cconv2_5.mat')
        res = ppu.cconv2(ppu.mkRamp((10,20)), ppu.mkRamp((20,30)))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/cconv2_6.mat')
        res = ppu.cconv2(ppu.mkRamp((20,30)), ppu.mkRamp((10,20)), 5)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/cconv2_7.mat')
        res = ppu.cconv2(ppu.mkRamp((10,20)), ppu.mkRamp((20,30)), 5)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))

class clipTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/clip0.mat')
        res = ppu.clip(ppu.mkRamp(20) / ppu.mkRamp(20).max())
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/clip1.mat')
        res = ppu.clip(ppu.mkRamp(20) / ppu.mkRamp(20).max(), 0.3)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/clip2.mat')
        res = ppu.clip(ppu.mkRamp(20) / ppu.mkRamp(20).max(), (0.3,0.7))
        self.failUnless(ppu.compareRecon(matPyr['res'], res))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/clip3.mat')
        res = ppu.clip(ppu.mkRamp(20) / ppu.mkRamp(20).max(), 0.3, 0.7)
        self.failUnless(ppu.compareRecon(matPyr['res'], res))

# python version of histo
# adding 0.7 to ramp to nullify rounding differences between Python and Matlab
class histoTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/histo0.mat')
        (N,X) = ppu.histo(ppu.mkRamp(10) + 0.7)
        # X will not be the same because matlab returns centers and 
        #   python return edges
        #self.failUnless(ppu.compareRecon(matPyr['X'], X))
        #self.failUnless(ppu.compareRecon(matPyr['N'], N))
        self.failUnless((matPyr['N'] == N).all())
# FIX: why does matlab version return N+1 bins??
#    def test1(self):
#        matPyr = scipy.io.loadmat('matFiles/histo1.mat')
#        (N,X) = ppu.histo(ppu.mkRamp(10) + 0.7, 26)
#        # X will not be the same because matlab returns centers and 
#        #   python return edges
#        #self.failUnless(ppu.compareRecon(matPyr['X'], X))
#        #self.failUnless(ppu.compareRecon(matPyr['N'], N))
#        self.failUnless(matPyr['N'] == N)
#    def test2(self):
#        matPyr = scipy.io.loadmat('matFiles/histo2.mat')
#        (N,X) = ppu.histo(ppu.mkRamp(10) + 0.7, -1.15)
#        # X will not be the same because matlab returns centers and 
#        #   python return edges
#        #self.failUnless(ppu.compareRecon(matPyr['X'], X))
#        self.failUnless(ppu.compareRecon(matPyr['N'], N))
#    def test3(self):
#        matPyr = scipy.io.loadmat('matFiles/histo3.mat')
#        (N,X) = ppu.histo(ppu.mkRamp(10) + 0.7, 26, 3)
#        # X will not be the same because matlab returns centers and 
#        #   python return edges
#        #self.failUnless(ppu.compareRecon(matPyr['X'], X))
#        self.failUnless(ppu.compareRecon(matPyr['N'], N))

#class entropy2Tests(unittest.TestCase):
#    def test0(self):
#        matPyr = scipy.io.loadmat('matFiles/entropy2_0.mat')
#        H = ppu.entropy2(ppu.mkRamp(10))
#        self.failUnless(matPyr['H'] == H)
#    def test1(self):
#        matPyr = scipy.io.loadmat('matFiles/entropy2_1.mat')
#        H = ppu.entropy2(ppu.mkRamp(10), 1)
#        self.failUnless(matPyr['H'] == H)

class factorialTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/factorial0.mat')
        res = ppu.factorial([[1,2],[3,4]])
        self.failUnless((matPyr['res'] == res).all())
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/factorial1.mat')
        res = ppu.factorial(4)
        self.failUnless(matPyr['res'] == res)

#class histoMatchTests(unittest.TestCase):
#    def test0(self):
#        matPyr = scipy.io.loadmat('matFiles/histoMatch0.mat')
#        # adding 0.7 to get the bins to line up between matlab and python
#        # answers between matlab and python may be different, 
#        #   but not necessarily incorrect.
#        # similar to histo above
#        ramp = ppu.mkRamp(10) + 0.7
#        disc = ppu.mkDisc(10) + 0.7
#        (rN,rX) = ppu.histo(ramp)
#        res = ppu.histoMatch(disc, rN, rX, 'edges')
#        self.failUnless(ppu.compareRecon(matPyr['res'], res))

class imGradientTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/imGradient0.mat')
        ramp = ppu.mkRamp(10)
        [dx,dy] = ppu.imGradient(ramp)
        dx = np.array(dx)
        dy = np.array(dy)
        self.failUnless(ppu.compareRecon(matPyr['res'][:,:,0], dx))
        self.failUnless(ppu.compareRecon(matPyr['res'][:,:,1], dy))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/imGradient1.mat')
        ramp = ppu.mkRamp(10)
        [dx,dy] = ppu.imGradient(ramp, 'reflect1')
        dx = np.array(dx)
        dy = np.array(dy)
        self.failUnless(ppu.compareRecon(matPyr['res'][:,:,0], dx))
        self.failUnless(ppu.compareRecon(matPyr['res'][:,:,1], dy))
 
class skew2Tests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/skew2_0.mat')
        # not sure why matPyr is [[ans]]???
        mres = matPyr['res'][0][0]
        disc = ppu.mkDisc(10)
        res = ppu.skew2(disc)
        self.failUnless(np.absolute(res - mres) <= math.pow(10,-11))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/skew2_1.mat')
        # not sure why matPyr is [[ans]]???
        mres = matPyr['res'][0][0]
        disc = ppu.mkDisc(10)
        # using incorrect mean for better test
        mn = disc.mean() + 0.1
        res = ppu.skew2(disc, mn)
        self.failUnless(np.absolute(res - mres) <= math.pow(10,-11))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/skew2_2.mat')
        # not sure why matPyr is [[ans]]???
        mres = matPyr['res'][0][0]
        disc = ppu.mkDisc(10)
        # using incorrect mean for better test
        mn = disc.mean() + 0.1
        v = ppu.var2(disc) + 0.1
        res = ppu.skew2(disc, mn, v)
        self.failUnless(np.absolute(res - mres) <= math.pow(10,-11))
       
class upBlurTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/upBlur0.mat')
        mres = matPyr['res']
        im = ppu.mkRamp((1,20))
        res = ppu.upBlur(im)
        self.failUnless(ppu.compareRecon(mres, res))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/upBlur1.mat')
        mres = matPyr['res']
        im = ppu.mkRamp((1,20))
        res = ppu.upBlur(im.T)
        self.failUnless(ppu.compareRecon(mres, res))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/upBlur2.mat')
        mres = matPyr['res']
        im = ppu.mkRamp(20)
        res = ppu.upBlur(im)
        self.failUnless(ppu.compareRecon(mres, res))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/upBlur3.mat')
        mres = matPyr['res']
        im = ppu.mkRamp((1,20))
        res = ppu.upBlur(im, 3)
        self.failUnless(ppu.compareRecon(mres, res))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/upBlur4.mat')
        mres = matPyr['res']
        im = ppu.mkRamp((1,20))
        res = ppu.upBlur(im.T, 3)
        self.failUnless(ppu.compareRecon(mres, res))
    def test5(self):
        matPyr = scipy.io.loadmat('matFiles/upBlur5.mat')
        mres = matPyr['res']
        im = ppu.mkRamp(20)
        res = ppu.upBlur(im, 3)
        self.failUnless(ppu.compareRecon(mres, res))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/upBlur6.mat')
        mres = matPyr['res']
        im = ppu.mkRamp((1,20))
        filt = ppu.namedFilter('qmf9')
        res = ppu.upBlur(im, 3, filt)
        self.failUnless(ppu.compareRecon(mres, res))
    #def test7(self):   # fails in matlab and python because of dim mismatch
    #    matPyr = scipy.io.loadmat('matFiles/upBlur7.mat')
    #    mres = matPyr['res']
    #    im = ppu.mkRamp((1,20))
    #    filt = ppu.namedFilter('qmf9')
    #    res = ppu.upBlur(im, 3, filt.T)
    #    self.failUnless(ppu.compareRecon(mres, res))
    def test8(self):
        matPyr = scipy.io.loadmat('matFiles/upBlur8.mat')
        mres = matPyr['res']
        im = ppu.mkRamp((1,20))
        filt = ppu.namedFilter('qmf9')
        res = ppu.upBlur(im.T, 3, filt)
        self.failUnless(ppu.compareRecon(mres, res))
    #def test9(self):  # fails in matlab and python because of dim mismatch
    #    matPyr = scipy.io.loadmat('matFiles/upBlur6.mat')
    #    mres = matPyr['res']
    #    im = ppu.mkRamp((1,20))
    #    filt = ppu.namedFilter('qmf9')
    #    res = ppu.upBlur(im.T, 3, filt.T)
    #    self.failUnless(ppu.compareRecon(mres, res))
    def test10(self):
        matPyr = scipy.io.loadmat('matFiles/upBlur10.mat')
        mres = matPyr['res']
        im = ppu.mkRamp(20)
        filt = ppu.mkDisc(3)
        res = ppu.upBlur(im, 3, filt)
        self.failUnless(ppu.compareRecon(mres, res))
    def test11(self):
        matPyr = scipy.io.loadmat('matFiles/upBlur11.mat')
        mres = matPyr['res']
        im = ppu.mkRamp((20,10))
        filt = ppu.mkDisc((5,3))
        res = ppu.upBlur(im, 3, filt)
        self.failUnless(ppu.compareRecon(mres, res))
    def test12(self):
        matPyr = scipy.io.loadmat('matFiles/upBlur12.mat')
        mres = matPyr['res']
        im = ppu.mkRamp((10,20))
        filt = ppu.mkDisc((3,5))
        res = ppu.upBlur(im, 3, filt)
        self.failUnless(ppu.compareRecon(mres, res))
    
class zconv2Tests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/zconv2_0.mat')
        mres = matPyr['res']
        ramp = ppu.mkRamp(10)
        disc = ppu.mkDisc(5)
        res = ppu.zconv2(ramp, disc)
        self.failUnless(ppu.compareRecon(mres, res))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/zconv2_1.mat')
        mres = matPyr['res']
        ramp = ppu.mkRamp((10,20))
        disc = ppu.mkDisc((5,10))
        res = ppu.zconv2(ramp, disc)
        self.failUnless(ppu.compareRecon(mres, res))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/zconv2_2.mat')
        mres = matPyr['res']
        ramp = ppu.mkRamp((20,10))
        disc = ppu.mkDisc((10,5))
        res = ppu.zconv2(ramp, disc)
        self.failUnless(ppu.compareRecon(mres, res))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/zconv2_3.mat')
        mres = matPyr['res']
        ramp = ppu.mkRamp(10)
        disc = ppu.mkDisc(5)
        res = ppu.zconv2(ramp, disc, 3)
        self.failUnless(ppu.compareRecon(mres, res))
    def test4(self):
        matPyr = scipy.io.loadmat('matFiles/zconv2_4.mat')
        mres = matPyr['res']
        ramp = ppu.mkRamp((10,20))
        disc = ppu.mkDisc((5,10))
        res = ppu.zconv2(ramp, disc, 3)
        self.failUnless(ppu.compareRecon(mres, res))
    def test5(self):
        matPyr = scipy.io.loadmat('matFiles/zconv2_5.mat')
        mres = matPyr['res']
        ramp = ppu.mkRamp((20,10))
        disc = ppu.mkDisc((10,5))
        res = ppu.zconv2(ramp, disc, 3)
        self.failUnless(ppu.compareRecon(mres, res))

class corrDnTests(unittest.TestCase):
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/corrDn1.mat')
        mres = matPyr['res']
        ramp = ppu.mkRamp(20)
        res = ppu.corrDn(ramp, ppu.namedFilter('binom5'))
        self.failUnless(ppu.compareRecon(mres, res))
    def test2(self):
        matPyr = scipy.io.loadmat('matFiles/corrDn2.mat')
        mres = matPyr['res']
        ramp = ppu.mkRamp(20)
        res = ppu.corrDn(ramp, ppu.namedFilter('qmf13'))
        self.failUnless(ppu.compareRecon(mres, res))
    def test3(self):
        matPyr = scipy.io.loadmat('matFiles/corrDn3.mat')
        mres = matPyr['res']
        ramp = ppu.mkRamp(20)
        res = ppu.corrDn(ramp, ppu.namedFilter('qmf16'))
        self.failUnless(ppu.compareRecon(mres, res))
    
def main():
    unittest.main()

if __name__ == '__main__':
    main()


