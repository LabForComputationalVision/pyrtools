#!/usr/bin/python
import unittest
import pyPyrUtils as ppu
import pyPyrTools as ppt
import numpy as np
import scipy.io
import Image
from operator import mul
from pyPyrCcode import *

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
    #def test2(self):
    #    matPyr = scipy.io.loadmat('matFiles/buildGpyr2row.mat')
    #    img = np.array(range(256)).astype(float)
    #    img = img.reshape(1, 256)
    #    pyPyr = ppt.Gpyr(img)
    #    self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    #def test3(self):
    #    matPyr = scipy.io.loadmat('matFiles/buildGpyr2col.mat')
    #    img = np.array(range(256)).astype(float)
    #    img = img.reshape(256, 1)
    #    pyPyr = ppt.Gpyr(img)
    #    self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
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

class mkRampTests(unittest.TestCase):
    def test1(self):
        matRamp = scipy.io.loadmat('matFiles/mkRamp1.mat')
        pyRamp = ppu.mkRamp(10)
        self.failUnless((matRamp['foo'] == pyRamp).all())
    def test2(self):
        matRamp = scipy.io.loadmat('matFiles/mkRamp2.mat')
        pyRamp = ppu.mkRamp(10,20)
        self.failUnless((matRamp['foo'] == pyRamp).all())
    def test3(self):
        matRamp = scipy.io.loadmat('matFiles/mkRamp3.mat')
        pyRamp = ppu.mkRamp(20, 10)
        self.failUnless((matRamp['foo'] == pyRamp).all())

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
#    def test5(self):
#        matPyr = scipy.io.loadmat('matFiles/buildLpyr5.mat')
#        pyRamp = np.array(range(200)).reshape(1, 200)
#        pyPyr = ppt.Lpyr(pyRamp)
#        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
#    def test6(self):
#        matPyr = scipy.io.loadmat('matFiles/buildLpyr6.mat')
#        pyRamp = np.array(range(200)).T
#        pyPyr = ppt.Lpyr(pyRamp)
#        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/buildLpyr7.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.Lpyr(img)
        recon = pyPyr.reconLpyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test8(self): 
        matPyr = scipy.io.loadmat('matFiles/buildLpyr8.mat')
        pyRamp = ppu.mkRamp(200)
        pyPyr = ppt.Lpyr(pyRamp)
        recon = pyPyr.reconLpyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test9(self): 
        matPyr = scipy.io.loadmat('matFiles/buildLpyr9.mat')
        pyRamp = ppu.mkRamp((200,100))
        pyPyr = ppt.Lpyr(pyRamp)
        recon = pyPyr.reconLpyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test10(self): 
        matPyr = scipy.io.loadmat('matFiles/buildLpyr10.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.Lpyr(pyRamp)
        recon = pyPyr.reconLpyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test11(self): 
        matPyr = scipy.io.loadmat('matFiles/buildLpyr11.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Lpyr(pyRamp)
        recon = pyPyr.reconLpyr([1])
        self.failUnless((matPyr['recon'] == recon).all())
    def test12(self): 
        matPyr = scipy.io.loadmat('matFiles/buildLpyr12.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Lpyr(pyRamp)
        recon = pyPyr.reconLpyr([0,2,4])
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
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
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
        pyRamp = ppu.mkRamp((20,20))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconSpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test5(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr5.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.Spyr(img)
        recon = pyPyr.reconSpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr6.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconSpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr7.mat')
        pyRamp = ppu.mkRamp((256,128))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconSpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr8.mat')
        pyRamp = ppu.mkRamp((128,256))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconSpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr9.mat')
        pyRamp = ppu.mkRamp((200,100))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconSpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr10.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconSpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr11.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconSpyr('sp1Filters', 'reflect1', [1])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat('matFiles/buildSpyr12.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Spyr(pyRamp)
        recon = pyPyr.reconSpyr('sp1Filters', 'reflect1', [0,2,4])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))

class pointOpTests(unittest.TestCase):
    def test1(self):
        matImg = scipy.io.loadmat('matFiles/pointOp1.mat')
        img = ppu.mkRamp((200,200))
        filt = np.array([0.2, 0.5, 1.0, 0.4, 0.1]);
        foo = pointOp(200, 200, img, 5, filt, 0, 1, 0);
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
        recon = pyPyr.reconSFpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr6.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.SFpyr(img)
        recon = pyPyr.reconSFpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr7.mat')
        pyRamp = ppu.mkRamp((256,128))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconSFpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr8.mat')
        pyRamp = ppu.mkRamp((128,256))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconSFpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr9.mat')
        pyRamp = ppu.mkRamp((200,100))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconSFpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr10.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconSFpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr11.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconSFpyr([0])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr12.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconSFpyr([0,2,4])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat('matFiles/buildSFpyr13.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.SFpyr(pyRamp)
        recon = pyPyr.reconSFpyr([0,2,4], [1])
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
        recon = pyPyr.reconSCFpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr6.mat')
        pyRamp = ppu.mkRamp((256,128))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconSCFpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr7.mat')
        pyRamp = ppu.mkRamp((128,256))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconSCFpyr()
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
        recon = pyPyr.reconSCFpyr([0])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr11.mat')
        pyRamp = ppu.mkRamp((256,256))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconSCFpyr([0,2,4])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat('matFiles/buildSCFpyr12.mat')
        pyRamp = ppu.mkRamp((256,256))
        pyPyr = ppt.SCFpyr(pyRamp)
        recon = pyPyr.reconSCFpyr([0,2,4], [1])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))

class WpyrTests(unittest.TestCase):
    def test0(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr0.mat')
        pyRamp = ppu.mkRamp((20,20))
        pyPyr = ppt.Wpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test1(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr1.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
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
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.Wpyr(img)
        recon = pyPyr.reconWpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test6(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr6.mat')
        pyRamp = ppu.mkRamp((256,128))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconWpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test7(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr7.mat')
        pyRamp = ppu.mkRamp((128,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconWpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test8(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr8.mat')
        pyRamp = ppu.mkRamp((200,200))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconWpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test9(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr9.mat')
        pyRamp = ppu.mkRamp((200,100))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconWpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test10(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr10.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconWpyr()
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test11(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr11.mat')
        pyRamp = ppu.mkRamp((256,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconWpyr('qmf9', 'reflect1', [0])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test12(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr12.mat')
        pyRamp = ppu.mkRamp((256,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconWpyr('qmf9', 'reflect1', [0,2,4])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test13(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr13.mat')
        pyRamp = ppu.mkRamp((256,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconWpyr('qmf9', 'reflect1', [0,2,4], [1])
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test14(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr14.mat')
        pyRamp = ppu.mkRamp((256,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconWpyr('qmf8')
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test15(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr15.mat')
        pyRamp = ppu.mkRamp((256,128))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconWpyr('qmf8')
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))
    def test16(self):
        matPyr = scipy.io.loadmat('matFiles/buildWpyr16.mat')
        pyRamp = ppu.mkRamp((128,256))
        pyPyr = ppt.Wpyr(pyRamp)
        recon = pyPyr.reconWpyr('qmf8')
        self.failUnless(ppu.compareRecon(matPyr['recon'], recon))

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
    

def main():
    unittest.main()

if __name__ == '__main__':
    main()


