#!/usr/bin/python
import unittest
import pyPyrUtils as ppu
import pyPyrTools as ppt
import numpy as np
import scipy.io
import Image
from operator import mul

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

class binomialFilterTests(unittest.TestCase):
    def test1(self):
        self.failUnless((ppu.binomialFilter(2) == np.array([[0.5], [0.5]])).all() )
    def test2(self):
        self.failUnless((ppu.binomialFilter(3) == np.array([[0.25], [0.5], [0.25]])).all())
    def test3(self):
        self.failUnless((ppu.binomialFilter(5) == np.array([[0.0625], [0.25], [0.3750], [0.25], [0.0625]])).all())

class GpyrTests(unittest.TestCase):
    def test1(self):
        matPyr = scipy.io.loadmat('buildGpyr1.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat('buildGpyr2row.mat')
        img = np.array(range(256)).astype(float)
        img = img.reshape(1, 256)
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat('buildGpyr2col.mat')
        img = np.array(range(256)).astype(float)
        img = img.reshape(256, 1)
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('buildGpyr3.mat')
        img = ppu.mkRamp(10)
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat('buildGpyr4.mat')
        img = ppu.mkRamp((10,20))
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test6(self):
        matPyr = scipy.io.loadmat('buildGpyr5.mat')
        img = ppu.mkRamp((200, 100))
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))

class mkRampTests(unittest.TestCase):
    def test1(self):
        matRamp = scipy.io.loadmat('mkRamp1.mat')
        pyRamp = ppu.mkRamp(10)
        self.failUnless((matRamp['foo'] == pyRamp).all())
    def test2(self):
        matRamp = scipy.io.loadmat('mkRamp2.mat')
        pyRamp = ppu.mkRamp(10,20)
        self.failUnless((matRamp['foo'] == pyRamp).all())
    def test3(self):
        matRamp = scipy.io.loadmat('mkRamp3.mat')
        pyRamp = ppu.mkRamp(20, 10)
        self.failUnless((matRamp['foo'] == pyRamp).all())

class LpyrTests(unittest.TestCase):
    def test1(self):
        matPyr = scipy.io.loadmat('buildLpyr1.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.Lpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self): 
        matPyr2 = scipy.io.loadmat('buildLpyr2.mat')
        pyRamp = ppu.mkRamp(200,200)
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr2['pyr'], pyPyr))
    def test3(self):
        matPyr2 = scipy.io.loadmat('buildLpyr3.mat')
        pyRamp = ppu.mkRamp(100,200)
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr2['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('buildLpyr4.mat')
        pyRamp = ppu.mkRamp(200,100)
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat('buildLpyr5.mat')
        pyRamp = np.array(range(200))
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test6(self):
        matPyr = scipy.io.loadmat('buildLpyr6.mat')
        pyRamp = np.array(range(200)).T
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test7(self):
        matPyr = scipy.io.loadmat('buildLpyr7.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.Lpyr(img)
        recon = pyPyr.reconLpyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test8(self): 
        matPyr = scipy.io.loadmat('buildLpyr8.mat')
        pyRamp = ppu.mkRamp(200)
        pyPyr = ppt.Lpyr(pyRamp)
        recon = pyPyr.reconLpyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test9(self): 
        matPyr = scipy.io.loadmat('buildLpyr9.mat')
        pyRamp = ppu.mkRamp((200,100))
        pyPyr = ppt.Lpyr(pyRamp)
        recon = pyPyr.reconLpyr()
        self.failUnless((matPyr['recon'] == recon).all())
    def test10(self): 
        matPyr = scipy.io.loadmat('buildLpyr10.mat')
        pyRamp = ppu.mkRamp((100,200))
        pyPyr = ppt.Lpyr(pyRamp)
        recon = pyPyr.reconLpyr()
        self.failUnless((matPyr['recon'] == recon).all())


class spFilterTests(unittest.TestCase):
    def test1(self):
        matFilt0 = scipy.io.loadmat('sp0Filters.mat')
        pySP0filt = ppu.sp0Filters()
        tmpKeys = []
        for key in matFilt0.keys():
            if "_" not in key:
                tmpKeys.append(key)
        self.failUnless(tmpKeys == pySP0filt.keys())
        for key in tmpKeys:
            self.failUnless((matFilt0[key] == pySP0filt[key]).all())

    def test2(self):
        matFilt1 = scipy.io.loadmat('sp1Filters.mat')
        pySP1filt = ppu.sp1Filters()
        tmpKeys = []
        for key in matFilt1.keys():
            if "_" not in key:
                tmpKeys.append(key)
        self.failUnless(tmpKeys == pySP1filt.keys())
        for key in tmpKeys:
            self.failUnless((matFilt1[key] == pySP1filt[key]).all())

class SpyrTests(unittest.TestCase):
    def test1(self):
        matPyr = scipy.io.loadmat('buildSpyr1.mat')
        img = Image.open('lenna-256x256.tif')
        img = np.array(img.getdata()).reshape(256,256)
        pyPyr = ppt.Spyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test2(self):
        matPyr = scipy.io.loadmat('buildSpyr2.mat')
        pyRamp = ppu.mkRamp(200,200)
        pyPyr = ppt.Spyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test3(self):
        matPyr = scipy.io.loadmat('buildSpyr3.mat')
        pyRamp = ppu.mkRamp(100,200)
        pyPyr = ppt.Spyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test4(self):
        matPyr = scipy.io.loadmat('buildSpyr4.mat')
        pyRamp = ppu.mkRamp(200,100)
        pyPyr = ppt.Spyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test5(self):
        matPyr = scipy.io.loadmat('buildSpyr5.mat')
        pyRamp = ppu.mkRamp(200,1)
        pyPyr = ppt.Spyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
#    ## both versions fail for 1D col vector (1x200)

def main():
    unittest.main()

if __name__ == '__main__':
    main()


