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
        img = ppu.mkRamp(10,20)
        pyPyr = ppt.Gpyr(img)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))
    def test6(self):
        matPyr = scipy.io.loadmat('buildGpyr5.mat')
        img = ppu.mkRamp(200, 100)
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
        pyRamp = np.array(range(200))
        pyPyr = ppt.Lpyr(pyRamp)
        self.failUnless(ppu.comparePyr(matPyr['pyr'], pyPyr))

def main():
    unittest.main()

if __name__ == '__main__':
    main()


