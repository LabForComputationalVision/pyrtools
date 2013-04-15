#!/usr/bin/python
import unittest
import pyPyrUtils as ppu
import numpy as np

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
        self.failUnless((ppu.binomialFilter(2) == np.matrix('0.5; 0.5')).all() )
    def test2(self):
        self.failUnless((ppu.binomialFilter(3) == np.matrix('0.25; 0.5; 0.25')).all())
    def test3(self):
        self.failUnless((ppu.binomialFilter(5) == np.matrix('0.0625; 0.25; 0.3750; 0.25; 0.0625')).all())

def main():
    unittest.main()

if __name__ == '__main__':
    main()
