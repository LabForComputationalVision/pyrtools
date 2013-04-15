#!/usr/bin/python
import unittest
import pyPyrUtils as ppu

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

def main():
    unittest.main()

if __name__ == '__main__':
    main()
