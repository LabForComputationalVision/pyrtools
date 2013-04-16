import numpy as np
import pyPyrUtils as ppu

class pyr:  # pyramid
    # properties
    pyrType
    image
    height

    # methods - display?

class Gpyr(pyr):
    filt
    edges

    # constructor
    def __init__(*args):    # (image, height, filter, edges)
        if len(args) < 1:
            print "[pyr, indices] = Gpyr(image, height, filter, edges)"
        elif len(args) < 2:
            image = args[0]
            # defaults
            filt = namedFilter('binom5')
            ht = 1 + ppu.maxPyrHt(image.shape, filt.shape)
            edges = 

        pyr.image = pImage
        pyr.height = pHeight
        filt = pFilt
        edges = pEdges

    # medthods
    
