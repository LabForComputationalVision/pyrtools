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
            print "First argument (image) is required"
            exit(1)
        else:
            pyr.image = args[0]

        if len(args) > 1:
            filt = args[1]
            if not (filt.shape == 1).any():
                print "Error: filt should be a 1D filter (i.e., a vector)"
                exit(1)
        else:
            filt = namedFilter('binom5')

        if len(args) > 2:
            maxHeight = 1 + ppu.maxPyrHt(image.shape, filt.shape)
            if args[2] is "auto":
                pyr.ht = maxHeight
            else:
                pyr.ht = args[2]
                if pyr.ht > maxHeight:
                    ( print "Error: cannot build pyramid higher than %d levels"
                      % (maxHeight) )
                    exit(1)
        else:
            pyr.ht = maxHeight

        if len(args) > 3:
            edges = args[3]
        else:
            edges = "reflect1"

        # make pyramid
        

    # medthods
    
