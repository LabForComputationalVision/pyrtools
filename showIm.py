import matplotlib.pyplot as plt
import numpy as np
import pylab
#import scipy as sp
import scipy.stats as sps

def showIm(*args):
    if len(args) == 0:
        print "showIm( matrix, range, zoom, label, nshades, colorbar )\n"
        # FIX: add description of function and parameters
    if len(args) > 0:   # matrix entered
        matrix = args[0]
        # defaults for all other values in case they weren't entered
        imRange = matrix.min(), matrix.max()
        zoom = 1;
        label = 1;
    if len(args) > 1:   # range entered
        if isinstance(args[1], basestring):
            if args[1] is "auto":
                imRange = matrix.min(), matrix.max()
            elif args[1] is "auto2":
                imRange = ( matrix.mean()-2*matrix.std(), 
                            matrix.mean()+2*matrix.std() )
            elif args[1] is "auto3":
                #p1 = np.percentile(matrix, 10)  not in python 2.6.6?!
                #p2 = np.percentile(matrix, 90)
                p1 = sps.scoreatpercentile(np.hstack(matrix), 10)
                p2 = sps.scoreatpercentile(np.hstack(matrix), 90)
                imRange = p1-(p2-p1)/8, p2+(p2-p1)/8
            else:
                print "Error: range of %s is not recognized." % args[1]
                print "       please use a two element tuple or "
                print "       'auto', 'auto2' or 'auto3'"
                print "       enter 'showIm' for more info about options"
                return
        else:
            imRange = args[1][0], args[1][1]
    if len(args) > 2:   # zoom entered
        # no equivalent to matlab's pixelAxes in matplotlib. need dpi
        zoom = 1;
    if len(args) > 3:   # label entered
        
        
    imgplot = plt.imshow(matrix).set_clim(imRange)
    plt.gca().invert_yaxis()  # default is inverted y from matlab
    if label != 0 && label != 1:
        plt.title(label)
    pylab.show()
