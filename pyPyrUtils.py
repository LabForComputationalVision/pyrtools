import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pylab
import scipy.signal as spsig
import scipy.stats as sps
import math

def showIm(*args):
    if len(args) == 0:
        print "showIm( matrix, range, zoom, label, colormap, colorbar )"
        print "  matrix is string. It should be the name of a 2D array."
        print "  range is a two element tuple.  It specifies the values that "
        print "    map to the min and max colormap values.  Passing a value "
        print "    of 'auto' (default) sets range=[min,max].  'auto2' sets "
        print "    range=[mean-2*stdev, mean+2*stdev].  'auto3' sets "
        print "    range=[p1-(p2-p1)/8, p2+(p2-p1)/8], where p1 is the 10th "
        print "    percientile value of the sorted matix samples, and p2 is "
        print "    the 90th percentile value."
        print "  zoom specifies the number of matrix smples per screen pixel. "
        print "    It will be rounded to an integer, or 1 divided by an "
        print "    integer.  A value of 'same' or 'auto' (default) causes the "
        print "    zoom value to be chosen automatically to fit the image into "
        print "    the current axes.  A value of 'full' fills the axis region "
        print "    (leaving no room for labels)."
        print "  If label (optional, default = 1, unless zoom = 'full') is "
        print "    non-zero, the range of values that are mapped into the "
        print "    colormap and the dimensions (size) of the matrix and zoom "
        print "    factor are printed below the image.  If label is a string, "
        print "    it is used as a title."
        print "  colormap must contain the string 'auto' (grey colormap with " 
        print "    size = matrix.max() - matrix.min() will be used), "
        print "    or a string that is the name of a colormap variable "
        print "  colorbar is a boolean that specifies whether or not a "
        print "    colorbar is displayed"
    if len(args) > 0:   # matrix entered
        matrix = args[0]
        # defaults for all other values in case they weren't entered
        imRange = matrix.min(), matrix.max()
        zoom = 1
        label = 1
        nshades = 256
        colorbar = False
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
        # might work with tkinter, but then have to change everything
        zoom = 1
    if len(args) > 3:   # label entered
        label = args[3]
    if len(args) > 4:   # colormap entered
        if args[4] is "auto":
            colormap = cm.Greys_r
        else:  # got a variable name
            colormap = args[4]
    if len(args) > 5 and args[5]:   # colorbar entered and set to true
        colorbar = args[5]
        
    imgplot = plt.imshow(matrix, colormap).set_clim(imRange)
    plt.gca().invert_yaxis()  # default is inverted y from matlab
    if label != 0 and label != 1:
        plt.title(label)
    if colorbar:
        plt.colorbar(imgplot, cmap=cmap)
    #pylab.show()
    plt.show()
    
# Compute maximum pyramid height for given image and filter sizes.
# Specifically: the number of corrDn operations that can be sequentially
# performed when subsampling by a factor of 2.
def maxPyrHt(imsz, filtsz):
    if isinstance(imsz, (int,long)):
        done = True
    elif 1 in imsz:  # 1D image
        imsz = imsz[0] * imsz[1]
        filtsz = filtsz[0] * filtsz[1]
    elif 1 in filtsz: # 2D image, 1D filter
        filtsz = (filtsz[0], filtsz[0])

    if isinstance(imsz, (int,long)) and imsz == 0:  # imsz is int == 0
        height = 0
    elif not isinstance(imsz, (int,long)):  # imsz is tuple
        if any( i < f for i,f in zip(imsz, filtsz) ):
            height = 0
        else:
            imsz = ( int( math.floor(imsz[0]/2) ), int( math.floor(imsz[1]/2) ))
            height = 1 + maxPyrHt(imsz, filtsz)
    else:  # imsz is an int but != 0
        if imsz < filtsz:
            height = 0;
        else:
            imsz = int( math.floor(imsz/2) )
            height = 1 + maxPyrHt(imsz, filtsz)

    return height

# returns a vector of binomial coefficients of order (size-1)
# Rob Young, 4/13
#
def binomialFilter(size):
    if size < 2:
        print "Error: size argument must be larger than 1"
        exit(1)
    
    kernel = np.matrix('0.5; 0.5')

    for i in range(0, size-2):
        kernel = spsig.convolve(np.matrix('0.5; 0.5'), kernel)

    return kernel

# Some standard 1D filter kernels. These are scaled such that their L2-norm 
#   is 1.0
#
# binomN              - binomial coefficient filter of order N-1
# haar                - Harr wavelet
# qmf8, qmf12, qmf16  - Symmetric Quadrature Mirror Filters [Johnston80]
# daub2, daub3, daub4 - Daubechies wavelet [Daubechies88]
# qmf5, qmf9, qmf13   - Symmetric Quadrature Mirror Filters [Simoncelli88, 
#                                                            Simoncelli90]
# See bottom of file for full citations
#
# Rob Young, 4/13
#
#def namedFilter(name):
#    if len(name) > 5 and name[0:5] is "binom":
#        kernel = math.sqrt(2) * 
