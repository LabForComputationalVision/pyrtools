pyPyrTools - A port of Eero Simoncelli's matlabPyrTools to Python.  This port
	     does not attept to recreate all of the matlab code from 
	     matlabPyrTools.  The goal is to create a Python interface for 
	     the C code at the heart of matlabPyrTools.

Rob Young and Eero Simoncelli, 7/13

All code should be considered a beta release.  By that we mean that it is being
actively developed and tested.  You can find unit tests in
TESTING/unitTests.py.
If you're using functions or parameters that do not have associated unit
tests you should test this yourself to make sure the results are correct.
You could then submit your test code, so that we can build more complete
unit tests.


Usage:

method parameters mimic the matlab function parameters except that there's no 
need to pass pyr or pind, since the pyPyrTools version pyr and pyrSize are 
properties of the class.

- load modules:
  >> import sys
  >> sys.path.append('path to pyPyrTools parent directory')
  >> import pyrPyrTools as ppt

- create pyramid:
  >> myPyr = ppt.Lpyr(img)

- reconstruct image from pyramid:
  >> reconImg = myPyr.reconLpyr()

Please see TUTORIALS/pyramids.ipynb for more examples.  You can start this with:
jupyter notebook pyramids.ipynb
if you have iPython and Jupyter installed.

compiling the associated C code:
assuming your path to python libraries is /usr/local/anaconda2, you would type:
gcc -shared -L/usr/local/anaconda2/lib -I/usr/local/anaconda2/include/python2.7/ -lpython2.7 -o wrapConv.so -fPIC convolve.c edges.c wrap.c internal_pointOp.c

function list:

+ showIm(image, range, label, colormap, colorbar)
  display an image in a figure window
  - image - a 2D numpy array
  - range - a two element tuple. It specifies the values that map to the min and
    	    max colormap value.  Passing a value of 'auto' (default) sets 
	    range=[min,max]. 'auto2' sets range=[mean-2*stdev, mean+2*stdev].
	    'auto3' sets range=[p1-(p2-p1)/8, p2+(p2-p1)/8], where p1 is the 
	    10th percentile value of the sorted matrix samples, and p2 is the 
	    90th percentile value.
  - label - a string that is used as a figure title
  - colormap - either the string 'auto' (grey colormap will be used) or a 
      	       string that is the name of a colormap variable
  - colorbar - a boolean that specifies whether or not a colorbar is displayed

+ maxPyrHt(imsz, filtsz)
  return the maximum possible pyramid height from the given image and filter 
  size
  - imsz - integer giving the image size
  - filtsz - integer giving the filter size

+ binomial filter
  returns a numpy array of binomial coefficients of order (size-1)
  - size - integer giving size of filter

+ namedFilter(name)
  returns numpy array of named filter.  Supported names are: binom<num> 
  (where <num> is a number denoting the size of the filter, qmf5, qmf9, qmf13, 
  qmf8, qmf12, qmf16, haar, daub2, daub3, daub4, gauss5, gauss3.

+ mkRamp(size, direction, slope, intercept, origin)
  Compute a matrix of dimension "size" (a (Y X) tuple, or a scalar (for square
  matrices)) containing samples of a ramp function, with a given gradient
  "direction" (radians, CW from X-axis, default=0), "slope" (per pixel, default
  = 1), and a value of "intercept" (default = 0) at the "origin" (default = 
  (size+1)/2, (1 1) = upper left). All but the first argument are optional.

+ sp0Filters - steerable pyramid filters (see pyPyrUtils.py for references)

+ sp1Filters - steerable pyramid filters (see pyPyrUtils.py for references)

+ mkImpulse(size, origin, amplitude)
  Compute a matrix of dimension "size" (a (Y X) tuple, or a scalar (for square
  matrices)) containing a single non-zero entry, at position "origin" (defaults 
  to ceil(size/2)), of value "amplitude" (defaults to 1).

+ Lpyr - Laplacian pyramid
  - constructor - (image, pyramid height, filter1, filter2, edges)
    		  image parameter is required, others are optional
		  image - a 2D numpy array
		  pyramid height - an integer denoting number of pyramid levels
		       	  	   desired.  Defaults to maxPyrHt from 
				   pyPyrUtils.
		  filter1 - can be a string namimg a standard filter (from 
		  	   pyPyrUtils.namedFilter()), or a numpy array which 
			   will be used for (separable) convolution. Default is
			   'binom5'.
		  filter2 - specifies the "expansion" filter (default = filt1).
		  edges - specifies edge-handling.  Options are:
		  	  * circular - circular convolution
			  * reflect1 - reflect about the edge pixels
			  * reflect2 - reflect, doubling the edge pixels
			  * repeat - repeat the edge pixels
			  * zero - assume values of zero outside image boundary
			  * extend - reflect and invert
			  * dont-compute - zero output when filter overhangs
			    		   imput boundaries.
		  returns pyramid object
   - reconLpyr(levs, filt2, edges)
     Reconstruct image from Laplacian pyramid object
     levs (optional) - numpy array of levels to include, or the string 'all'
                       (default). The finest scale is 0.
     filt2 (optional) - valid string name for pyPyrUtils.namedFilter() or a
                        numpy array which will be used for (separable) 
			convolution. Default = 'binom5'.
     edges (optional) - same as edges for constructor above.
   - pyrLow()
     Returns the coarsest band.
   - showPyr(range, gap, level_scale_factor)
     Show all bands of the pyramid in a figure window.
     range - a two element tuple specifying the values that map to black and 
             white respectively. These values are scaled by 
	     level_scale_factor**(lev-1) for bands at each level. Passing a 
	     value of 'auto1' sets range to the min and max values of matrix.
	     'auto2' sets range to 3 standard deviations below and above 0.0.  
	     In both of these cases, the lowpass band is independently scaled.
	     A value of 'indep1' sets the range of each subband independently,
	     as in a call to pyPyrUtils.showIm(subband, 'auto1').  Similarly,
	     'indep2' causes each subband to be scaled independently as if by 
	     pyPyrUtils.showIm(subband, 'indep2').  The default value for range
	     is 'auto1' for 1D images and 'auto2' for 2D images.
     gap - specifies the gap in pixels to leave between subbands (2D images 
           only). default = 1.
     level_scale_factor - indicates the relative scaling between pyramid levels.
                          This should be set to the sum of the kernel taps of 
			  the lowpass filter used to construct the pyramid 
			  (default assumes L2-normalized filters, using a value
			  of 2 for 2D images, sqrt(2) for 1D images).
    
+ Gpyr - Gaussian pyramid
  (subclass of Lpyr)
  - constructor - (image, pyramid height, filter, edges)
    		  image parameter is required, others are optional
		  image - a 2D numpy array
		  pyramid height - an integer denoting number of pyramid levels
		       	  	   desired.  Defaults to maxPyrHt from 
				   pyPyrUtils.
		  filter - can be a string namimg a standard filter (from 
		  	   pyPyrUtils.namedFilter()), or a numpy array which 
			   will be used for (separable) convolution. Default is
			   'binom5'.
		  edges - specifies edge-handling.  Options are:
		  	  * circular - circular convolution
			  * reflect1 - reflect about the edge pixels
			  * reflect2 - reflect, doubling the edge pixels
			  * repeat - repeat the edge pixels
			  * zero - assume values of zero outside image boundary
			  * extend - reflect and invert
			  * dont-compute - zero output when filter overhangs
			    		   imput boundaries.
		  returns pyramid object

+ Spyr - steerable pyramid
  - constructor - (image, pyramid height, filter, edges)
    		  image parameter is required, others are optional
		  image - a 2D numpy array
		  pyramid height - an integer denoting number of pyramid levels
		       	  	   desired.  Defaults to maxPyrHt from 
				   pyPyrUtils. You can specify 'auto' to use
				   this value.
		  filter - The name of one of the steerable pyramid filters in
		           pyPyrUtils: sp0Filters, sp1Filters, sp3Filters, 
			   sp5Filters.  Default is sp1Filters.
		  edges - specifies edge-handling.  Options are:
		  	  * circular - circular convolution
			  * reflect1 - reflect about the edge pixels
			  * reflect2 - reflect, doubling the edge pixels
			  * repeat - repeat the edge pixels
			  * zero - assume values of zero outside image boundary
			  * extend - reflect and invert
			  * dont-compute - zero output when filter overhangs
			    		   imput boundaries.
		  returns pyramid object
  - spyrHt() - return the height of the pyramid
  - numBands() - return the number of bands in the pyramid
  - pyrLow() - return the lowest band
  - pyrHight() - return the highest band
  - reconSpyr(filter, edges, levs, bands) - reconstruct image from pyramid
    filter - (optional) same as for constructor above. Default is 'sp1Filters'.
    edges - (optional) same as for constructor above. Default is 'reflect1'.
    levs - (optional) should be a numpy array  of levels to include, or the 
     	   string 'all' (default). 0 corresponds to the residual highpass 
	   subband. 1 corresponds to number spyrHt()+1. 
    bands - (optional) should be a list of bands to include, or the string 
            'all' (default). 0 = vertical, rest proceeding anti-clockwise.
           

