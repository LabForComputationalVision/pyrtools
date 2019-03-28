# pyrtools: tools for multi-scale image processing

Briefly, the tools include:
  - Recursive multi-scale image decompositions (pyramids), including
    Laplacian pyramids, QMFs, Wavelets, and steerable pyramids.  These
    operate on 1D or 2D signals of arbitrary dimension.
  - Fast 2D convolution routines, with subsampling and boundary-handling.
  - Fast point-operations, histograms, histogram-matching.
  - Fast synthetic image generation: sine gratings, zone plates, fractals, etc.
  - Display routines for images and pyramids.  These include several
    auto-scaling options, rounding to integer zoom factors to avoid
    resampling artifacts, and useful labeling (dimensions and gray-range).

# Installation

It's recommended you install from pip: `pip install pyrtools`. The pip
install has been tested on Linux and on OSX. Windowsis NOT supported
because of issues with the C compiler (`gcc` isn't necessarily
installed); if you have experience with C compilation on Windows,
please open a pull request.


## Dependencies
Other requirements:
 - numpy
 - scipy
 - matplotlib
 - Pillow
 - tqdm
 - requests

IPython is optional. If it's not installed,
`pyrtools.display_tools.animshow` must be called with `as_html5=False`
(but since this is for displaying the animated image in a Jupyter /
IPython notebook, you probably won't need that functionality).

# Authors

Rob Young and Eero Simoncelli, 7/13

William Broderick, 6/17

A python 3.6 port of Eero Simoncelli's matlabPyrTools. This port does
not attempt to recreate all of the matlab code from matlabPyrTools.
The goal is to create a Python interface for the C code at the heart
of matlabPyrTools.

# Usage:

method parameters mimic the matlab function parameters except that there's no
need to pass pyr or pind, since the pyPyrTools version pyr and pyrSize are
properties of the class.

- load modules (note that if you installed via pip, you can skip the
  first two lines):
```
import pyrtools as pt
```

- create pyramid:
```
pyr = pt.pyramids.LaplacianPyramid(img)
```

- reconstruct image from pyramid:
```
recon_img = pyr.recon_pyr()
```

Please see `TUTORIALS/02_pyramids.ipynb` for more examples.  You can
start this with: `jupyter notebook 02_pyramids.ipynb` if you have iPython
and Jupyter installed.

# Testing

All code should be considered a beta release.  By that we mean that it is being
actively developed and tested.  You can find unit tests in
`TESTING/unitTests.py`.
and run
`python unitTests.py`.

If you're using functions or parameters that do not have associated unit
tests you should test this yourself to make sure the results are correct.
You could then submit your test code, so that we can build more complete
unit tests.
