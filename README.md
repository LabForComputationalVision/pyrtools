# pyrtools: tools for multi-scale image processing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/LabForComputationalVision/pyrtools/blob/master/LICENSE)
![Python version](https://img.shields.io/badge/python-3.5%7C3.6%7C3.7-blue.svg)
[![Build Status](https://travis-ci.com/LabForComputationalVision/pyrtools.svg?branch=master)](https://travis-ci.com/LabForComputationalVision/pyrtools)
[![Documentation Status](https://readthedocs.org/projects/pyrtools/badge/?version=latest)](https://pyrtools.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LabForComputationalVision/pyrtools/master?filepath=TUTORIALS%2F)

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

This is a python 3 port of Eero Simoncelli's matlabPyrTools, but it
does not attempt to recreate all of the matlab code from
matlabPyrTools. The goal is to create a Python interface for the C
code at the heart of matlabPyrTools.

# Installation

It's recommended you install from pip: `pip install pyrtools`. The pip
install has been tested on Linux and on OSX. Windows is NOT supported
because of issues with the C compiler (`gcc` isn't necessarily
installed); if you have experience with C compilation on Windows,
please open a pull request. It's possible that the way to fix this is
to use Cython, ensuring that Cython is installed before attempting to
run the pip command, and then adding: `from Cython.Build import
cythonize` and wrapping the `ext_modules` in the `setup` call with
`cythonize`, but I'm not sure.

If you wish to install from the master branch, it's still recommended
to use pip, just run `pip install .` (or `pip install -e .` if you
want the changes you make in the directory to be reflected in your
install) from the root directory of this project. The core of this
code is the C code, and the pip install will compile it nicely.

## Dependencies

Python 3.5, 3.6, and 3.7 all officially supported.

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

William Broderick, Pierre-Étienne Fiquet, Zhuo Wang, Zahra Kadkhodaie,
Nikhil Parthasarathy, and the Lab for Computational Vision, 4/19

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

# Build the documentation

Documentation is built automatically on readthedocs, but can be built
locally as well. The virtual environment required to do so is defined
in `docs/environment.yml`, so to create that environment and build the
docs, do the following from the project's root directory:

```
# install sphinx and required packages to build documentation
conda env create -f docs/environment.yml
# install pyrtools
pip install .
# build documentation
cd docs/
make html
```

The index page of the documentation will then be located at
`docs/_build/html/index.html`, open it in your browser to navigate
around.
