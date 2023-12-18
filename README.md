# pyrtools: tools for multi-scale image processing

[![PyPI Version](https://img.shields.io/pypi/v/pyrtools.svg)](https://pypi.org/project/pyrtools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/LabForComputationalVision/pyrtools/blob/main/LICENSE)
![Python version](https://img.shields.io/badge/python-3.7|3.8|3.9|3.10-blue.svg)
[![Build Status](https://github.com/LabForComputationalVision/pyrtools/workflows/build/badge.svg)](https://github.com/LabForComputationalVision/pyrtools/actions?query=workflow%3Abuild)
[![Documentation Status](https://readthedocs.org/projects/pyrtools/badge/?version=latest)](https://pyrtools.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/137527035.svg)](https://zenodo.org/doi/10.5281/zenodo.10161031)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LabForComputationalVision/pyrtools/v1.0.3?filepath=TUTORIALS%2F)
[![codecov](https://codecov.io/gh/LabForComputationalVision/pyrtools/branch/main/graph/badge.svg?token=Ei9TYftdYi)](https://codecov.io/gh/LabForComputationalVision/pyrtools)

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

**NOTE**: If you are only interested in the complex steerable pyramid, we have a
pytorch implementation in the
[plenoptic](https://github.com/LabForComputationalVision/plenoptic/) package;
the implementation in plenoptic is differentiable.

# Citing us

If you use `pyrtools` in a published academic article or presentation, please
cite us! You can find the link to the most recent release on Zenodo
[here](https://zenodo.org/doi/10.5281/zenodo.10161031) (though please specify
the version you used not the most recent one!). You can also get a formatted
citation at the top right of our [GitHub
repo](https://github.com/LabForComputationalVision/pyrtools)

# Installation

It's recommended you install from pip: `pip install pyrtools`.

If you wish to install from the main branch, it's still recommended
to use pip, just run `pip install .` (or `pip install -e .` if you
want the changes you make in the directory to be reflected in your
install) from the root directory of this project. The core of this
code is the C code, and the pip install will compile it nicely.

## Dependencies

Dependencies are documented in `setup.py`.

IPython is optional. If it's not installed,
`pyrtools.display_tools.animshow` must be called with `as_html5=False`
(but since this is for displaying the animated image in a Jupyter /
IPython notebook, you probably won't need that functionality).

For the C code to compile, we require `gcc` version >= 6, because of
[this
issue](https://stackoverflow.com/questions/46504700/gcc-compiler-not-recognizing-fno-plt-option)

# Pyramid resources

If you would like to learn more about pyramids and why they're helpful
for image processing, here are some resources to get you started:

 - Brian Wandell's [Foundations of
   Vision](https://foundationsofvision.stanford.edu/chapter-8-multiresolution-image-representations/),
   chapter 8 (the rest of the book is helpful if you want to
   understand the basics of the visual system).
 - [Adelson et al, 1984, "Pyramid methods in image
   processing".](http://persci.mit.edu/pub_pdfs/RCA84.pdf)
 - Notes from David Heeger on [steerable
   filters](http://www.cns.nyu.edu/~david/handouts/steerable.pdf)
 - Notes from Eero Simoncelli on [the Steerable
   Pyramid](http://www.cns.nyu.edu/~eero/STEERPYR/)

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
`TESTS/unitTests.py` and run them with `python TESTS/unitTests.py`.

If you're using functions or parameters that do not have associated unit
tests you should test this yourself to make sure the results are correct.
You could then submit your test code, so that we can build more complete
unit tests.

# Build the documentation

NOTE: If you just want to read the documentation, you do not need to
do this; documentation is built automatically on
[readthedocs](https://pyrtools.readthedocs.io/en/latest/).

However, it can be built locally as well. You would do this if you've
made changes locally to the documentation (or the docstrings) that you
would like to examine before pushing. The virtual environment required
to do so is defined in `docs/environment.yml`, so to create that
environment and build the docs, do the following from the project's
root directory:

```
# install sphinx and required packages to build documentation
conda env create -f docs/environment.yml
# activate the environment
conda activate pyrtools_docs
# install pyrtools
pip install -e .
# build documentation
cd docs/
make html
```

The index page of the documentation will then be located at
`docs/_build/html/index.html`, open it in your browser to navigate
around.

The `pyrtools_docs` environment you're creating contains the package
`sphinx` and several extensions for it that are required to build the
documentation. You also need to install `pyrtools` from your local
version so that `sphinx` can import the library and grab all of the
docstrings (you're installing the local version so you can see all the
changes you've made).
