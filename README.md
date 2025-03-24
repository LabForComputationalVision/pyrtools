# pyrtools: tools for multi-scale image processing

[![PyPI Version](https://img.shields.io/pypi/v/pyrtools.svg)](https://pypi.org/project/pyrtools/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pyrtools/badges/version.svg)](https://anaconda.org/conda-forge/pyrtools)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/LabForComputationalVision/pyrtools/blob/main/LICENSE)
![Python version](https://img.shields.io/badge/python-3.8|3.9|3.10|3.11|3.12-blue.svg)
[![Build Status](https://github.com/LabForComputationalVision/pyrtools/workflows/build/badge.svg)](https://github.com/LabForComputationalVision/pyrtools/actions?query=workflow%3Abuild)
[![Documentation Status](https://readthedocs.org/projects/pyrtools/badge/?version=latest)](https://pyrtools.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/137527035.svg)](https://zenodo.org/doi/10.5281/zenodo.10161031)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LabForComputationalVision/pyrtools/v1.0.7?filepath=TUTORIALS%2F)
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

This is a python 3 port of Eero Simoncelli's
[matlabPyrTools](https://github.com/LabForComputationalVision/matlabPyrTools),
but it does not attempt to recreate all of the matlab code from matlabPyrTools.
The goal is to create a Python interface for the C code at the heart of
matlabPyrTools.

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

You can install `pyrtools` using either pip:

```sh
pip install pyrtools
```

or conda:

```sh
conda install pyrtools -c conda-forge
```

You may also install from source, directly from the git repository. This is
largely useful if you are seeking to modify the code or make contributions. To
do so, clone the repository and run `pip install`. On Mac or Linux, that looks
like:

``` sh
git clone https://github.com/LabForComputationalVision/pyrtools.git
cd pyrtools/
pip install .
```

You may also want an editable install, `pip install -e .`, in which case changes
you make in the source code will be reflected in your install.

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

# Usage:

- load modules:
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

Please see `TUTORIALS/02_pyramids.ipynb` for more examples.

# For developres

## Testing

You can find unit tests in `TESTS/unitTests.py` and run them with `python
TESTS/unitTests.py`.

## Build the documentation

NOTE: If you just want to read the documentation, you do not need to
do this; documentation is built automatically on
[readthedocs](https://pyrtools.readthedocs.io/en/latest/).

However, it can be built locally as well. You would do this if you've
made changes locally to the documentation (or the docstrings) that you
would like to examine before pushing.

```
# create a new virtual environment and then...
# install pyrtools with sphinx and documentation-related dependencies
pip install -e .[docs]
# build documentation
cd docs/
make html
```

The index page of the documentation will then be located at
`docs/_build/html/index.html`, open it in your browser to navigate
around.
