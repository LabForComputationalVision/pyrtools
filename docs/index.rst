.. |pypi-shield| image:: https://img.shields.io/pypi/v/pyrtools.svg
			 :target: https://pypi.org/project/pyrtools/

.. |license-shield| image:: https://img.shields.io/badge/license-MIT-yellow.svg
			    :target: https://github.com/LabForComputationalVision/pyrtools/blob/main/LICENSE

.. |python-version-shield| image:: https://img.shields.io/badge/python-3.7%7C3.8%7C3.9%7C3.10-blue.svg

.. |build| image:: https://github.com/LabForComputationalVision/pyrtools/workflows/build/badge.svg
		     :target: https://github.com/LabForComputationalVision/pyrtools/actions?query=workflow%3Abuild

.. |binder| image:: https://mybinder.org/badge_logo.svg
		    :target: https://mybinder.org/v2/gh/LabForComputationalVision/pyrtools/v1.0.4?filepath=TUTORIALS%2F

.. |doi| image:: https://zenodo.org/badge/137527035.svg
  :target: https://zenodo.org/doi/10.5281/zenodo.10161031

.. pyrtools documentation master file, created by
   sphinx-quickstart on Mon Mar 25 17:57:12 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyrtools
====================================

|pypi-shield| |license-shield| |python-version-shield| |build| |binder| |doi|

Pyrtools is a python package for multi-scale image processing, adapted
from Eero Simoncelli's `matlabPyrTools
<https://github.com/LabForComputationalVision/matlabPyrTools/>`_.

The tools include:
  - Recursive multi-scale image decompositions (pyramids), including
    Laplacian pyramids, QMFs, Wavelets, and steerable pyramids.  These
    operate on 1D or 2D signals of arbitrary dimension.
  - Fast 2D convolution routines, with subsampling and boundary-handling.
  - Fast point-operations, histograms, histogram-matching.
  - Fast synthetic image generation: sine gratings, zone plates, fractals, etc.
  - Display routines for images and pyramids.  These include several
    auto-scaling options, rounding to integer zoom factors to avoid
    resampling artifacts, and useful labeling (dimensions and gray-range).

**NOTE**: If you are only interested in the complex steerable pyramid, we have a pytorch implementation in the `plenoptic <https://plenoptic.readthedocs.io/en/>`_ package; the implementation in plenoptic is differentiable.

Citing us
---------

If you use ``pyrtools`` in a published academic article or presentation, please
cite us! You can find the link to the most recent release on Zenodo `here
<https://zenodo.org/doi/10.5281/zenodo.10161031>`_ (though please specify the
version you used not the most recent one!). You can also get a formatted
citation at the top right of our `GitHub repo
<https://github.com/LabForComputationalVision/pyrtools>`_

.. include:: quickstart.rst

Pyramid resources
------------------

If you would like to learn more about pyramids and why they're helpful
for image processing, here are some resources to get you started:

 - Brian Wandell's `Foundations of Vision
   <https://foundationsofvision.stanford.edu/chapter-8-multiresolution-image-representations/>`_,
   chapter 8 (the rest of the book is helpful if you want to
   understand the basics of the visual system).
 - `Adelson et al, 1984, "Pyramid methods in image
   processing". <http://persci.mit.edu/pub_pdfs/RCA84.pdf>`_
 - Notes from David Heeger on `steerable filters
   <http://www.cns.nyu.edu/~david/handouts/steerable.pdf>`_
 - Notes from Eero Simoncelli on `the Steerable Pyramid
   <http://www.cns.nyu.edu/~eero/STEERPYR/>`_

	     
.. toctree::
   :maxdepth: 2

   installation
   developerguide
   api/modules

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :glob:
   :numbered:

   tutorials/*
