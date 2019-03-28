.. |license-shield| image:: https://img.shields.io/badge/license-MIT-blue.svg

.. |python-version-shield| image:: https://img.shields.io/badge/python-3.5%7C3.6%7C3.7-blue.svg

.. pyrtools documentation master file, created by
   sphinx-quickstart on Mon Mar 25 17:57:12 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyrtools
====================================

|license-shield| |python-version-shield|

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

.. include:: quickstart.rst
    
.. toctree::
   :maxdepth: 2

   installation
   developerguide
   tutorial1
   tutorial2
   tutorial3
   modules
