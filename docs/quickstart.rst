Quick Start
*************

On Linux or macOS, open a shell and run::

  pip install pyrtools

More instructions available at :ref:`install`.

In the python interpreter, then call::

  import pyrtools as pt

which should run without errors if the install worked correctly. If
you have an issue with the installation, it will most likely be with
the compilation of the C code. There is hopefully a warning of this
when you import the library, but if you get an error message along the
lines of `lib not defined` when attempting to build a pyramid or call
the functions `corrDn`, `upConv`, or `pointOp`, this is probably
what's at fault.

Create pyramid::

  pyr = pt.pyramids.LaplacianPyramid(img)

Reconstruct image from pyramid::

  recon_img = pyr.recon_pyr()

For more details, see the jupyter notebooks included in the
`TUTORIALS/` directory, static versions of which are linked in the
navigation sidebar. You can play around with a live version of them in
order to test out the code before downloading on `binder
<https://mybinder.org/v2/gh/LabForComputationalVision/pyrtools/v0.9.0?filepath=TUTORIALS%2F>`_
