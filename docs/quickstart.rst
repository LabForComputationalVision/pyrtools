Quick Start
*************

On Linux or macOS, open a shell and run::

  pip install pyrtools

More instructions available at :ref:`install`.

In the python interpreter, then call::

  import pyrtools as pt

Create pyramid::

  pyr = pt.pyramids.LaplacianPyramid(img)

Reconstruct image from pyramid::

  recon_img = pyr.recon_pyr()

For more details, see the jupyter notebooks included in the
``TUTORIALS/`` directory, static versions of which are linked in the
navigation sidebar. You can play around with a live version of them in
order to test out the code before downloading on `binder
<https://mybinder.org/v2/gh/LabForComputationalVision/pyrtools/v1.0.4?filepath=TUTORIALS%2F>`_
