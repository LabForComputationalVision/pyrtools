Quick Start
*************

On Linux or macOS, open a shell and run::

  pip install pyrtools

More instructions available at :ref:`install`.

In the python interpreter, then call::

  import pyrtools as pt

which should run without errors if the install worked correctly.

Method parameters mimic the matlab function parameters except that
there's no need to pass `pyr` or `pind`.

Create pyramid::

  pyr = pt.pyramids.LaplacianPyramid(img)

Reconstruct image from pyramid::

  recon_img = pyr.recon_pyr()

For more details, see the jupyter notebooks included in the
`TUTORIALS/` directory, static versions of which are linked in the
navigation sidebar.
