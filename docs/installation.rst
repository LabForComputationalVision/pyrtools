.. _install:

Installation
************

There are two ways to install pyrtools: via the ``pip`` package management
system, or directly from source.

.. attention:: Windows support was added in version 1.0.3. If you are on Windows and get an installation error, make sure you are installing the newest version.

Recommended
===========

In a shell, please run::

    pip install pyrtools

From source
===========

Obtain the latest version of pyrtools::

    git clone https://github.com/LabForComputationalVision/pyrtools

(If you have already cloned the repo, you can update it with ``git pull``.)

Finally, the package is installed by running::

    cd pyrtools
    pip install -e .

This will install an editable version of the package, so changes made
to the files within the pyrtools directory will be reflected in the
version of pyrtools you use.

When installing from source on Linux or Mac, we require ``gcc`` version >= 6 in
order for the C code to compile, because of `this issue
<https://stackoverflow.com/questions/46504700/gcc-compiler-not-recognizing-fno-plt-option>`_

When installing from source on Windows, Microsoft Visual C++ 14.0 or greater is required, which can be obtained with `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_.
