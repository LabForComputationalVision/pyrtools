.. _install:

Installation
************

There are two ways to install pyrtools: via the `pip` package
management system, or directly from source. Note that we only support
Linux and macOS; we've had issues compiling the C code on Windows. If
you know how to fix this, please open up a Pull Request
(:ref:`dev-guide`). Note that we require `gcc` version >= 6 in order
for the C code to compile, because of `this
issue<https://stackoverflow.com/questions/46504700/gcc-compiler-not-recognizing-fno-plt-option>`_

Recommended
===========

In a shell, please run::

    pip install pyrtools

From source
===========

Obtain the latest version of pyrtools::

    git clone https://github.com/LabForComputationalVision/pyrtools

Finally, the package is installed by running::

    cd pyrtools
    python setup.py -e .

This will install an editable version of the package, so changes made
to the files within the pyrtools directory will be reflected in the
version of pyrtools you use.
