Installation Instructions
===========================

The ``kinematics`` package is simple to install. 
It is available from both ``pypi`` as well as ``anaconda``.
Furthermore, the only hard dependency is on the ``numpy`` package which is mostly likely already being used in your project.

Installing
===========

To install using ``pip``, you can run::

    pip install kinematics

Instead if you're already using the [Anaconda](www.anaconda.org) distribution, then it is preferrable to use ``conda`` to manage your installed packages.

This is beneficial as ``conda`` provides many additional tools to build independent environments and duplicate these environments between many systems. Here is a good `blog post <https://www.anaconda.com/blog/developer-blog/using-pip-in-a-conda-environment/>`_ describing some of the approaches to utilizing ``pip`` and ``conda``.

To instead install using ``conda`` simply use::
    
    conda install -c skulumani kinematics

Building from source
===================

The package has been extensively tested on both OSX and Linux (Ubuntu). 
Binary distributions are provided which should also allow it to installed and used on Windows but this has not been tested. 

To build from source, one should first clone the repository::

    git clone https://github.com/skulumani/kinematics.git

Ensure that you have ``numpy`` installed on your system::

    pip install numpy

or if you're using ``conda`` create a new enviornment with the appropriate dependencies::

    conda create -n kinematics_env python=3 numpy

With the correct dependencies you can then install a development version of the package to ease development::

    cd kinematics
    pip setup.py -e .


Testing
===============

The package has a series of unit tests, located in the ``tests`` directory.
You can run the tests yourself using ``pytest``::

    pytest --vv --pyargs=kinematics






