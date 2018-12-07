Attitude Kinematics in Python
=======================

``kinematics`` is Python package to perform attitude kinematics.
It is written completely in Python and only requires ``numpy`` as a runtime 
dependency.

+-------------------------+---------------------+--------------------------+------------+
| Continuous Integration  | Code Coverage       | Docs                     | Citation   |
+=========================+=====================+==========================+============+
| |Travis Build Status|   | |Coverage Status|   | |Documentation Status|   | |Citation| |
+-------------------------+---------------------+--------------------------+------------+

.. |Travis Build Status| image:: https://travis-ci.org/skulumani/kinematics.svg?branch=master
    :target: https://travis-ci.org/skulumani/kinematics
.. |Coverage Status| image:: https://coveralls.io/repos/github/skulumani/kinematics/badge.svg?branch=master
   :target: https://coveralls.io/github/skulumani/kinematics?branch=master
.. |Documentation Status| image:: https://readthedocs.org/projects/kinematics/badge/?version=latest
    :target: http://kinematics.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |Citation| image:: https://zenodo.org/badge/82479376.svg
   :target: https://zenodo.org/badge/latestdoi/82479376

Installation
============

Install ``kinematics`` by running : ``pip install kinematics`` to install from pypi

To install a development version (for local testing), you can clone the 
repository and run ``pip install -e .`` from the source directory.

Documentation
=============

Docs will be hosted on Read the Docs

Update travis to do the build, install, and test for both pypi install and conda install

conda build

conda convert

anaconda uplaod

Citing ``kinematics``
================

If you find this package useful, it would be very helpful to cite it in your work.
You can find a citation link above.

Dependencies
============

The only hard dependency is on ``numpy``. 
All vectors and operations utilize the numerical tools of numpy.
You should already have it installed, ``pip install numpy``.
