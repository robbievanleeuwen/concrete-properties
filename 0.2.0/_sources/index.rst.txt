.. concreteproperties documentation master file, created by
   sphinx-quickstart on Tue Feb 22 13:36:09 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/cp_logo_page.png
 :width: 100 %
 :alt: concreteproperties
 :align: left
 :class: only-light

.. image:: _static/cp_logo_page_dark.png
 :width: 100 %
 :alt: concreteproperties
 :align: left
 :class: only-dark

|Run Tests| |Lint with Black| |Build Documentation| |codecov| |PyPI version| |Python versions| |GitHub license|


Documentation
=============

*concreteproperties* is a python package that can be used to calculate the section
properties of arbitrary reinforced concrete sections. *concreteproperties* can calculate
gross, cracked and ultimate properties. It can perform moment curvature analyses
and generate moment interaction and biaxial bending diagrams. On top of this,
*concreteproperties* can also generate pretty stress plots!

Here's an example of some of the non-linear output *concreteproperties* can generate:

.. image:: _static/anim/anim_compress.gif
 :width: 67 %
 :alt: concreteproperties
 :align: center

A list of the `current features of the package and implementation goals for future
releases <https://github.com/robbievanleeuwen/concrete-properties/tree/master/README.md>`_
can be found in the README file on github.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   rst/installation
   rst/materials
   rst/geometry
   rst/analysis
   rst/results
   rst/design_codes
   rst/examples
   rst/api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |Run Tests| image:: https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/tests.yml
.. |Lint with Black| image:: https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/black.yml/badge.svg
   :target: https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/black.yml
.. |Build Documentation| image:: https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/build_docs.yml/badge.svg
   :target: https://robbievanleeuwen.github.io/concrete-properties/
.. |codecov| image:: https://codecov.io/gh/robbievanleeuwen/concrete-properties/branch/master/graph/badge.svg?token=3WXMUQITTD
   :target: https://codecov.io/gh/robbievanleeuwen/concrete-properties
.. |PyPI version| image:: https://badge.fury.io/py/concreteproperties.svg
   :target: https://badge.fury.io/py/concreteproperties
.. |Python versions| image:: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue?style=flat&logo=python
   :target: https://badge.fury.io/py/concreteproperties
.. |GitHub license| image:: https://img.shields.io/github/license/robbievanleeuwen/concrete-properties
   :target: https://github.com/robbievanleeuwen/concrete-properties/blob/master/LICENSE.md
