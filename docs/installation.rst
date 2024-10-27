.. _label-installation:

Installation
============

These instructions will get you a copy of ``concreteproperties`` up and running on your
machine. You will need a working copy of python 3.10, 3.11 or 3.12 to get started.

Installing ``concreteproperties``
---------------------------------

``concreteproperties`` uses `shapely <https://github.com/shapely/shapely>`_ to prepare
the cross-section geometry and `CyTriangle <https://github.com/m-clare/cytriangle>`_ to
efficiently generate a conforming triangular mesh.
`sectionproperties <https://github.com/robbievanleeuwen/section-properties>`_ is used to
generate concrete geometries, while `numpy <https://github.com/numpy/numpy>`_ and
`scipy <https://github.com/scipy/scipy>`_ are used to aid computations, and
`matplotlib <https://github.com/matplotlib/matplotlib>`_ and
`rich <https://github.com/Textualize/rich>`_ are used for post-processing.

``concreteproperties`` and all of its dependencies can be installed through the python
package index:

.. code-block:: shell

    pip install concreteproperties

Using ``sectionproperties`` CAD Modules
---------------------------------------

To import geometry from CAD files, i.e. ``dxf`` or ``.3dm`` files, the optional CAD
extras must be installed. To install ``sectionproperties`` with the above functionality,
use the ``dxf`` and/or ``rhino`` options:

.. code-block:: shell

    pip install sectionproperties[dxf]
    pip install sectionproperties[rhino]
