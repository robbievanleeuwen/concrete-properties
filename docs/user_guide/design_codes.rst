Design Codes
============

The design code module allows ``concreteproperties`` to be easily used in the context of
common reinforced concrete design standards. ``concreteproperties`` currently supports the
following design codes:

.. toctree::
   :maxdepth: 1

   design_codes/as3600
   design_codes/nzs3101

.. warning::

   The current implementation of moment interaction diagrams in ``concreteproperties``
   allows the user to define the angle the neutral axis makes with the horizontal.
   Asymmetric sections with a non-zero neutral axis angle will result in biaxial
   bending moments. When generating moment interaction diagrams using this approach, the
   ratio between the bending moments ``m_x`` and ``m_y`` will change depending on the
   level of axial load. As a result, without generating biaxial bending diagrams,
   a false impression of demand/capacity may be generated when operating close to the
   design curve. A future version of ``concreteproperties`` will incorporate a constant
   'load-angle' approach, which will keep the ratio of ``m_x`` to ``m_y`` constant
   for a given moment interaction diagram, rather than keeping the neutral axis angle
   constant. Further discussion of this issue can be found
   `here <https://github.com/robbievanleeuwen/concrete-properties/discussions/31>`_.
