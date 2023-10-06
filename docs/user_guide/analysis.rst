.. _label-analysis:

Analysis
========

This section of the documentation outlines how to perform analyses in
``concreteproperties``. The :ref:`label-results` section outlines how to retrieve and
display the results obtained from these analyses.

An analysis in ``concreteproperties`` begins by creating a
:class:`~concreteproperties.concrete_section.ConcreteSection` object from a
:class:`~sectionproperties.pre.geometry.CompoundGeometry` object with assigned
material properties.

..  autoclass:: concreteproperties.concrete_section.ConcreteSection
  :noindex:
  :members: __init__

If a prestressed concrete section is being analysed, a
:class:`~concreteproperties.prestressed_section.PrestressedSection` object must be used
instead of a :class:`~concreteproperties.concrete_section.ConcreteSection` object, see
:ref:`label-prestressed-analysis`.

.. warning::

  If the cross-section geometry contains a
  :class:`~concreteproperties.material.SteelStrand` material object, a ``ValueError``
  will be raised if trying to create a
  :class:`~concreteproperties.concrete_section.ConcreteSection` object.


Visualising the Cross-Section
-----------------------------

The :class:`~concreteproperties.concrete_section.ConcreteSection` object can be
visualised by calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.plot_section` method.

..  automethod:: concreteproperties.concrete_section.ConcreteSection.plot_section
  :noindex:


Gross Area Properties
---------------------

Upon creating a :class:`~concreteproperties.concrete_section.ConcreteSection` object,
``concreteproperties`` will automatically calculate the area properties based on the gross
reinforced concrete cross-section.

.. seealso::
  For an application of the above, see the example
  :ref:`/examples/area_properties.ipynb`.


Cracked Area Properties
-----------------------

The area properties of the cracked cross-section can be determined by calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_cracked_properties`
method. By default the cracked properties are calculated for bending about the ``x``
axis, but this can be modified by providing a bending axis angle ``theta``.

..  automethod:: concreteproperties.concrete_section.ConcreteSection.calculate_cracked_properties
  :noindex:

The cracking moment is determined assuming cracking occurs once the stress in the
concrete reaches the ``flexural_tensile_strength``. Cracked properties are calculated
assuming the concrete is linear elastic and can only resist compression.

.. seealso::
  For an application of the above, see the example
  :ref:`/examples/cracked_properties.ipynb`.


Moment Curvature Analysis
-------------------------

A moment curvature analysis can be performed on the reinforced concrete cross-section
by calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.moment_curvature_analysis`
method. By default the moment curvature analysis is calculated for bending about the
``x`` axis, but this can be modified by providing a bending axis angle ``theta``.

..  automethod:: concreteproperties.concrete_section.ConcreteSection.moment_curvature_analysis
  :noindex:

This analysis uses the ``stress_strain_profile`` given to the
:class:`~concreteproperties.material.Concrete` and
:class:`~concreteproperties.material.Steel` material properties to calculate a moment
curvature response. The analysis is displacement controlled with an adaptive curvature
increment controlled by the parameters ``kappa_inc``, ``kappa_mult``, ``kappa_inc_max``,
``delta_m_min`` and ``delta_m_max``.

.. seealso::
  For an application of the above, see the example
  :ref:`/examples/moment_curvature.ipynb`.


Ultimate Bending Capacity
-------------------------

The ultimate bending capacity of the reinforced concrete cross-section can be calculated
by calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.ultimate_bending_capacity`
method. By default the ultimate bending capacity is calculated for bending about the
``x`` axis with zero axial force, but this can be modified by providing a bending axis
angle ``theta`` and axial force ``n``.

..  automethod:: concreteproperties.concrete_section.ConcreteSection.ultimate_bending_capacity
  :noindex:

This analysis uses the ``ultimate_stress_strain_profile`` given to the
:class:`~concreteproperties.material.Concrete` materials and the
``stress_strain_profile`` given to the :class:`~concreteproperties.material.Steel`
materials. The ultimate strain profile within the cross-section is determined by setting
the strain at the extreme compressive fibre to the ``ultimate_strain`` parameter (see
:ref:`label-conc-ult-profile`) and finding the neutral axis that satisfies the
equilibrium of axial forces.

.. seealso::
  For an application of the above, see the example
  :ref:`/examples/ultimate_bending.ipynb`.


Moment Interaction Diagram
--------------------------

A moment interaction diagram can be generated for the reinforced concrete cross-section
by calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.moment_interaction_diagram`
method. By default the moment interaction diagram is generated for bending about the
``x`` axis, but this can be modified by providing a bending axis angle ``theta``.

The moment interaction diagram is generated by shifting the neutral axis throughout the
cross-section between the ``limits`` using either ``n_points`` or ``n_spacing``.
Additional ``control_points`` can be added to the analysis.

..  automethod:: concreteproperties.concrete_section.ConcreteSection.moment_interaction_diagram
  :noindex:

.. seealso::
  For an application of the above, see the example
  :ref:`/examples/moment_interaction.ipynb`.


Biaxial Bending Diagram
-----------------------

A biaxial bending diagram can be generated for the reinforced concrete cross-section,
by calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.biaxial_bending_diagram`
method. By default the biaxial bending diagram is generated for pure bending, but this
can be modified by providing an axial force ``n``.

The biaxial bending diagram is generated by rotating the bending axis angle through its
permissable range :math:`-\pi \leq \theta \leq \pi` and calculating the resultant
ultimate bending moments about the ``x`` and ``y`` axes.

..  automethod:: concreteproperties.concrete_section.ConcreteSection.biaxial_bending_diagram
  :noindex:

.. seealso::
  For an application of the above, see the example
  :ref:`/examples/biaxial_bending.ipynb`.


Stress Analysis
---------------

``concreteproperties`` allows you to perform four different kinds of stress analysis. Each
is detailed separately below.

.. seealso::
  For an application of stress analysis, see the example
  :ref:`/examples/stress_analysis.ipynb`.


Uncracked Stress
^^^^^^^^^^^^^^^^

A stress analysis can be performed on the gross reinforced concrete cross-section by
calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_uncracked_stress`
method.

..  automethod:: concreteproperties.concrete_section.ConcreteSection.calculate_uncracked_stress
  :noindex:

.. note::
  Forces/moments are assumed to be acting at the geometric centroid, i.e. ``cx`` and
  ``cy`` in
  :meth:`~concreteproperties.concrete_section.ConcreteSection.get_gross_properties`


Cracked Stress
^^^^^^^^^^^^^^

A stress analysis can be performed on the cracked reinforced concrete cross-section by
calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_cracked_stress`
method. Prior to calling this method, the cracked properties must be calculated using
the :meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_cracked_properties`
method and these results passed to
:meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_cracked_stress`.


..  automethod:: concreteproperties.concrete_section.ConcreteSection.calculate_cracked_stress
  :noindex:

.. note::
  Forces/moments are assumed to be acting at the geometric centroid, i.e. ``cx`` and
  ``cy`` in
  :meth:`~concreteproperties.concrete_section.ConcreteSection.get_gross_properties`


Service Stress
^^^^^^^^^^^^^^

A service stress analysis can be performed on the reinforced concrete cross-section by
calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_service_stress`
method. Prior to calling this method, a moment curvature analysis must be performed by
calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.moment_curvature_analysis`
method and these results passed to
:meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_service_stress`.


..  automethod:: concreteproperties.concrete_section.ConcreteSection.calculate_service_stress
  :noindex:

.. note::
  Forces/moments are assumed to be acting at the gross centroid, i.e. ``cx_gross`` and
  ``cy_gross`` in
  :meth:`~concreteproperties.concrete_section.ConcreteSection.get_gross_properties`


Ultimate Stress
^^^^^^^^^^^^^^^

An ultimate stress analysis can be performed on the reinforced concrete cross-section by
calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_ultimate_stress`
method. Prior to calling this method, the ultimate bending capacity must be calculated
by calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.ultimate_bending_capacity`
method and these results passed to
:meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_ultimate_stress`.


..  automethod:: concreteproperties.concrete_section.ConcreteSection.calculate_ultimate_stress
  :noindex:

.. note::
  Forces/moments are assumed to be acting at the gross centroid, i.e. ``cx_gross`` and
  ``cy_gross`` in
  :meth:`~concreteproperties.concrete_section.ConcreteSection.get_gross_properties`
