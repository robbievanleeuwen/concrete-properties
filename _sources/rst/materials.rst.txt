Materials
=========

*concreteproperties* requires material properties to be defined for the concrete and
steel components of the reinforced concrete section. Any number of different material
properties can be used for a single cross-section. For example, higher strength precast
sections can be topped with lower grade in-situ slabs, and high tensile steel can be
used in combination with normal grade reinforcing steel.

The structural behaviour of both concrete and steel materials are described by
:ref:`stress-strain-profiles`.


Concrete
--------

..  autoclass:: concreteproperties.material.Concrete
  :noindex:


Steel
-----

..  autoclass:: concreteproperties.material.Steel
  :noindex:


.. _stress-strain-profiles:

Stress-Strain Profiles
----------------------

*concreteproperties* uses stress-strain profiles to define material behaviour for both
service and ultimate analyses. A :class:`~concreteproperties.material.Concrete` object
requires both a service stress-strain profile (calculation of area properties,
moment-curvature analysis, elastic and service stress analysis) and an ultimate
stress-strain profile (ultimate bending capacity, moment interaction diagram, biaxial
bending diagram, ultimate stress analysis). A
:class:`~concreteproperties.material.Steel` object only requires one stress-strain
profile which is used for both service and ultimate analyses.

.. note::

   Stress values are interpolated from stresses and strains supplied to the profile. If
   the strain is outside of the range of the stress-strain profile, the stress is
   extrapolated based off the closest exterior two points of the stress-strain profile.

..  autoclass:: concreteproperties.stress_strain_profile.StressStrainProfile
  :noindex:
  :members: print_properties, plot_stress_strain


Concrete Service Stress-Strain Profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Generic Concrete Service Profile
""""""""""""""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.ConcreteServiceProfile
  :noindex:

.. plot::
  :include-source: True
  :caption: ConcreteServiceProfile Stress-Strain Profile

  from concreteproperties.stress_strain_profile import ConcreteServiceProfile

  ConcreteServiceProfile(
    strains=[-5 / 35e3, -4 / 35e3, -3 / 35e3, 0, 40 / 35e3, 0.003],
    stresses=[0, 0, -3, 0, 40, 40],
    ultimate_strain=0.003,
  ).plot_stress_strain()


Linear Concrete Service Profile
"""""""""""""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.ConcreteLinear
  :noindex:

.. plot::
  :include-source: True
  :caption: ConcreteLinear Stress-Strain Profile

  from concreteproperties.stress_strain_profile import ConcreteLinear

  ConcreteLinear(elastic_modulus=35e3).plot_stress_strain()


Linear Concrete (No Tension) Service Profile
""""""""""""""""""""""""""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.ConcreteLinearNoTension
  :noindex:

.. plot::
  :include-source: True
  :caption: ConcreteLinearNoTension Stress-Strain Profile

  from concreteproperties.stress_strain_profile import ConcreteLinearNoTension

  ConcreteLinearNoTension(elastic_modulus=35e3).plot_stress_strain()


Eurocode Non-Linear Concrete Service Profile
""""""""""""""""""""""""""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.EurocodeNonLinear
  :noindex:

.. plot::
  :include-source: True
  :caption: EurocodeNonLinear Stress-Strain Profile

  from concreteproperties.stress_strain_profile import EurocodeNonLinear

  EurocodeNonLinear(
      elastic_modulus=35e3,
      ultimate_strain=0.0035,
      compressive_strength=40,
      compressive_strain=0.0023,
      tensile_strength=3.5,
      tension_softening_stiffness=7e3,
  ).plot_stress_strain()


Concrete Ultimate Stress-Strain Profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Generic Concrete Ultimate Profile
"""""""""""""""""""""""""""""""""

xxxx
