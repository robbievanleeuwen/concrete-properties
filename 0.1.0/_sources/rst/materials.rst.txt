Materials
=========

*concreteproperties* requires material properties to be defined for the concrete and
steel components of the reinforced concrete section. Any number of different material
properties can be used for a single cross-section. For example, higher strength precast
sections can be topped with lower grade in-situ slabs, and high tensile steel can be
used in combination with normal grade reinforcing steel.

The structural behaviour of both concrete and steel materials are described by
:ref:`stress-strain-profiles`.

.. note::

  In *concreteproperties*, a positive sign is given to compressive forces, stresses and
  strains, while a negative sign is given to tensile forces, stresses and strains.

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
requires both a **service** stress-strain profile (calculation of area properties,
moment-curvature analysis, elastic and service stress analysis) and an **ultimate**
stress-strain profile (ultimate bending capacity, moment interaction diagram, biaxial
bending diagram, ultimate stress analysis). A
:class:`~concreteproperties.material.Steel` object only requires one stress-strain
profile which is used for both service and ultimate analyses.

.. note::

   Stress values are interpolated from stresses and strains supplied to the profile. If
   the strain is outside of the range of the stress-strain profile, the stress is
   extrapolated based off the closest two points of the stress-strain profile.

..  autoclass:: concreteproperties.stress_strain_profile.StressStrainProfile
  :noindex:
  :members: print_properties, plot_stress_strain


Concrete Service Stress-Strain Profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  Unless assigned in the class constructor, the ``elastic_modulus`` of the concrete is
  determined by the initial compressive slope of the stress-strain profile. This
  ``elastic_modulus`` is used in the calculation of area properties and elastic stress
  analysis.

Generic Concrete Service Profile
""""""""""""""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.ConcreteServiceProfile
  :noindex:
  :show-inheritance:

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
  :show-inheritance:

.. plot::
  :include-source: True
  :caption: ConcreteLinear Stress-Strain Profile

  from concreteproperties.stress_strain_profile import ConcreteLinear

  ConcreteLinear(elastic_modulus=35e3).plot_stress_strain()


Linear Concrete (No Tension) Service Profile
""""""""""""""""""""""""""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.ConcreteLinearNoTension
  :noindex:
  :show-inheritance:

.. plot::
  :include-source: True
  :caption: ConcreteLinearNoTension Stress-Strain Profile

  from concreteproperties.stress_strain_profile import ConcreteLinearNoTension

  ConcreteLinearNoTension(elastic_modulus=35e3).plot_stress_strain()


Eurocode Non-Linear Concrete Service Profile
""""""""""""""""""""""""""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.EurocodeNonLinear
  :noindex:
  :show-inheritance:

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


.. _label-conc-ult-profile:

Concrete Ultimate Stress-Strain Profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  Unless assigned in the class constructor, the ``ultimate_strain`` of the concrete is
  taken as the largest compressive strain in the stress-strain profile. This
  ``ultimate_strain`` defines the curvature and strain profile used in ultimate
  analyses.

.. warning::

  *concreteproperties* currently only supports a single unique ``ultimate_strain`` to be
  used for a given :class:`~concreteproperties.concrete_section.ConcreteSection`. While
  multiple concrete materials, with differing stress-strain profiles, can be
  used within a given :class:`~concreteproperties.concrete_section.ConcreteSection`, the
  ultimate analysis will use the smallest value of the ``ultimate_strain`` amongst the
  various concrete materials to define the strain profile at ultimate.


Generic Concrete Ultimate Profile
"""""""""""""""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.ConcreteUltimateProfile
  :noindex:
  :show-inheritance:

.. plot::
  :include-source: True
  :caption: ConcreteUltimateProfile Stress-Strain Profile

  from concreteproperties.stress_strain_profile import ConcreteUltimateProfile

  ConcreteUltimateProfile(
    strains=[-20 / 30e3, 0, 20 / 30e3, 30 / 25e3, 40 / 20e3, 0.003],
    stresses=[0, 0, 20, 30, 40, 40],
    compressive_strength=32,
  ).plot_stress_strain()


Rectangular Stress Block
""""""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.RectangularStressBlock
  :noindex:
  :show-inheritance:

.. plot::
  :include-source: True
  :caption: RectangularStressBlock Stress-Strain Profile

  from concreteproperties.stress_strain_profile import RectangularStressBlock

  RectangularStressBlock(
      compressive_strength=40,
      alpha=0.85,
      gamma=0.77,
      ultimate_strain=0.003,
  ).plot_stress_strain()


Bilinear Ultimate Profile
"""""""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.BilinearStressStrain
  :noindex:
  :show-inheritance:

.. plot::
  :include-source: True
  :caption: BilinearStressStrain Stress-Strain Profile

  from concreteproperties.stress_strain_profile import BilinearStressStrain

  BilinearStressStrain(
      compressive_strength=40,
      compressive_strain=0.00175,
      ultimate_strain=0.0035,
  ).plot_stress_strain()


Eurocode Parabolic Ultimate Profile
"""""""""""""""""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.EurocodeParabolicUltimate
  :noindex:
  :show-inheritance:

.. plot::
  :include-source: True
  :caption: EurocodeParabolicUltimate Stress-Strain Profile

  from concreteproperties.stress_strain_profile import EurocodeParabolicUltimate

  EurocodeParabolicUltimate(
      compressive_strength=40,
      compressive_strain=0.00175,
      ultimate_strain=0.0035,
      n=2,
  ).plot_stress_strain()



Steel Stress-Strain Profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Generic Steel Profile
"""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.SteelProfile
  :noindex:
  :show-inheritance:

.. plot::
  :include-source: True
  :caption: SteelProfile Stress-Strain Profile

  from concreteproperties.stress_strain_profile import SteelProfile

  SteelProfile(
    strains=[-0.05, -0.03, -0.02, -500 / 200e3, 0, 500 / 200e3, 0.02, 0.03, 0.05],
    stresses=[-600, -600, -500, -500, 0, 500, 500, 600, 600],
    yield_strength=500,
    elastic_modulus=200e3,
    fracture_strain=0.05,
  ).plot_stress_strain()


Elastic-Plastic Steel Profile
"""""""""""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.SteelElasticPlastic
  :noindex:
  :show-inheritance:

.. plot::
  :include-source: True
  :caption: SteelElasticPlastic Stress-Strain Profile

  from concreteproperties.stress_strain_profile import SteelElasticPlastic

  SteelElasticPlastic(
    yield_strength=500,
    elastic_modulus=200e3,
    fracture_strain=0.05,
  ).plot_stress_strain()


Elastic-Plastic Hardening Steel Profile
"""""""""""""""""""""""""""""""""""""""

..  autoclass:: concreteproperties.stress_strain_profile.SteelHardening
  :noindex:
  :show-inheritance:

.. plot::
  :include-source: True
  :caption: SteelHardening Stress-Strain Profile

  from concreteproperties.stress_strain_profile import SteelHardening

  SteelHardening(
    yield_strength=500,
    elastic_modulus=200e3,
    fracture_strain=0.05,
    ultimate_strength=600,
  ).plot_stress_strain()
