Assumptions
===========

Below are a list of assumptions and conventions used by ``concreteproperties``:

General
-------

* Plane sections remain plane (Euler-Bernoulli) i.e. linear strain profile throughout
  the cross-section
* For a given depth, the strain in the reinforcement is equal to the strain in the
  concrete, i.e. there is a perfect bond between concrete and reinforcement
* Materials that have ``meshed=False`` (e.g.
  :class:`~concreteproperties.material.SteelBar`) are characterised by a constant strain
  at it's geometric centroid
* Stresses in materials are determined from strains by linear interpolation from the
  corresponding stress-strain profile (linear extrapolation is used where the strains
  are outside the range of the profile)
* All moments in service and ultimate analyses (moment-curvature, ultimate bending,
  moment interaction, biaxial bending, service stress, ultimate stress) are calculated
  about, or assumed to be about the gross centroid (no material properties applied),
  unless otherwise specified in the intitialisation of
  :class:`~concreteproperties.concrete_section.ConcreteSection`
* Moment inputs for elastic stress analysis (cracked and uncracked), are assumed to be
  taken to be about the geometric centroid (material properties applied)
* Finite element assumptions and background can be found
  `here <https://sectionproperties.readthedocs.io/en/stable/user_guide/theory.html>`_.

Cracked Analysis
----------------

* Concrete is assumed to have zero tensile strength

Ultimate Analysis
-----------------

* The maximum strain at the extreme concrete compression fibre is assumed to reach a
  maximum limiting compressive strain, see note in :ref:`label-conc-ult-profile`
* The tensile strength of the concrete is assumed to be zero
* There are no geometric instabilities (e.g. buckling) prior to the cross-section
  reaching its ultimate strength
* The stress-strain relationship for concrete may be assumed to be rectangular,
  trapezoidal, bilinear, parabolic, or any other arbitrary shape that results in the
  prediction of strength in substantial agreement with the results of comprehensive
  tests

Conventions
-----------

* Compressive stresses are positive, tensile stresses area negative
* Angles are measured in radians and are positive counter clockwise from the positive
  ``x`` axis

Prestressed Analysis
--------------------

* All prestressing strands are fully bonded to the concrete
* Internal actions arising from prestressing forces are automatically included in all
  analyses
