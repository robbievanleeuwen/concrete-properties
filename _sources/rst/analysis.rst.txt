Analysis
========

An analysis in *concreteproperties* begins with creating a
:class:`~concreteproperties.concrete_section.ConcreteSection` object from a
:class:`~sectionproperties.pre.geometry.CompoundGeometry` object with assigned
materials. The following code creates a 600 mm deep x 300 mm wide concrete beam with
3N20 bars top and bottom. 40 MPa concrete and 500 MPa steel are used with material
properties defined in accordance with AS 3600:2018.

.. plot::
  :include-source: True
  :caption: 600 mm deep x 300 mm wide concrete beam

  from concreteproperties.material import Concrete, Steel
  from concreteproperties.stress_strain_profile import (
      ConcreteLinear,
      RectangularStressBlock,
      SteelElasticPlastic,
  )
  from sectionproperties.pre.library.primitive_sections import rectangular_section
  from concreteproperties.pre import add_bar_rectangular_array
  from concreteproperties.concrete_section import ConcreteSection


  concrete = Concrete(
      name="40 MPa Concrete",
      density=2.4e-6,
      stress_strain_profile=ConcreteLinear(elastic_modulus=32.8e3),
      ultimate_stress_strain_profile=RectangularStressBlock(
          compressive_strength=40,
          alpha=0.79,
          gamma=0.87,
          ultimate_strain=0.003,
      ),
      alpha_squash=0.85,
      flexural_tensile_strength=3.8,
      colour="lightgrey",
  )

  steel = Steel(
      name="500 MPa Steel",
      density=7.85e-6,
      stress_strain_profile=SteelElasticPlastic(
          yield_strength=500,
          elastic_modulus=200e3,
          fracture_strain=0.05,
      ),
      colour="grey",
  )

  beam = rectangular_section(d=600, b=300, material=concrete)
  geom = add_bar_rectangular_array(
    geometry=beam, area=310, material=steel, n_x=3, x_s=110, n_y=2, y_s=520,
    anchor=(40, 40)
  )
  geom.plot_geometry()
