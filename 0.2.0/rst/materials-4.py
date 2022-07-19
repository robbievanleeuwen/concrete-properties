from concreteproperties.stress_strain_profile import ConcreteLinearNoTension

ConcreteLinearNoTension(
  elastic_modulus=35e3,
  ultimate_strain=0.003,
  compressive_strength=40,
).plot_stress_strain()