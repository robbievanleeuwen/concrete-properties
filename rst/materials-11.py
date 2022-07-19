from concreteproperties.stress_strain_profile import SteelElasticPlastic

SteelElasticPlastic(
  yield_strength=500,
  elastic_modulus=200e3,
  fracture_strain=0.05,
).plot_stress_strain()