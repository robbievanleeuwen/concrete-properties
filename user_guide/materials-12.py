from concreteproperties.stress_strain_profile import SteelHardening

SteelHardening(
  yield_strength=500,
  elastic_modulus=200e3,
  fracture_strain=0.05,
  ultimate_strength=600,
).plot_stress_strain()