from concreteproperties.stress_strain_profile import SteelProfile

SteelProfile(
  strains=[-0.05, -0.03, -0.02, -500 / 200e3, 0, 500 / 200e3, 0.02, 0.03, 0.05],
  stresses=[-600, -600, -500, -500, 0, 500, 500, 600, 600],
  yield_strength=500,
  elastic_modulus=200e3,
  fracture_strain=0.05,
).plot_stress_strain()