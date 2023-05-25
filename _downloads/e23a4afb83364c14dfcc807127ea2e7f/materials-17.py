from concreteproperties.stress_strain_profile import StrandPCI1992

StrandPCI1992(
  yield_strength=1500,
  elastic_modulus=195e3,
  fracture_strain=0.035,
  breaking_strength=1830,
).plot_stress_strain()