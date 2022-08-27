from concreteproperties.stress_strain_profile import ConcreteUltimateProfile

ConcreteUltimateProfile(
  strains=[-20 / 30e3, 0, 20 / 30e3, 30 / 25e3, 40 / 20e3, 0.003],
  stresses=[0, 0, 20, 30, 40, 40],
  compressive_strength=32,
).plot_stress_strain()