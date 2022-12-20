from concreteproperties.stress_strain_profile import StrandProfile

StrandProfile(
  strains=[-0.03, -0.01, -1400 / 195e3, 0, 1400 / 195e3, 0.01, 0.03],
  stresses=[-1800, -1600, -1400, 0, 1400, 1600, 1800],
  yield_strength=500,
).plot_stress_strain()