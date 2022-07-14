from concreteproperties.stress_strain_profile import ConcreteServiceProfile

ConcreteServiceProfile(
  strains=[-5 / 35e3, -4 / 35e3, -3 / 35e3, 0, 40 / 35e3, 0.003],
  stresses=[0, 0, -3, 0, 40, 40],
  ultimate_strain=0.003,
).plot_stress_strain()