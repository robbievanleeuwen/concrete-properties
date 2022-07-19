from concreteproperties.stress_strain_profile import RectangularStressBlock

RectangularStressBlock(
    compressive_strength=40,
    alpha=0.85,
    gamma=0.77,
    ultimate_strain=0.003,
).plot_stress_strain()