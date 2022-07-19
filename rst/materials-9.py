from concreteproperties.stress_strain_profile import EurocodeParabolicUltimate

EurocodeParabolicUltimate(
    compressive_strength=40,
    compressive_strain=0.00175,
    ultimate_strain=0.0035,
    n=2,
).plot_stress_strain()