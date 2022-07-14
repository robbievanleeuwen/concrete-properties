from concreteproperties.stress_strain_profile import EurocodeNonLinear

EurocodeNonLinear(
    elastic_modulus=35e3,
    ultimate_strain=0.0035,
    compressive_strength=40,
    compressive_strain=0.0023,
    tensile_strength=3.5,
    tension_softening_stiffness=7e3,
).plot_stress_strain()