from concreteproperties.stress_strain_profile import BilinearStressStrain

BilinearStressStrain(
    compressive_strength=40,
    compressive_strain=0.00175,
    ultimate_strain=0.0035,
).plot_stress_strain()