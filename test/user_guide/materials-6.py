from concreteproperties.stress_strain_profile import ModifiedMander

ModifiedMander(elastic_modulus=30e3,
      compressive_strength=30,
      tensile_strength=4.5,
      sect_type="rect",
      conc_tension=True,
      conc_spalling=True,
      n_points=25
).plot_stress_strain()