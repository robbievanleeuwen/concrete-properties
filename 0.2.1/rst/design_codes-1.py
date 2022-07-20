from concreteproperties.design_codes import AS3600

design_code = AS3600()
concrete = design_code.create_concrete_material(compressive_strength=40)

concrete.stress_strain_profile.plot_stress_strain(
  title=f"{concrete.name} - Service Profile"
)
concrete.ultimate_stress_strain_profile.plot_stress_strain(
  title=f"{concrete.name} - Ultimate Profile"
)