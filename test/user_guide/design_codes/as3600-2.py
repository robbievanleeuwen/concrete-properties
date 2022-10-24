from concreteproperties.design_codes.as3600 import AS3600

design_code = AS3600()
steel = design_code.create_steel_material()

steel.stress_strain_profile.plot_stress_strain(
  title=f"{steel.name} - Stress-Strain Profile"
)