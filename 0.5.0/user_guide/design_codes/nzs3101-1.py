from concreteproperties.design_codes.nzs3101 import NZS3101

design_code = NZS3101()
concrete = design_code.create_concrete_material(compressive_strength=40)

concrete.stress_strain_profile.plot_stress_strain(
  title=f"{concrete.name} - Service Profile"
)
concrete.ultimate_stress_strain_profile.plot_stress_strain(
  title=f"{concrete.name} - Ultimate Profile"
)