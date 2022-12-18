from concreteproperties.design_codes.nzs3101 import NZS3101

design_code = NZS3101()
steel = design_code.create_steel_material(steel_grade="500E")

steel.stress_strain_profile.plot_stress_strain(
  title=f"{steel.name} - Stress-Strain Profile"
)