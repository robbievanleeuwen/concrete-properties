import numpy as np
import matplotlib.pyplot as plt
from sectionproperties.pre.library.concrete_sections import concrete_rectangular_section
from concreteproperties.material import Concrete, Steel
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.stress_strain_profile import (
    StressStrainProfile,
    BilinearProfile,
    WhitneyStressBlock,
)

# concrete_profile = StressStrainProfile(
#     strains=[-40 / 32.8e3, 0, 40 / 32.8e3],
#     stresses=[0, 0, 40],
# )
concrete_profile = StressStrainProfile(
    strains=[-41 / 32.8e3, -40 / 32.8e3, -3.8 / 32.8e3, 0, 40 / 32.8e3],
    stresses=[0, 0, -3.8, 0, 40],
)
# concrete_profile.plot_stress_strain(title="Concrete Stress-Strain Profile")

steel_profile = BilinearProfile(
    strain1=500 / 200e3,
    strain2=0.025,
    stress1=500,
    stress2=595,
)
# steel_profile.plot_stress_strain(title="Steel Stress-Strain Profile")

concrete = Concrete(
    name="40 MPa Concrete",
    density=2.4e-6,
    stress_strain_profile=concrete_profile,
    ultimate_stress_strain_profile=WhitneyStressBlock(
        alpha_2=0.85,
        gamma=0.77,
        compressive_strength=40,
        ultimate_strain=0.003,
    ),
    alpha_1=0.85,
    flexural_tensile_strength=3.8,
    residual_shrinkage_stress=0,
    colour="lightgrey",
)

steel = Steel(
    name="500 MPa Steel",
    density=7.85e-6,
    yield_strength=500,
    stress_strain_profile=steel_profile,
    colour="grey",
)


geom = concrete_rectangular_section(
    b=300,
    d=600,
    dia_top=20,
    n_top=3,
    dia_bot=20,
    n_bot=3,
    n_circle=4,
    cover=30,
    area_top=310,
    area_bot=310,
    conc_mat=concrete,
    steel_mat=steel,
)

conc_sec = ConcreteSection(geom)
conc_sec.plot_section()

mcr = conc_sec.moment_curvature_diagram()
mcr.plot_results()

moments = [50e6, 100e6, 150e6, 200e6, 250e6, 275e6]

for m in moments:
    stress_res = conc_sec.calculate_service_stress(moment_curvature_results=mcr, m=m)
    stress_res.plot_stress()
