import numpy as np
from sectionproperties.pre.library.concrete_sections import concrete_rectangular_section
from sectionproperties.analysis.section import Section
import concreteproperties.stress_strain_profile as cp_profile
from concreteproperties.material import Concrete, Steel
from concreteproperties.concrete_section import ConcreteSection

concrete_profile = cp_profile.WhitneyStressBlock(
    alpha_2=0.85,
    gamma=0.77,
    compressive_strength=40,
    ultimate_strain=0.003,
)

steel_profile = cp_profile.SteelElasticPlastic(
    yield_strength=500,
    elastic_modulus=200e3,
    fracture_strain=0.05,
)

concrete = Concrete(
    name="40 MPa Concrete",
    elastic_modulus=32.8e3,
    compressive_strength=40,
    alpha_1=0.85,
    density=2.4e-6,
    stress_strain_profile=concrete_profile,
)

steel = Steel(
    name="500 MPa Steel",
    elastic_modulus=200e3,
    yield_strength=500,
    density=7.85e-6,
    stress_strain_profile=steel_profile,
)

geometry = concrete_rectangular_section(
    b=300,
    d=600,
    dia=20,
    n_bar=3,
    n_circle=16,
    cover=30,
    area=310,
    conc_mat=concrete,
    steel_mat=steel,
)
geometry.create_mesh(mesh_sizes=[500])
section = Section(geometry)

conc_sec = ConcreteSection(section)
# print(conc_sec.calculate_section_actions(d_n=600, theta=0))
print(conc_sec.ultimate_bending_capacity(theta=0, n=4500e3))
