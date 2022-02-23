import numpy as np
from sectionproperties.pre.library.concrete_sections import concrete_tee_section
from sectionproperties.analysis.section import Section
import concreteproperties.stress_strain_profile as cp_profile
from concreteproperties.material import Concrete, Steel
from concreteproperties.concrete_section import ConcreteSection

concrete_profile = cp_profile.WhitneyStressBlock(
    alpha=0.85,
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

geometry = concrete_tee_section(
    b=450,
    d=900,
    b_f=1200,
    d_f=250,
    dia=24,
    n_bar=2,
    n_circle=16,
    cover=30,
    conc_mat=concrete,
    steel_mat=steel,
)
geometry.create_mesh(mesh_sizes=[500])
section = Section(geometry)

conc_sec = ConcreteSection(section)
conc_sec.calculate_section_actions(d_n=315, theta=-np.pi/4 - np.pi/2)
# conc_sec.calculate_section_actions(d_n=500, theta=0)
