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
    poissons_ratio=0.3,
    compressive_strength=concrete_profile.compressive_strength,
    alpha_1=0.85,
    density=2.4e-6,
    color="lightgrey",
    stress_strain_profile=concrete_profile,
)

steel = Steel(
    name="500 MPa Steel",
    elastic_modulus=steel_profile.elastic_modulus,
    poissons_ratio=0.2,
    yield_strength=steel_profile.yield_strength,
    density=7.85e-6,
    color="grey",
    stress_strain_profile=steel_profile,
)

n_results = []
m_results = []
labels = []

for idx in range(4):
    geometry = concrete_rectangular_section(
        b=400,
        d=600,
        dia_top=16,
        n_top=6,
        dia_bot=16,
        n_bot=6,
        n_circle=4,
        cover=66,
        area_top=200 * (idx + 1),
        area_bot=200 * (idx + 1),
        conc_mat=concrete,
        steel_mat=steel,
    )

    geometry.create_mesh(mesh_sizes=[0])
    section = Section(geometry)

    conc_sec = ConcreteSection(section)
    n, m = conc_sec.moment_interaction_diagram(theta=0, plot=False)
    n_results.append(n)
    m_results.append(m)
    labels.append("p = {0}".format(0.01 * (idx + 1)))

conc_sec.plot_moment_interaction_diagram(n_i=n_results, m_i=m_results, labels=labels)
