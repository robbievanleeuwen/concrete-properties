import numpy as np
from sectionproperties.pre.library.concrete_sections import concrete_rectangular_section
from concreteproperties.material import Concrete, Steel
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.stress_strain_profile import (
    LinearProfile,
    WhitneyStressBlock,
    SteelElasticPlastic,
)


concrete = Concrete(
    name="40 MPa Concrete",
    density=2.4e-6,
    stress_strain_profile=LinearProfile(elastic_modulus=32.8e3),
    ultimate_stress_strain_profile=WhitneyStressBlock(
        alpha_2=0.85,
        gamma=0.77,
        compressive_strength=40,
        ultimate_strain=0.003,
    ),
    alpha_1=0.85,
    flexural_tensile_strength=0.6 * np.sqrt(40),
    residual_shrinkage_stress=0,
    colour="lightgrey",
)

steel = Steel(
    name="500 MPa Steel",
    density=7.85e-6,
    yield_strength=500,
    stress_strain_profile=SteelElasticPlastic(
        yield_strength=500,
        elastic_modulus=200e3,
        fracture_strain=0.05,
    ),
    colour="grey",
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

    conc_sec = ConcreteSection(geometry)
    n, m = conc_sec.moment_interaction_diagram(theta=0, plot=False)
    n_results.append(n)
    m_results.append(m)
    labels.append("p = {0}".format(0.01 * (idx + 1)))

conc_sec.plot_moment_interaction_diagram(n_i=n_results, m_i=m_results, labels=labels)
