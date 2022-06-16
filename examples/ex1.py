import numpy as np
from sectionproperties.pre.library.concrete_sections import concrete_circular_section
from concreteproperties.material import Concrete, Steel
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.stress_strain_profile import (
    WhitneyStressBlock,
    SteelElasticPlastic,
)

concrete_compressive_strength = 40
steel_yield_strength = 500
steel_elastic_modulus = 200e3

concrete = Concrete(
    name="40 MPa Concrete",
    elastic_modulus=32.8e3,
    poissons_ratio=0.2,
    compressive_strength=concrete_compressive_strength,
    alpha_1=0.85,
    density=2.4e-6,
    color="lightgrey",
    stress_strain_profile=WhitneyStressBlock(
        alpha_2=0.85,
        gamma=0.77,
        compressive_strength=concrete_compressive_strength,
        ultimate_strain=0.003,
    ),
)

steel = Steel(
    name="500 MPa Steel",
    elastic_modulus=steel_elastic_modulus,
    poissons_ratio=0.3,
    yield_strength=steel_yield_strength,
    density=7.85e-6,
    color="grey",
    stress_strain_profile=SteelElasticPlastic(
        yield_strength=steel_yield_strength,
        elastic_modulus=steel_elastic_modulus,
        fracture_strain=0.05,
    ),
)

geometry = concrete_circular_section(
    d=600,
    n=32,
    dia=20,
    n_bar=6,
    n_circle=4,
    area_conc=np.pi * 600 * 600 / 4,
    area_bar=310,
    cover=45,
    conc_mat=concrete,
    steel_mat=steel,
)

conc_sec = ConcreteSection(geometry)

print(conc_sec.calculate_section_actions(d_n=100, theta=0))
print(conc_sec.ultimate_bending_capacity(theta=0, n=0))
conc_sec.moment_interaction_diagram(theta=0)
