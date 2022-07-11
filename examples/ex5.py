import numpy as np
from sectionproperties.pre.library.concrete_sections import concrete_rectangular_section
from concreteproperties.material import Concrete, Steel
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.stress_strain_profile import (
    ConcreteLinear,
    RectangularStressBlock,
    SteelElasticPlastic,
)
from concreteproperties.results import BiaxialBendingResults


concrete = Concrete(
    name="40 MPa Concrete",
    density=2.4e-6,
    stress_strain_profile=ConcreteLinear(elastic_modulus=32.8e3),
    ultimate_stress_strain_profile=RectangularStressBlock(
        compressive_strength=40,
        alpha=0.85,
        gamma=0.77,
        ultimate_strain=0.003,
    ),
    alpha_squash=0.85,
    flexural_tensile_strength=0.6 * np.sqrt(40),
    colour="lightgrey",
)

steel = Steel(
    name="500 MPa Steel",
    density=7.85e-6,
    stress_strain_profile=SteelElasticPlastic(
        yield_strength=500,
        elastic_modulus=200e3,
        fracture_strain=0.05,
    ),
    colour="grey",
)

geometry = concrete_rectangular_section(
    b=400,
    d=600,
    dia_top=16,
    n_top=3,
    dia_bot=16,
    n_bot=3,
    n_circle=4,
    cover=30,
    area_top=200,
    area_bot=200,
    conc_mat=concrete,
    steel_mat=steel,
)

conc_sec = ConcreteSection(geometry)
conc_sec.moment_interaction_diagram().plot_diagram()

n_list = np.linspace(0, 6585e3, 11)
biaxial_results = []

for n in n_list:
    biaxial_results.append(conc_sec.biaxial_bending_diagram(n=n))

BiaxialBendingResults.plot_multiple_diagrams(biaxial_results)
