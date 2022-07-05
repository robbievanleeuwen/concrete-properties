import numpy as np
from sectionproperties.pre.library.concrete_sections import concrete_rectangular_section
from concreteproperties.material import Concrete, Steel
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.stress_strain_profile import (
    LinearProfile,
    WhitneyStressBlock,
    SteelElasticPlastic,
)

from rich.pretty import pprint


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

geometry = concrete_rectangular_section(
    b=400,
    d=600,
    dia_top=20,
    n_top=3,
    dia_bot=24,
    n_bot=3,
    n_circle=4,
    cover=30,
    area_top=310,
    area_bot=450,
    conc_mat=concrete,
    steel_mat=steel,
)

conc_sec = ConcreteSection(geometry)
conc_sec.plot_section()

# CALCULATE GROSS PROPERTIES
conc_sec.gross_properties.print_results()
conc_sec.get_transformed_gross_properties(elastic_modulus=32.8e3).print_results()

# CALCULATE CRACKED PROPERTIES
cracked_res = conc_sec.calculate_cracked_properties()
cracked_res.calculate_transformed_properties(elastic_modulus=32.8e3)
cracked_res.print_results()

# CALCULATE MOMENT INTERACTION DIAGRAM
mi_res = conc_sec.moment_interaction_diagram(m_neg=True)
mi_res.plot_diagram()

# CALCULATE UNCRACKED AND CRACKED STRESSES
uncr_stress_res = conc_sec.calculate_uncracked_stress(mx=80e6, my=0e6)
pprint(uncr_stress_res)
uncr_stress_res.plot_stress(title="Uncracked Stress", pause=False)

cr_stress_res = conc_sec.calculate_cracked_stress(cracked_results=cracked_res, m=100e6)
pprint(cr_stress_res)
cr_stress_res.plot_stress(title="Cracked Stress")
