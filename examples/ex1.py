import numpy as np
from sectionproperties.pre.library.concrete_sections import concrete_circular_section
from concreteproperties.material import Concrete, Steel
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.stress_strain_profile import (
    ConcreteLinear,
    RectangularStressBlock,
    SteelElasticPlastic,
)

from rich.pretty import pprint


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
conc_sec.plot_section()

# CRACKED PROPERTIES
cracked_res = conc_sec.calculate_cracked_properties()
cracked_res.print_results()

# ULTIMATE CAPACITY
ultimate_res = conc_sec.ultimate_bending_capacity()
ultimate_res.print_results()
conc_sec.moment_interaction_diagram().plot_diagram()
conc_sec.biaxial_bending_diagram(n=4000e3).plot_diagram()

# PLOT STRESSES
uncr_stress_res = conc_sec.calculate_uncracked_stress(mx=50e6)
pprint(uncr_stress_res)
uncr_stress_res.plot_stress(title="Uncracked Stress")
cracked_stress_res = conc_sec.calculate_cracked_stress(
    cracked_results=cracked_res, m=100e6
)
pprint(cracked_stress_res)
cracked_stress_res.plot_stress(title="Cracked Stress")
ult_stress_res = conc_sec.calculate_ultimate_stress(ultimate_results=ultimate_res)
pprint(ult_stress_res)
ult_stress_res.plot_stress(title="Ultimate Stress")
