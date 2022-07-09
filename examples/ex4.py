import numpy as np
import matplotlib.pyplot as plt
from sectionproperties.pre.library.concrete_sections import concrete_rectangular_section
from concreteproperties.material import Concrete, Steel
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.stress_strain_profile import (
    RectangularStressBlock,
    SteelHardening,
    EurocodeNonLinear,
)

concrete_profile = EurocodeNonLinear(35e3, 0.0035, 40, 0.0023, 3.5, 7e3)
concrete_profile.plot_stress_strain(title="Concrete Stress-Strain Profile")

steel_profile = SteelHardening(
    yield_strength=500,
    elastic_modulus=200e3,
    fracture_strain=0.025,
    ultimate_strength=595,
)
steel_profile.plot_stress_strain(title="Steel Stress-Strain Profile")

concrete = Concrete(
    name="40 MPa Concrete",
    density=2.4e-6,
    stress_strain_profile=concrete_profile,
    ultimate_stress_strain_profile=RectangularStressBlock(
        compressive_strength=40,
        alpha=0.85,
        gamma=0.77,
        ultimate_strain=0.003,
    ),
    alpha_squash=0.85,
    flexural_tensile_strength=3.8,
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
    dia_top=16,
    n_top=3,
    dia_bot=20,
    n_bot=3,
    n_circle=4,
    cover=30,
    area_top=200,
    area_bot=310,
    conc_mat=concrete,
    steel_mat=steel,
)

conc_sec = ConcreteSection(geom)
conc_sec.plot_section()

ultimate_res = conc_sec.ultimate_bending_capacity()
ultimate_res.print_results()
mc = conc_sec.moment_curvature_analysis()
mc.plot_results()

moments = [50e6, 114e6, 120e6, 200e6, 230e6, 250e6, 294e6]

for m in moments:
    stress_res = conc_sec.calculate_service_stress(moment_curvature_results=mc, m=m)
    stress_res.plot_stress(title=f"Moment = {m / 1e6:.0f} kN.m")
