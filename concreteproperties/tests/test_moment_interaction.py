import numpy as np
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.material import Concrete, SteelBar
from concreteproperties.stress_strain_profile import (
    ConcreteLinear,
    RectangularStressBlock,
    SteelElasticPlastic,
)
from sectionproperties.pre.library.concrete_sections import concrete_rectangular_section

concrete = Concrete(
    name="32 MPa Concrete",
    density=2.4e-6,
    stress_strain_profile=ConcreteLinear(elastic_modulus=30.1e3),
    ultimate_stress_strain_profile=RectangularStressBlock(
        compressive_strength=32,
        alpha=0.85,
        gamma=0.83,
        ultimate_strain=0.003,
    ),
    flexural_tensile_strength=1.0,
    colour="lightgrey",
)

steel = SteelBar(
    name="500 MPa Steel",
    density=7.85e-6,
    stress_strain_profile=SteelElasticPlastic(
        yield_strength=500,
        elastic_modulus=200e3,
        fracture_strain=0.05,
    ),
    colour="grey",
)

D = 450

geometry = concrete_rectangular_section(
    b=300,
    d=D,
    dia_top=24,
    n_top=3,
    dia_bot=24,
    n_bot=3,
    n_circle=4,
    cover=30,
    area_top=450,
    area_bot=450,
    conc_mat=concrete,  # type: ignore
    steel_mat=steel,  # type: ignore
)

conc_sec = ConcreteSection(geometry)


def test_control_points():
    control_points = [
        ("D", 1.0),
        ("D", 0.8),
        ("d_n", 310),
        ("d_n", 200),
        ("N", 1000.0),
        ("N", 0.0),
    ]

    n = 10

    mi_res = conc_sec.moment_interaction_diagram(
        control_points=control_points, n_points=n
    )

    d_n_list = []
    n_list = []

    for ult_res in mi_res.results:
        d_n_list.append(ult_res.d_n)
        n_list.append(ult_res.n)

    d_n_list = np.array(d_n_list).astype(int)
    n_list = np.array(n_list).astype(int)

    # check neutral axes in list
    assert int(D * 1) in d_n_list
    assert int(D * 0.8) in d_n_list
    assert 310 in d_n_list
    assert 200 in d_n_list
    assert int(conc_sec.ultimate_bending_capacity(n=1000).n) in n_list
    assert int(conc_sec.ultimate_bending_capacity(n=0).n) in n_list

    # check length of list
    assert len(d_n_list) == 5 * (n - 1) + 1
