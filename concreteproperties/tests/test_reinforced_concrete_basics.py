import numpy as np
import pytest
import sectionproperties.pre.library.primitive_sections as sp_ps
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.material import Concrete, SteelBar
from concreteproperties.pre import add_bar, add_bar_rectangular_array
from concreteproperties.stress_strain_profile import (
    ConcreteLinear,
    RectangularStressBlock,
    SteelElasticPlastic,
)
from sectionproperties.pre.library.concrete_sections import concrete_rectangular_section

# All examples come from:
# Warner, R. F., Foster, S. J., & Kilpatrick, A. E. (2007). Reinforced Concrete Basics (1st ed.). Pearson Australia.


def test_example_3_1():
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

    geometry = concrete_rectangular_section(
        b=300,
        d=450,
        dia_top=24,
        n_top=0,
        dia_bot=24,
        n_bot=3,
        n_circle=4,
        cover=48,
        area_top=0,
        area_bot=450,
        conc_mat=concrete,  # type: ignore
        steel_mat=steel,  # type: ignore
    )

    conc_sec = ConcreteSection(geometry)
    props = conc_sec.get_transformed_gross_properties(elastic_modulus=30.1e3)
    cracked_results = conc_sec.calculate_cracked_properties()
    cracked_results.calculate_transformed_properties(elastic_modulus=30.1e3)
    cracked_stress = conc_sec.calculate_cracked_stress(
        cracked_results=cracked_results, m=100e6
    )

    assert pytest.approx(conc_sec.gross_properties.cy, abs=1) == 450 - 234
    assert pytest.approx(props.ixx_c, rel=0.01) == 2.47e9
    assert pytest.approx(cracked_results.m_cr, rel=0.01) == 11.4e6
    assert pytest.approx(cracked_results.d_nc, rel=0.01) == 125
    assert pytest.approx(cracked_results.ixx_c_cr, rel=0.01) == 821e6
    assert pytest.approx(cracked_results.iuu_cr, rel=0.01) == 821e6
    assert pytest.approx(max(cracked_stress.concrete_stresses[0]), rel=0.01) == 15.3
    assert (
        pytest.approx(min(cracked_stress.lumped_reinforcement_stresses), rel=0.01)
        == -213
    )


def test_example_3_2():
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
        flexural_tensile_strength=0.6,
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

    beam = sp_ps.rectangular_section(
        d=800 - 120, b=300, material=concrete  # type: ignore
    ).shift_section(x_offset=350)
    slab = sp_ps.rectangular_section(d=120, b=1000, material=concrete).align_to(  # type: ignore
        other=beam, on="top"
    )
    geom = beam + slab

    # top bars
    geom = add_bar_rectangular_array(
        geometry=geom,
        area=310,
        material=steel,
        n_x=7,
        x_s=920 / 6,
        anchor=(40, 740),
    )

    # bot bars
    geom = add_bar_rectangular_array(
        geometry=geom,
        area=620,
        material=steel,
        n_x=3,
        x_s=212 / 2,
        n_y=2,
        y_s=60,
        anchor=(394, 60),
    )

    conc_sec = ConcreteSection(geom)  # type: ignore
    props = conc_sec.get_transformed_gross_properties(elastic_modulus=30.1e3)
    cracked_results = conc_sec.calculate_cracked_properties()
    cracked_results.calculate_transformed_properties(elastic_modulus=30.1e3)
    cracked_stress = conc_sec.calculate_cracked_stress(
        cracked_results=cracked_results, m=450e6
    )

    assert pytest.approx(conc_sec.gross_properties.cy, abs=1) == 800 - 327
    assert pytest.approx(props.ixx_c, rel=0.01) == 24.1e9
    assert pytest.approx(cracked_results.m_cr, rel=0.01) == 30.6e6
    assert pytest.approx(cracked_results.d_nc, rel=0.01) == 160
    assert pytest.approx(cracked_results.ixx_c_cr, rel=0.01) == 8.9e9
    assert pytest.approx(cracked_results.iuu_cr, rel=0.01) == 8.9e9

    # combine concrete stresses
    conc_stresses = []
    for cs in cracked_stress.concrete_stresses:
        conc_stresses.extend(cs)

    assert pytest.approx(max(conc_stresses), rel=0.01) == 8.1
    assert (
        pytest.approx(max(cracked_stress.lumped_reinforcement_stresses), rel=0.02) == 33
    )
    assert (
        pytest.approx(min(cracked_stress.lumped_reinforcement_stresses), rel=0.01)
        == -193
    )
    assert (
        pytest.approx(cracked_stress.lumped_reinforcement_stresses[-1], rel=0.01)
        == -173
    )


def test_example_3_4():
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
        flexural_tensile_strength=0.6,
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

    beam = sp_ps.rectangular_section(
        d=800 - 120, b=300, material=concrete  # type: ignore
    ).shift_section(x_offset=350)
    slab = sp_ps.rectangular_section(d=120, b=1000, material=concrete).align_to(  # type: ignore
        other=beam, on="top"
    )
    geom = beam + slab

    # top bars
    geom = add_bar_rectangular_array(
        geometry=geom,
        area=800,
        material=steel,
        n_x=7,
        x_s=908 / 6,
        anchor=(46, 740),
    )

    # bot bars
    geom = add_bar_rectangular_array(
        geometry=geom,
        area=620,
        material=steel,
        n_x=3,
        x_s=212 / 2,
        anchor=(394, 60),
    )

    conc_sec = ConcreteSection(geom)  # type: ignore
    cracked_results = conc_sec.calculate_cracked_properties(theta=np.pi)
    cracked_results.calculate_transformed_properties(elastic_modulus=30.1e3)

    assert pytest.approx(cracked_results.d_nc, rel=0.01) == 302
    assert pytest.approx(cracked_results.ixx_c_cr, rel=0.01) == 10.46e9
    assert pytest.approx(cracked_results.iuu_cr, rel=0.01) == 10.46e9


def test_example_3_8():
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

    geometry = concrete_rectangular_section(
        b=300,
        d=450,
        dia_top=24,
        n_top=0,
        dia_bot=24,
        n_bot=4,
        n_circle=4,
        cover=48,
        area_top=0,
        area_bot=450,
        conc_mat=concrete,  # type: ignore
        steel_mat=steel,  # type: ignore
    )

    conc_sec = ConcreteSection(geometry)
    ultimate_results = conc_sec.ultimate_bending_capacity()
    assert pytest.approx(ultimate_results.d_n, abs=1) == 133
    assert pytest.approx(ultimate_results.k_u, rel=0.01) == 133 / 390
    assert pytest.approx(ultimate_results.m_x, rel=0.01) == 302e6


def test_example_3_9():
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

    geometry = concrete_rectangular_section(
        b=300,
        d=450,
        dia_top=24,
        n_top=2,
        dia_bot=24,
        n_bot=4,
        n_circle=4,
        cover=48,
        area_top=450,
        area_bot=450,
        conc_mat=concrete,  # type: ignore
        steel_mat=steel,  # type: ignore
    )

    conc_sec = ConcreteSection(geometry)
    ultimate_results = conc_sec.ultimate_bending_capacity()
    assert pytest.approx(ultimate_results.d_n, rel=0.03) == 100.7
    assert pytest.approx(ultimate_results.k_u, rel=0.03) == 100.7 / 390
    assert pytest.approx(ultimate_results.m_x, rel=0.01) == 309e6


def test_example_3_11():
    concrete = Concrete(
        name="25 MPa Concrete",
        density=2.4e-6,
        stress_strain_profile=ConcreteLinear(elastic_modulus=26.7e3),
        ultimate_stress_strain_profile=RectangularStressBlock(
            compressive_strength=25,
            alpha=0.85,
            gamma=0.85,
            ultimate_strain=0.003,
        ),
        flexural_tensile_strength=0,
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

    beam = sp_ps.rectangular_section(d=726 - 120, b=400, material=concrete)  # type: ignore
    slab = sp_ps.rectangular_section(d=120, b=1100, material=concrete).align_to(  # type: ignore
        other=beam, on="top"
    )
    geom = beam + slab

    # bot bars
    geom = add_bar_rectangular_array(
        geometry=geom,
        area=800,
        material=steel,
        n_x=4,
        x_s=308 / 3,
        n_y=2,
        y_s=60,
        anchor=(46, 46),
    )

    conc_sec = ConcreteSection(geom)  # type: ignore
    ultimate_results = conc_sec.ultimate_bending_capacity()
    assert pytest.approx(ultimate_results.d_n, abs=1) == 196
    assert pytest.approx(ultimate_results.k_u, rel=0.01) == 196 / (726 - 46)
    assert pytest.approx(ultimate_results.m_x, rel=0.01) == 1860e6


def test_example_5_2():
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
        flexural_tensile_strength=3.4,
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

    geom = sp_ps.rectangular_section(d=600, b=400, material=concrete)  # type: ignore
    geom = add_bar(
        geometry=geom,
        area=1200,
        material=steel,
        x=200,
        y=74,
    )
    geom = add_bar(
        geometry=geom,
        area=1200,
        material=steel,
        x=200,
        y=600 - 74,
    )

    conc_sec = ConcreteSection(geom)  # type: ignore
    decomp = conc_sec.calculate_ultimate_section_actions(d_n=526)
    balanced = conc_sec.calculate_ultimate_section_actions(d_n=287)
    pure = conc_sec.ultimate_bending_capacity()

    assert pytest.approx(decomp.n, rel=0.015) == 6108e3
    assert pytest.approx(decomp.m_xy, rel=0.015) == 672e6
    assert pytest.approx(balanced.n, rel=0.015) == 2939e3
    assert pytest.approx(balanced.m_xy, rel=0.015) == 826e6
    assert pytest.approx(pure.n, abs=20) == 0
    assert pytest.approx(pure.m_xy, rel=0.015) == 306e6
