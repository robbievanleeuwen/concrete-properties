import pytest

from concreteproperties.material import Concrete, Steel
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.stress_strain_profile import (
    WhitneyStressBlock,
    SteelElasticPlastic,
)

from sectionproperties.pre.library.concrete_sections import (
    concrete_rectangular_section,
    concrete_tee_section,
)
from sectionproperties.pre.library.primitive_sections import circular_section_by_area

# All examples come from:
# Warner, R. F., Foster, S. J., & Kilpatrick, A. E. (2007). Reinforced Concrete Basics (1st ed.). Pearson Australia.


def test_example_3_1():
    concrete = Concrete(
        name="32 MPa Concrete",
        elastic_modulus=30.1e3,
        density=2.4e-6,
        ultimate_stress_strain_profile=WhitneyStressBlock(
            alpha_2=0.85,
            gamma=0.83,
            compressive_strength=32,
            ultimate_strain=0.003,
        ),
        alpha_1=0.85,
        flexural_tensile_strength=3.4,
        residual_shrinkage_stress=2.4,
        colour="lightgrey",
    )

    steel = Steel(
        name="500 MPa Steel",
        elastic_modulus=200e3,
        density=7.85e-6,
        yield_strength=500,
        ultimate_stress_strain_profile=SteelElasticPlastic(
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
        conc_mat=concrete,
        steel_mat=steel,
    )

    conc_sec = ConcreteSection(geometry)
    props = conc_sec.get_transformed_gross_properties(elastic_modulus=30.1e3)
    m_c = conc_sec.calculate_cracking_moment()

    assert pytest.approx(conc_sec.gross_properties.cy, abs=1) == 450 - 234
    assert pytest.approx(props.ixx_c, rel=0.01) == 2.47e9
    assert pytest.approx(m_c, rel=0.01) == 11.4e6

    # TODO: cracked neutral axis, cracking second moment of area, stresses


def test_example_3_2():
    concrete = Concrete(
        name="32 MPa Concrete",
        elastic_modulus=30.1e3,
        density=2.4e-6,
        ultimate_stress_strain_profile=WhitneyStressBlock(
            alpha_2=0.85,
            gamma=0.83,
            compressive_strength=32,
            ultimate_strain=0.003,
        ),
        alpha_1=0.85,
        flexural_tensile_strength=3.4,
        residual_shrinkage_stress=2.8,
        colour="lightgrey",
    )

    steel = Steel(
        name="500 MPa Steel",
        elastic_modulus=200e3,
        density=7.85e-6,
        yield_strength=500,
        ultimate_stress_strain_profile=SteelElasticPlastic(
            yield_strength=500,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="grey",
    )

    geom = concrete_tee_section(
        b=300,
        d=800,
        b_f=1000,
        d_f=120,
        dia_top=20,
        n_top=0,
        dia_bot=28,
        n_bot=0,
        n_circle=4,
        cover=30,
        conc_mat=concrete,
        steel_mat=steel,
    ).shift_section(x_offset=350)

    # top bars
    for idx in range(7):
        bar = circular_section_by_area(area=310, n=4, material=steel).shift_section(
            x_offset=40 + idx * 920 / 6, y_offset=740
        )

        geom = (geom - bar) + bar

    # bot bars 1
    for idx in range(3):
        bar = circular_section_by_area(area=620, n=4, material=steel).shift_section(
            x_offset=394 + idx * 212 / 2, y_offset=60
        )

        geom = (geom - bar) + bar

    # bot bars 2
    for idx in range(3):
        bar = circular_section_by_area(area=620, n=4, material=steel).shift_section(
            x_offset=394 + idx * 212 / 2, y_offset=120
        )

        geom = (geom - bar) + bar

    conc_sec = ConcreteSection(geom)
    props = conc_sec.get_transformed_gross_properties(elastic_modulus=30.1e3)
    m_c = conc_sec.calculate_cracking_moment()

    assert pytest.approx(conc_sec.gross_properties.cy, abs=1) == 800 - 327
    assert pytest.approx(props.ixx_c, rel=0.01) == 24.1e9
    assert pytest.approx(m_c, rel=0.01) == 30.6e6

    # TODO: cracked neutral axis, cracking second moment of area, stresses


def test_example_3_4():
    concrete = Concrete(
        name="32 MPa Concrete",
        elastic_modulus=30.1e3,
        density=2.4e-6,
        ultimate_stress_strain_profile=WhitneyStressBlock(
            alpha_2=0.85,
            gamma=0.83,
            compressive_strength=32,
            ultimate_strain=0.003,
        ),
        alpha_1=0.85,
        flexural_tensile_strength=3.4,
        residual_shrinkage_stress=2.8,
        colour="lightgrey",
    )

    steel = Steel(
        name="500 MPa Steel",
        elastic_modulus=200e3,
        density=7.85e-6,
        yield_strength=500,
        ultimate_stress_strain_profile=SteelElasticPlastic(
            yield_strength=500,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="grey",
    )

    geom = concrete_tee_section(
        b=300,
        d=800,
        b_f=1000,
        d_f=120,
        dia_top=20,
        n_top=0,
        dia_bot=28,
        n_bot=0,
        n_circle=4,
        cover=30,
        conc_mat=concrete,
        steel_mat=steel,
    ).shift_section(x_offset=350)

    # top bars
    for idx in range(7):
        bar = circular_section_by_area(area=800, n=4, material=steel).shift_section(
            x_offset=46 + idx * 908 / 6, y_offset=740
        )

        geom = (geom - bar) + bar

    # bot bars
    for idx in range(3):
        bar = circular_section_by_area(area=620, n=4, material=steel).shift_section(
            x_offset=394 + idx * 212 / 2, y_offset=60
        )

        geom = (geom - bar) + bar

    conc_sec = ConcreteSection(geom)

    # TODO: cracked neutral axis, cracking second moment of area


def test_example_3_8():
    concrete = Concrete(
        name="32 MPa Concrete",
        elastic_modulus=30.1e3,
        density=2.4e-6,
        ultimate_stress_strain_profile=WhitneyStressBlock(
            alpha_2=0.85,
            gamma=0.83,
            compressive_strength=32,
            ultimate_strain=0.003,
        ),
        alpha_1=0.85,
        flexural_tensile_strength=3.4,
        residual_shrinkage_stress=2.4,
        colour="lightgrey",
    )

    steel = Steel(
        name="500 MPa Steel",
        elastic_modulus=200e3,
        density=7.85e-6,
        yield_strength=500,
        ultimate_stress_strain_profile=SteelElasticPlastic(
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
        conc_mat=concrete,
        steel_mat=steel,
    )

    conc_sec = ConcreteSection(geometry)
    n, mx, my, mv, d_n = conc_sec.ultimate_bending_capacity()
    assert pytest.approx(d_n, abs=1) == 133
    assert pytest.approx(mx, rel=0.01) == 302e6


def test_example_3_9():
    concrete = Concrete(
        name="32 MPa Concrete",
        elastic_modulus=30.1e3,
        density=2.4e-6,
        ultimate_stress_strain_profile=WhitneyStressBlock(
            alpha_2=0.85,
            gamma=0.83,
            compressive_strength=32,
            ultimate_strain=0.003,
        ),
        alpha_1=0.85,
        flexural_tensile_strength=3.4,
        residual_shrinkage_stress=2.4,
        colour="lightgrey",
    )

    steel = Steel(
        name="500 MPa Steel",
        elastic_modulus=200e3,
        density=7.85e-6,
        yield_strength=500,
        ultimate_stress_strain_profile=SteelElasticPlastic(
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
        conc_mat=concrete,
        steel_mat=steel,
    )

    conc_sec = ConcreteSection(geometry)
    n, mx, my, mv, d_n = conc_sec.ultimate_bending_capacity()
    assert pytest.approx(d_n, rel=0.03) == 100.7
    assert pytest.approx(mx, rel=0.01) == 309e6


def test_example_3_11():
    concrete = Concrete(
        name="25 MPa Concrete",
        elastic_modulus=26.7e3,
        density=2.4e-6,
        ultimate_stress_strain_profile=WhitneyStressBlock(
            alpha_2=0.85,
            gamma=0.85,
            compressive_strength=25,
            ultimate_strain=0.003,
        ),
        alpha_1=0.85,
        flexural_tensile_strength=0,
        residual_shrinkage_stress=0,
        colour="lightgrey",
    )

    steel = Steel(
        name="500 MPa Steel",
        elastic_modulus=200e3,
        density=7.85e-6,
        yield_strength=500,
        ultimate_stress_strain_profile=SteelElasticPlastic(
            yield_strength=500,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="grey",
    )

    geom = concrete_tee_section(
        b=400,
        d=726,
        b_f=1100,
        d_f=120,
        dia_top=20,
        n_top=0,
        dia_bot=28,
        n_bot=0,
        n_circle=4,
        cover=30,
        conc_mat=concrete,
        steel_mat=steel,
    )

    # bot bars 1
    for idx in range(4):
        bar = circular_section_by_area(area=800, n=4, material=steel).shift_section(
            x_offset=46 + idx * 308 / 3, y_offset=46
        )

        geom = (geom - bar) + bar

    # bot bars 2
    for idx in range(4):
        bar = circular_section_by_area(area=800, n=4, material=steel).shift_section(
            x_offset=46 + idx * 308 / 3, y_offset=106
        )

        geom = (geom - bar) + bar

    conc_sec = ConcreteSection(geom)
    n, mx, my, mv, d_n = conc_sec.ultimate_bending_capacity()
    assert pytest.approx(d_n, abs=1) == 196
    assert pytest.approx(mx, rel=0.01) == 1860e6


def test_example_3_14():
    pass
    # TODO: implement!
