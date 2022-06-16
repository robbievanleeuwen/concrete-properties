import pytest

from concreteproperties.material import Concrete, Steel
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.stress_strain_profile import (
    WhitneyStressBlock,
    SteelElasticPlastic,
)

from sectionproperties.pre.library.concrete_sections import concrete_rectangular_section

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
