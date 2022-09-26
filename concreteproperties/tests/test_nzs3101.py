import pytest
import numpy as np
from concreteproperties.design_codes.nzs3101 import NZS3101
from concreteproperties.concrete_section import ConcreteSection
from sectionproperties.pre.library.concrete_sections import concrete_rectangular_section


def create_dummy_section(design_code, prob_section=False):
    # dummy beam section with fixed properties to fulfil design_code.concrete_section
    # checks, not utilised for any section analysis
    concrete = design_code.create_concrete_material(compressive_strength=40)
    steel_grade = "275" if prob_section else "500e"
    steel = design_code.create_steel_material(steel_grade=steel_grade)

    b = 550
    d = 900
    dia = 20
    n = 5
    cvr = 30

    geom = concrete_rectangular_section(
        b=b,
        d=d,
        dia_top=dia,
        n_top=n,
        dia_bot=dia,
        n_bot=n,
        n_circle=16,
        cover=cvr,
        area_top=dia**2 * np.pi / 4,
        area_bot=dia**2 * np.pi / 4,
        conc_mat=concrete,
        steel_mat=steel,
    )
    concrete_section = ConcreteSection(geom)
    design_code.assign_concrete_section(concrete_section=concrete_section)

    return design_code.concrete_section


@pytest.mark.parametrize(
    "compressive_strength, rel_tol, calc_value",
    [
        (20, 0, 0.85),
        (36, 0, 0.85),
        (55, 0, 0.85),
        (56, 0, 0.846),
        (60, 0, 0.83),
        (78.5, 0, 0.756),
        (80, 0, 0.75),
        (90, 0, 0.75),
    ],
)
def test_nzs3101_alpha_1(compressive_strength, rel_tol, calc_value):
    design_code = NZS3101()
    assert (
        pytest.approx(design_code.alpha_1(compressive_strength), rel=rel_tol)
        == calc_value
    )


@pytest.mark.parametrize(
    "compressive_strength, rel_tol, calc_value",
    [
        (20, 0, 0.85),
        (30, 0, 0.85),
        (35, 0, 0.81),
        (42.5, 0, 0.75),
        (47.5, 0, 0.71),
        (54, 0, 0.658),
        (55, 0, 0.65),
        (90, 0, 0.65),
    ],
)
def test_nzs3101_beta_1(compressive_strength, rel_tol, calc_value):
    design_code = NZS3101()
    assert (
        pytest.approx(design_code.beta_1(compressive_strength), rel=rel_tol)
        == calc_value
    )


@pytest.mark.parametrize(
    "density, rel_tol, calc_value",
    [
        (2800, 0, 1),
        (2400, 0, 1),
        (2199, 0.00001, 0.999727),
        (2000, 0.00001, 0.945454),
        (1800, 0.00001, 0.890909),
    ],
)
def test_nzs3101_lamda(density, rel_tol, calc_value):
    design_code = NZS3101()
    assert pytest.approx(design_code.lamda(density), rel=rel_tol) == calc_value


@pytest.mark.parametrize(
    "compressive_strength, density, rel_tol, calc_value",
    [
        (30, 2300, 0.01, 25742.960),
        (50, 1800, 0.01, 23009.112),
        (65, 2800, 0.01, 50897.895),
        (25, 1900, 0.01, 17644.384),
        (45, 2225, 0.01, 29999.041),
        (20, 2100, 0.01, 18337.918),
    ],
)
def test_nzs3101_e_conc(compressive_strength, density, rel_tol, calc_value):
    design_code = NZS3101()
    assert (
        pytest.approx(design_code.e_conc(compressive_strength, density), rel=rel_tol)
        == calc_value
    )
    with pytest.raises(ValueError):
        design_code.e_conc(20, 1500)


@pytest.mark.parametrize(
    "analysis_type, calc_value",
    [
        ("nom_chk", (0.85, False, False, False)),
        ("cpe_chk", (1.0, True, False, False)),
        ("os_chk", (1.0, False, True, False)),
        ("prob_chk", (1.0, False, False, True)),
        ("prob_os_chk", (1.0, False, True, True)),
    ],
)
def test_nzs3101_capacity_reduction_factor(analysis_type, calc_value):
    design_code = NZS3101()
    create_dummy_section(design_code)
    assert (
        pytest.approx(design_code.capacity_reduction_factor(analysis_type))
        == calc_value
    )


@pytest.mark.parametrize(
    "analysis_type",
    [
        ("this_is_not_a_valid_analysis_type"),
    ],
)
def test_nzs3101_capacity_reduction_factor_valueerror(analysis_type):
    design_code = NZS3101()
    create_dummy_section(design_code)
    with pytest.raises(ValueError):
        design_code.capacity_reduction_factor(analysis_type)


@pytest.mark.parametrize(
    "analysis_type",
    [
        ("nom_chk"),
        ("cpe_chk"),
        ("os_chk"),
    ],
)
def test_nzs3101_capacity_reduction_factor_exception(analysis_type):
    design_code = NZS3101()
    create_dummy_section(design_code)
    with pytest.raises(Exception):
        design_code.capacity_reduction_factor(analysis_type, prob_section=True)


@pytest.mark.parametrize(
    "pphr_class, compressive_strength",
    [
        ("this_is_not_a_valid_pphr_class", 20),
        ("NDPR", 19),
        ("NDPR", 101),
        ("LDPR", 19),
        ("LDPR", 71),
        ("DPR", 19),
        ("DPR", 71),
    ],
)
def test_nzs3101_check_f_c_limits_valueerror(pphr_class, compressive_strength):
    design_code = NZS3101()
    create_dummy_section(design_code)

    for conc_geom in design_code.concrete_section.concrete_geometries:
        conc_geom.material.ultimate_stress_strain_profile.__setattr__(
            "compressive_strength", compressive_strength
        )
    with pytest.raises(ValueError):
        design_code.check_f_c_limits(pphr_class)


@pytest.mark.parametrize(
    "pphr_class, compressive_strength",
    [
        ("NDPR", 20),
        ("NDPR", 100),
        ("LDPR", 20),
        ("LDPR", 70),
        ("DPR", 20),
        ("DPR", 70),
    ],
)
def test_nzs3101_check_f_c_limits_valid(pphr_class, compressive_strength):
    design_code = NZS3101()
    create_dummy_section(design_code)

    for conc_geom in design_code.concrete_section.concrete_geometries:
        conc_geom.material.ultimate_stress_strain_profile.__setattr__(
            "compressive_strength", compressive_strength
        )
    try:
        design_code.check_f_c_limits(pphr_class)
    except ValueError:
        assert False


@pytest.mark.parametrize(
    "yield_strength",
    [
        (501),
    ],
)
def test_nzs3101_check_f_y_limit_valueerror(yield_strength):
    design_code = NZS3101()
    create_dummy_section(design_code)

    for steel_geom in design_code.concrete_section.reinf_geometries_lumped:
        steel_geom.material.stress_strain_profile.__setattr__(
            "yield_strength", yield_strength
        )
    with pytest.raises(ValueError):
        design_code.check_f_y_limit()


@pytest.mark.parametrize(
    "yield_strength",
    [
        (1),
        (500),
    ],
)
def test_nzs3101_check_f_y_limit_valid(yield_strength):
    design_code = NZS3101()
    create_dummy_section(design_code)

    for steel_geom in design_code.concrete_section.reinf_geometries_lumped:
        steel_geom.material.stress_strain_profile.__setattr__(
            "yield_strength", yield_strength
        )
    try:
        design_code.check_f_y_limit()
    except ValueError:
        assert False
