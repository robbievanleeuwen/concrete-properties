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


@pytest.mark.parametrize(
    "steel_grade, yield_strength, fracture_strain, phi_os",
    [
        ("pre_1945", 280, 0.1, 1.25),
        ("33", 280, 0.1, 1.25),
        ("40", 324, 0.15, 1.25),
        ("275", 324, 0.15, 1.25),
        ("hy60", 455, 0.12, 1.5),
        ("380", 455, 0.12, 1.5),
        ("430", 464, 0.12, 1.25),
        ("300", 324, 0.15, 1.25),
        ("500n", 500, 0.05, 1.5),
        ("500", 540, 0.1, 1.25),
        ("cd_mesh", 600, 0.015, 1.2),
        ("duc_mesh", 540, 0.03, 1.2),
        ("300e", 300, 0.15, 1.35),
        ("500e", 500, 0.1, 1.35),
    ],
)
def test_nzs3101_create_steel_material_predefined(
    steel_grade, yield_strength, fracture_strain, phi_os
):
    design_code = NZS3101()
    steel_mat = design_code.create_steel_material(steel_grade)
    assert pytest.approx(steel_mat.__getattribute__("steel_grade")) == steel_grade
    assert (
        pytest.approx(
            steel_mat.stress_strain_profile.__getattribute__("yield_strength")
        )
        == yield_strength
    )
    assert (
        pytest.approx(
            steel_mat.stress_strain_profile.__getattribute__("fracture_strain")
        )
        == fracture_strain
    )
    assert pytest.approx(steel_mat.__getattribute__("phi_os")) == phi_os


@pytest.mark.parametrize(
    "steel_grade, yield_strength, fracture_strain, phi_os",
    [
        (None, 280, 0.1, 1.25),
    ],
)
def test_nzs3101_create_steel_material_user_defined(
    steel_grade, yield_strength, fracture_strain, phi_os
):
    design_code = NZS3101()
    steel_mat = design_code.create_steel_material(
        steel_grade, yield_strength, fracture_strain, phi_os
    )
    assert (
        pytest.approx(steel_mat.__getattribute__("steel_grade"))
        == f"user_{yield_strength:.0f}"
    )


@pytest.mark.parametrize(
    "steel_grade, yield_strength, fracture_strain, phi_os",
    [
        (None, 500, None, None),
        (None, None, 0.1, None),
        (None, None, None, 1.35),
        (None, None, None, None),
        (
            "this_is_not_a_predefined_steel_grade_without_all_the_properties_required",
            None,
            0.1,
            None,
        ),
    ],
)
def test_nzs3101_create_steel_material_exception(
    steel_grade, yield_strength, fracture_strain, phi_os
):
    design_code = NZS3101()
    with pytest.raises(Exception):
        design_code.create_steel_material(
            steel_grade, yield_strength, fracture_strain, phi_os
        )


@pytest.mark.parametrize(
    "compressive_strength, ultimate_strain, density, calc_value_e_conc, "
    "calc_value_alpha_1, calc_value_beta_1, calc_value_modulus_of_rupture",
    [
        (20, 0.004, 2400, 22404.639, 0.85, 0.85, 2.68328),
        (30, 0.004, 2200, 24082.454, 0.85, 0.85, 3.28633),
        (40, 0.004, 2100, 25933.733, 0.85, 0.77, 3.69124),
        (50, 0.004, 1800, 23009.112, 0.85, 0.69, 3.77980),
        (60, 0.004, 1950, 28420.626, 0.83, 0.65, 4.3306),
        (70, 0.004, 2700, 50015.057, 0.79, 0.65, 5.01996),
        (80, 0.004, 2000, 34087.573, 0.75, 0.65, 5.0738),
        (90, 0.004, 2400, 47527.417, 0.75, 0.65, 5.69209),
    ],
)
def test_nzs3101_create_concrete_material(
    compressive_strength,
    ultimate_strain,
    density,
    calc_value_e_conc,
    calc_value_alpha_1,
    calc_value_beta_1,
    calc_value_modulus_of_rupture,
):
    design_code = NZS3101()
    concrete_mat = design_code.create_concrete_material(
        compressive_strength, ultimate_strain, density
    )
    assert pytest.approx(concrete_mat.__getattribute__("density")) == density
    assert (
        pytest.approx(
            concrete_mat.__getattribute__("flexural_tensile_strength"), rel=0.001
        )
        == calc_value_modulus_of_rupture
    )

    assert (
        pytest.approx(
            concrete_mat.ultimate_stress_strain_profile.__getattribute__(
                "compressive_strength"
            )
        )
        == compressive_strength
    )
    assert (
        pytest.approx(
            concrete_mat.ultimate_stress_strain_profile.__getattribute__("alpha")
        )
        == calc_value_alpha_1
    )
    assert (
        pytest.approx(
            concrete_mat.ultimate_stress_strain_profile.__getattribute__("gamma")
        )
        == calc_value_beta_1
    )
    assert (
        pytest.approx(
            concrete_mat.ultimate_stress_strain_profile.__getattribute__(
                "ultimate_strain"
            )
        )
        == ultimate_strain
    )

    assert (
        pytest.approx(
            concrete_mat.stress_strain_profile.__getattribute__("compressive_strength")
        )
        == compressive_strength
    )
    assert (
        pytest.approx(
            concrete_mat.stress_strain_profile.__getattribute__("ultimate_strain")
        )
        == ultimate_strain
    )
    assert (
        pytest.approx(
            concrete_mat.stress_strain_profile.__getattribute__("elastic_modulus"),
            rel=0.01,
        )
        == calc_value_e_conc
    )
