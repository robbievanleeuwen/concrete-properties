"""Tests for the NZS3101 class."""

import numpy as np
import pytest
import sectionproperties.pre.library.primitive_sections as sp_ps
import sectionproperties.pre.library.steel_sections as sp_ss
from sectionproperties.pre.library.concrete_sections import concrete_rectangular_section

import concreteproperties.stress_strain_profile as ssp
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.design_codes.nzs3101 import NZS3101
from concreteproperties.material import Concrete, Steel, SteelBar


def create_dummy_section(
    design_code, prob_section=False, section_type="column"
) -> ConcreteSection:
    """Creates a dummy section.

    Args:
        design_code: Design code to use
        prob_section: Whether or not to use probable strength properties
        section_type: The type of member being analysed

    Returns:
        ConcreteSection object
    """
    # dummy beam section with fixed properties to fulfil design_code.concrete_section
    # checks, not utilised for any section analysis
    concrete = design_code.create_concrete_material(compressive_strength=40)
    steel_grade = "275" if prob_section else "500e"
    steel = design_code.create_steel_material(steel_grade=steel_grade)

    b = 550
    d = 900
    dia = 20
    area = np.pi * dia**2 / 4
    n = 5
    cvr = 30

    geom = concrete_rectangular_section(
        b=b,
        d=d,
        dia_top=dia,
        area_top=area,
        n_top=n,
        c_top=cvr,
        dia_bot=dia,
        area_bot=area,
        n_bot=n,
        c_bot=cvr,
        n_circle=16,
        conc_mat=concrete,
        steel_mat=steel,
    )
    concrete_section = ConcreteSection(geom)
    design_code.assign_concrete_section(
        concrete_section=concrete_section, section_type=section_type
    )

    return design_code.concrete_section


@pytest.mark.parametrize(
    "section_type",
    [
        ("this_is_not_a_valid_section_type"),
    ],
)
def test_nzs3101_assign_concrete_section_valueerror(section_type):
    """Tests assigning concrete sections ValueError."""
    design_code = NZS3101()
    conc_sec = create_dummy_section(design_code)

    with pytest.raises(ValueError, match="The specified section type of"):
        design_code.assign_concrete_section(conc_sec, section_type)


@pytest.mark.parametrize(
    ("compressive_strength", "rel_tol", "calc_value"),
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
    """Tests alpha_1."""
    design_code = NZS3101()
    assert (
        pytest.approx(design_code.alpha_1(compressive_strength), rel=rel_tol)
        == calc_value
    )


@pytest.mark.parametrize(
    ("compressive_strength", "rel_tol", "calc_value"),
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
    """Tests beta_1."""
    design_code = NZS3101()
    assert (
        pytest.approx(design_code.beta_1(compressive_strength), rel=rel_tol)
        == calc_value
    )


@pytest.mark.parametrize(
    ("density", "rel_tol", "calc_value"),
    [
        (2800, 0, 1),
        (2400, 0, 1),
        (2199, 0.00001, 0.999727),
        (2000, 0.00001, 0.945454),
        (1800, 0.00001, 0.890909),
    ],
)
def test_nzs3101_lamda(density, rel_tol, calc_value):
    """Tests lamda."""
    design_code = NZS3101()
    assert pytest.approx(design_code.lamda(density), rel=rel_tol) == calc_value


@pytest.mark.parametrize(
    ("compressive_strength", "density", "prob_design", "calc_value"),
    [
        (25, 2800, True, 3.36805),
        (35, 2400, False, 2.24811),
        (45, 2199, False, 2.54842),
        (45, 2000, True, 4.36549),
        (75, 1800, False, 2.93189),
    ],
)
def test_nzs3101_concrete_tensile_strength(
    compressive_strength, density, prob_design, calc_value
):
    """Tests concrete tensile strength."""
    design_code = NZS3101()
    assert (
        pytest.approx(
            design_code.concrete_tensile_strength(
                compressive_strength, density, prob_design
            ),
            rel=0.0001,
        )
        == calc_value
    )


@pytest.mark.parametrize(
    ("compressive_strength", "density", "calc_value"),
    [
        (25, 2800, 3.00000),
        (35, 2400, 3.54965),
        (45, 2199, 4.02382),
        (45, 2000, 3.80538),
        (75, 1800, 4.62930),
    ],
)
def test_nzs3101_modulus_of_rupture(compressive_strength, density, calc_value):
    """Tests modulus of rupture."""
    design_code = NZS3101()
    assert (
        pytest.approx(
            design_code.modulus_of_rupture(compressive_strength, density), rel=0.0001
        )
        == calc_value
    )


@pytest.mark.parametrize(
    "compressive_strength, ",
    [
        (20),
        (30),
        (40),
        (50),
    ],
)
def test_nzs3101_prob_compressive_strength(compressive_strength):
    """Tests prob compressive strength."""
    design_code = NZS3101()
    assert pytest.approx(
        design_code.prob_compressive_strength(compressive_strength), rel=0.01
    ) == compressive_strength * (1.5 if compressive_strength <= 40 else 1.4)


@pytest.mark.parametrize(
    ("compressive_strength", "density", "rel_tol", "calc_value"),
    [
        (30, 2300, 0.01, 25742.960),
        (50, 1800, 0.01, 23009.112),
        (65, 2800, 0.01, 50897.895),
        (25, 1900, 0.01, 17644.384),
        (45, 2225, 0.01, 29999.041),
        (20, 2100, 0.01, 18337.918),
    ],
)
def test_nzs3101_e_conc_valid(compressive_strength, density, rel_tol, calc_value):
    """Tests validity of elastic modulus."""
    design_code = NZS3101()
    assert (
        pytest.approx(design_code.e_conc(compressive_strength, density), rel=rel_tol)
        == calc_value
    )


def test_nzs3101_e_conc_valueerror():
    """Tests elastic modulus ValueError."""
    design_code = NZS3101()
    with pytest.raises(ValueError, match="The specified concrete density of"):
        design_code.e_conc(20, 1799)
    with pytest.raises(ValueError, match="The specified concrete density of"):
        design_code.e_conc(20, 2801)


@pytest.mark.parametrize(
    ("analysis_type", "section_type", "calc_value"),
    [
        ("nom_chk", "column", (0.85, False, False, False)),
        ("nom_chk", "wall", (0.85, False, False, False)),
        ("nom_chk", "wall_sr_s", (0.7, False, False, False)),
        ("nom_chk", "wall_sr_m", (0.85, False, False, False)),
        ("cpe_chk", "column", (1.0, True, False, False)),
        ("os_chk", "column", (1.0, False, True, False)),
        ("prob_chk", "column", (1.0, False, False, True)),
        ("prob_os_chk", "column", (1.0, False, True, True)),
    ],
)
def test_nzs3101_capacity_reduction_factor(analysis_type, section_type, calc_value):
    """Tests capacity reduction factor."""
    design_code = NZS3101()
    create_dummy_section(design_code, section_type=section_type)
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
    """Tests capacity reduction factor ValueError."""
    design_code = NZS3101()
    create_dummy_section(design_code)

    with pytest.raises(ValueError, match="The specified analysis type of"):
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
    """Tests capacity reduction factor exception."""
    design_code = NZS3101()
    create_dummy_section(design_code, prob_section=True)
    with pytest.raises(RuntimeError):
        design_code.capacity_reduction_factor(analysis_type)


@pytest.mark.parametrize(
    "analysis_type",
    [
        ("this_is_not_a_valid_analysis_type"),
    ],
)
def test_nzs3101_assign_analysis_section_valueerror(analysis_type):
    """Tests assign analysis section ValueError."""
    design_code = NZS3101()
    create_dummy_section(design_code)

    with pytest.raises(ValueError, match="The specified analysis type of"):
        design_code.assign_analysis_section(analysis_type)


@pytest.mark.parametrize(
    ("pphr_class", "compressive_strength"),
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
    """Tests check fc limits ValueError."""
    design_code = NZS3101()
    create_dummy_section(design_code)

    for conc_geom in design_code.concrete_section.concrete_geometries:
        conc_geom.material.ultimate_stress_strain_profile.__setattr__(
            "compressive_strength", compressive_strength
        )

    if pphr_class == "this_is_not_a_valid_pphr_class":
        with pytest.raises(ValueError, match="The specified PPHR class specified"):
            design_code.check_f_c_limits(pphr_class)
    else:
        with pytest.raises(ValueError, match="material must be between"):
            design_code.check_f_c_limits(pphr_class)


@pytest.mark.parametrize(
    ("pphr_class", "compressive_strength"),
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
    """Tests check fc limits valid."""
    design_code = NZS3101()
    create_dummy_section(design_code)

    for conc_geom in design_code.concrete_section.concrete_geometries:
        conc_geom.material.ultimate_stress_strain_profile.__setattr__(
            "compressive_strength", compressive_strength
        )
    assert design_code.check_f_c_limits(pphr_class) is None


@pytest.mark.parametrize(
    "yield_strength",
    [
        (501),
    ],
)
def test_nzs3101_check_f_y_limit_valueerror(yield_strength):
    """Tests check fy limits ValueError."""
    design_code = NZS3101()
    create_dummy_section(design_code)

    for steel_geom in design_code.concrete_section.reinf_geometries_lumped:
        steel_geom.material.stress_strain_profile.__setattr__(
            "yield_strength", yield_strength
        )
    with pytest.raises(ValueError, match="material must be less than"):
        design_code.check_f_y_limit()


@pytest.mark.parametrize(
    "yield_strength",
    [
        (1),
        (500),
    ],
)
def test_nzs3101_check_f_y_limit_valid(yield_strength):
    """Tests check fy limits valid."""
    design_code = NZS3101()
    create_dummy_section(design_code)

    for steel_geom in design_code.concrete_section.reinf_geometries_lumped:
        steel_geom.material.stress_strain_profile.__setattr__(
            "yield_strength", yield_strength
        )
    assert design_code.check_f_y_limit() is None


@pytest.mark.parametrize(
    "n_design",
    [
        (1000000000),
        (-1000000000),
    ],
)
def test_nzs3101_check_axial_limits_valueerror(n_design):
    """Tests check axial limits ValueError."""
    design_code = NZS3101()
    create_dummy_section(design_code)

    if n_design > 0:
        with pytest.raises(ValueError, match="kN, is greater"):
            design_code.check_axial_limits(n_design, 1)

    if n_design < 0:
        with pytest.raises(ValueError, match="kN, is less"):
            design_code.check_axial_limits(n_design, 1)


@pytest.mark.parametrize(
    ("steel_grade", "yield_strength", "fracture_strain", "phi_os"),
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
    """Tests create predefined steel material."""
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
    ("steel_grade", "yield_strength", "fracture_strain", "phi_os"),
    [
        (None, 280, 0.1, 1.25),
    ],
)
def test_nzs3101_create_steel_material_user_defined(
    steel_grade, yield_strength, fracture_strain, phi_os
):
    """Tests create user defined steel material."""
    design_code = NZS3101()
    steel_mat = design_code.create_steel_material(
        steel_grade, yield_strength, fracture_strain, phi_os
    )
    assert (
        pytest.approx(steel_mat.__getattribute__("steel_grade"))
        == f"user_{yield_strength:.0f}"
    )


def test_nzs3101_create_steel_material_meshed_valueerror():
    """Tests meshed steel ValueError."""
    design_code = NZS3101()
    # create concrete material
    concrete = Concrete(
        name="50 MPa Concrete",
        density=2300,
        stress_strain_profile=ssp.ConcreteLinearNoTension(
            elastic_modulus=design_code.e_conc(50),
            ultimate_strain=0.003,
            compressive_strength=50,
        ),
        ultimate_stress_strain_profile=ssp.RectangularStressBlock(
            compressive_strength=50,
            alpha=design_code.alpha_1(50),
            gamma=design_code.beta_1(50),
            ultimate_strain=0.003,
        ),
        flexural_tensile_strength=design_code.concrete_tensile_strength(50),
        colour="lightgrey",
    )

    # create meshed steel material
    steel = Steel(
        name="Meshed Steel",
        density=7850,
        stress_strain_profile=ssp.SteelElasticPlastic(
            yield_strength=300,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="tan",
    )

    # create concrete section
    conc = sp_ps.rectangular_section(d=1000, b=1000, material=concrete)

    # create UC section and centre to concrete section
    uc = sp_ss.i_section(
        d=308,
        b=305,
        t_f=15.4,
        t_w=9.9,
        r=16.5,
        n_r=8,
        material=steel,
    ).align_center(align_to=conc)

    # create geometry
    geom = conc - uc + uc

    concrete_section = ConcreteSection(geom)

    with pytest.raises(ValueError, match="Meshed reinforcement is not supported"):
        design_code.assign_concrete_section(concrete_section=concrete_section)


@pytest.mark.parametrize(
    ("steel_grade", "yield_strength", "fracture_strain", "phi_os"),
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
    """Tests create steel material exception."""
    design_code = NZS3101()
    with pytest.raises(RuntimeError):
        design_code.create_steel_material(
            steel_grade, yield_strength, fracture_strain, phi_os
        )


@pytest.mark.parametrize(
    (
        "compressive_strength",
        "ultimate_strain",
        "density",
        "calc_value_e_conc",
        "calc_value_alpha_1",
        "calc_value_beta_1",
        "calc_value_tensile_strength",
    ),
    [
        (20, 0.004, 2400, 22404.639, 0.85, 0.85, 1.69941),
        (30, 0.004, 2200, 24082.454, 0.85, 0.85, 2.08135),
        (40, 0.004, 2100, 25933.733, 0.85, 0.77, 2.33779),
        (50, 0.004, 1800, 23009.112, 0.85, 0.69, 2.39388),
        (60, 0.004, 1950, 28420.626, 0.83, 0.65, 2.74278),
        (70, 0.004, 2700, 50015.057, 0.79, 0.65, 3.17931),
        (80, 0.004, 2000, 34087.573, 0.75, 0.65, 3.21343),
        (90, 0.004, 2400, 47527.417, 0.75, 0.65, 3.60500),
    ],
)
def test_nzs3101_create_concrete_material(
    compressive_strength,
    ultimate_strain,
    density,
    calc_value_e_conc,
    calc_value_alpha_1,
    calc_value_beta_1,
    calc_value_tensile_strength,
):
    """Tests create concrete material."""
    design_code = NZS3101()
    concrete_mat = design_code.create_concrete_material(
        compressive_strength, ultimate_strain, density
    )
    assert pytest.approx(concrete_mat.__getattribute__("density")) == density
    assert (
        pytest.approx(
            concrete_mat.__getattribute__("flexural_tensile_strength"), rel=0.001
        )
        == calc_value_tensile_strength
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


@pytest.mark.parametrize(
    (
        "yield_strength",
        "fracture_strain",
        "phi_os",
        "compressive_strength",
        "ultimate_strain",
        "density",
        "calc_value_e_conc",
        "calc_value_alpha_1",
        "calc_value_beta_1",
        "calc_value_tensile_strength",
    ),
    [
        (280, 0.1, 1.25, 20, 0.003, 2400, 29638.552, 0.85, 0.81, 2.24811),
        (280, 0.1, 1.25, 27.5, 0.003, 2200, 28663.854, 0.85, 0.75, 2.47730),
        (324, 0.15, 1.25, 30, 0.003, 2100, 27506.878, 0.85, 0.73, 2.47960),
        (324, 0.15, 1.25, 45, 0.003, 1800, 25205.219, 0.83, 0.65, 2.62236),
        (455, 0.12, 1.5, 35, 0.003, 1950, 25944.363, 0.85, 0.69, 2.50380),
        (455, 0.12, 1.5, 55, 0.003, 2700, 50015.057, 0.79, 0.65, 3.17931),
        (464, 0.12, 1.25, 32, 0.003, 2000, 26127.630, 0.85, 0.714, 2.46305),
        (324, 0.15, 1.25, 65, 0.003, 2400, 44809.279, 0.75, 0.65, 3.39882),
        (500, 0.05, 1.5, 25, 0.003, 1875, 21879.531, 0.85, 0.77, 2.19031),
        (540, 0.1, 1.25, 50, 0.003, 2125, 33651.248, 0.81, 0.65, 3.00099),
        (600, 0.015, 1.2, 65, 0.003, 2300, 42038.078, 0.75, 0.65, 3.39882),
        (540, 0.03, 1.2, 37.5, 0.003, 2100, 29710.824, 0.85, 0.67, 2.67827),
        (300, 0.15, 1.35, 52.5, 0.003, 2025, 31900.356, 0.80, 0.65, 2.97301),
        (500, 0.1, 1.35, 62.5, 0.003, 2600, 49729.830, 0.76, 0.65, 3.34530),
    ],
)
def test_nzs3101_create_os_section(
    yield_strength,
    fracture_strain,
    phi_os,
    compressive_strength,
    ultimate_strain,
    density,
    calc_value_e_conc,
    calc_value_alpha_1,
    calc_value_beta_1,
    calc_value_tensile_strength,
):
    """Tests create overstrength section."""
    design_code = NZS3101()
    create_dummy_section(design_code)
    # update section properties for scaling to overstrength
    for steel_geom in design_code.concrete_section.reinf_geometries_lumped:
        steel_geom.material.__setattr__("steel_grade", None)
        steel_geom.material.stress_strain_profile.__setattr__(
            "yield_strength", yield_strength
        )
        steel_geom.material.stress_strain_profile.__setattr__(
            "fracture_strain", fracture_strain
        )
        steel_geom.material.__setattr__("phi_os", phi_os)
    for conc_geom in design_code.concrete_section.concrete_geometries:
        conc_geom.material.ultimate_stress_strain_profile.__setattr__(
            "compressive_strength", compressive_strength
        )
        conc_geom.material.ultimate_stress_strain_profile.__setattr__(
            "ultimate_strain", ultimate_strain
        )
        conc_geom.material.__setattr__("density", density)

    concrete_os_section = design_code.create_os_section()
    # check steel overstrength properties
    for steel_geom in concrete_os_section.reinf_geometries_lumped:
        assert (
            pytest.approx(steel_geom.material.__getattribute__("steel_grade"))
            == f"user_{yield_strength * phi_os:.0f}"
        )
        assert (
            pytest.approx(
                steel_geom.material.stress_strain_profile.__getattribute__(
                    "yield_strength"
                )
            )
            == yield_strength * phi_os
        )
        assert (
            pytest.approx(
                steel_geom.material.stress_strain_profile.__getattribute__(
                    "fracture_strain"
                )
            )
            == fracture_strain
        )
        assert pytest.approx(steel_geom.material.__getattribute__("phi_os")) == phi_os

    # check concrete overstrength properties
    for conc_geom in concrete_os_section.concrete_geometries:
        assert pytest.approx(conc_geom.material.__getattribute__("density")) == density
        assert (
            pytest.approx(
                conc_geom.material.__getattribute__("flexural_tensile_strength"),
                rel=0.001,
            )
            == calc_value_tensile_strength
        )
        assert (
            pytest.approx(
                conc_geom.material.ultimate_stress_strain_profile.__getattribute__(
                    "compressive_strength"
                )
            )
            == compressive_strength + 15
        )
        assert (
            pytest.approx(
                conc_geom.material.ultimate_stress_strain_profile.__getattribute__(
                    "alpha"
                )
            )
            == calc_value_alpha_1
        )
        assert (
            pytest.approx(
                conc_geom.material.ultimate_stress_strain_profile.__getattribute__(
                    "gamma"
                )
            )
            == calc_value_beta_1
        )
        assert (
            pytest.approx(
                conc_geom.material.ultimate_stress_strain_profile.__getattribute__(
                    "ultimate_strain"
                )
            )
            == ultimate_strain
        )
        assert (
            pytest.approx(
                conc_geom.material.stress_strain_profile.__getattribute__(
                    "compressive_strength"
                )
            )
            == compressive_strength + 15
        )
        assert (
            pytest.approx(
                conc_geom.material.stress_strain_profile.__getattribute__(
                    "ultimate_strain"
                )
            )
            == ultimate_strain
        )
        assert (
            pytest.approx(
                conc_geom.material.stress_strain_profile.__getattribute__(
                    "elastic_modulus"
                ),
                rel=0.01,
            )
            == calc_value_e_conc
        )


@pytest.mark.parametrize(
    (
        "yield_strength",
        "fracture_strain",
        "phi_os",
        "compressive_strength",
        "ultimate_strain",
        "calc_value_e_conc",
        "calc_value_alpha_1",
        "calc_value_beta_1",
        "calc_value_tensile_strength",
    ),
    [
        (280, 0.1, 1.25, 20, 0.003, 25742.960, 0.85, 0.85, 3.01247),
        (280, 0.1, 1.25, 27.5, 0.003, 30186.297, 0.85, 0.76, 3.53243),
        (324, 0.15, 1.25, 30, 0.003, 31528.558, 0.85, 0.73, 3.68951),
        (324, 0.15, 1.25, 45, 0.003, 37305.093, 0.818, 0.65, 4.36548),
        (455, 0.12, 1.5, 35, 0.003, 34054.735, 0.85, 0.67, 3.98512),
        (455, 0.12, 1.5, 55, 0.003, 41242.333, 0.762, 0.65, 4.82623),
        (464, 0.12, 1.25, 32, 0.003, 32562.555, 0.85, 0.706, 3.81051),
        (324, 0.15, 1.25, 65, 0.003, 44835.142, 0.75, 0.65, 5.24666),
        (500, 0.05, 1.5, 25, 0.003, 28781.504, 0.85, 0.79, 3.36804),
        (540, 0.1, 1.25, 50, 0.003, 39323.021, 0.79, 0.65, 4.60163),
        (600, 0.015, 1.2, 65, 0.003, 44835.142, 0.75, 0.65, 5.24666),
        (540, 0.03, 1.2, 37.5, 0.003, 35250.000, 0.845, 0.65, 4.12500),
        (300, 0.15, 1.35, 52.5, 0.003, 40294.106, 0.776, 0.65, 4.71526),
        (500, 0.1, 1.35, 62.5, 0.003, 43964.474, 0.75, 0.65, 5.14477),
    ],
)
def test_nzs3101_create_prob_section(
    yield_strength,
    fracture_strain,
    phi_os,
    compressive_strength,
    ultimate_strain,
    calc_value_e_conc,
    calc_value_alpha_1,
    calc_value_beta_1,
    calc_value_tensile_strength,
):
    """Tests create prob section."""
    design_code = NZS3101()
    create_dummy_section(design_code)
    # update section properties for scaling to overstrength
    for steel_geom in design_code.concrete_section.reinf_geometries_lumped:
        steel_geom.material.__setattr__("steel_grade", None)
        steel_geom.material.stress_strain_profile.__setattr__(
            "yield_strength", yield_strength
        )
        steel_geom.material.stress_strain_profile.__setattr__(
            "fracture_strain", fracture_strain
        )
        steel_geom.material.__setattr__("phi_os", phi_os)
    for conc_geom in design_code.concrete_section.concrete_geometries:
        conc_geom.material.ultimate_stress_strain_profile.__setattr__(
            "compressive_strength", compressive_strength
        )
        conc_geom.material.ultimate_stress_strain_profile.__setattr__(
            "ultimate_strain", ultimate_strain
        )
        conc_geom.material.__setattr__("density", 2300)

    concrete_prob_section = design_code.create_prob_section()
    # check steel overstrength properties
    for steel_geom in concrete_prob_section.reinf_geometries_lumped:
        assert (
            pytest.approx(steel_geom.material.__getattribute__("steel_grade"))
            == f"user_{yield_strength * 1.08:.0f}"
        )
        assert (
            pytest.approx(
                steel_geom.material.stress_strain_profile.__getattribute__(
                    "yield_strength"
                )
            )
            == yield_strength * 1.08
        )
        assert (
            pytest.approx(
                steel_geom.material.stress_strain_profile.__getattribute__(
                    "fracture_strain"
                )
            )
            == fracture_strain
        )
        assert pytest.approx(steel_geom.material.__getattribute__("phi_os")) == phi_os

    # check concrete overstrength properties
    for conc_geom in concrete_prob_section.concrete_geometries:
        assert (
            pytest.approx(
                conc_geom.material.__getattribute__("flexural_tensile_strength"),
                rel=0.001,
            )
            == calc_value_tensile_strength
        )
        assert pytest.approx(
            conc_geom.material.ultimate_stress_strain_profile.__getattribute__(
                "compressive_strength"
            )
        ) == compressive_strength * (1.5 if compressive_strength <= 40 else 1.4)
        assert (
            pytest.approx(
                conc_geom.material.ultimate_stress_strain_profile.__getattribute__(
                    "alpha"
                )
            )
            == calc_value_alpha_1
        )
        assert (
            pytest.approx(
                conc_geom.material.ultimate_stress_strain_profile.__getattribute__(
                    "gamma"
                )
            )
            == calc_value_beta_1
        )
        assert (
            pytest.approx(
                conc_geom.material.ultimate_stress_strain_profile.__getattribute__(
                    "ultimate_strain"
                )
            )
            == ultimate_strain
        )
        assert pytest.approx(
            conc_geom.material.stress_strain_profile.__getattribute__(
                "compressive_strength"
            )
        ) == compressive_strength * (1.5 if compressive_strength <= 40 else 1.4)
        assert (
            pytest.approx(
                conc_geom.material.stress_strain_profile.__getattribute__(
                    "ultimate_strain"
                )
            )
            == ultimate_strain
        )
        assert (
            pytest.approx(
                conc_geom.material.stress_strain_profile.__getattribute__(
                    "elastic_modulus"
                ),
                rel=0.01,
            )
            == calc_value_e_conc
        )


@pytest.mark.parametrize(
    (
        "yield_strength",
        "fracture_strain",
        "phi_os",
        "compressive_strength",
        "ultimate_strain",
        "calc_value_e_conc",
        "calc_value_alpha_1",
        "calc_value_beta_1",
        "calc_value_tensile_strength",
    ),
    [
        (280, 0.1, 1.25, 20, 0.003, 25742.960, 0.85, 0.85, 3.01247),
        (280, 0.1, 1.25, 27.5, 0.003, 30186.297, 0.85, 0.76, 3.53243),
        (324, 0.15, 1.25, 30, 0.003, 31528.558, 0.85, 0.73, 3.68951),
        (324, 0.15, 1.25, 45, 0.003, 37305.093, 0.818, 0.65, 4.36548),
        (455, 0.12, 1.5, 35, 0.003, 34054.735, 0.85, 0.67, 3.98512),
        (455, 0.12, 1.5, 55, 0.003, 41242.333, 0.762, 0.65, 4.82623),
        (464, 0.12, 1.25, 32, 0.003, 32562.555, 0.85, 0.706, 3.81051),
        (324, 0.15, 1.25, 65, 0.003, 44835.142, 0.75, 0.65, 5.24666),
        (500, 0.05, 1.5, 25, 0.003, 28781.504, 0.85, 0.79, 3.36804),
        (540, 0.1, 1.25, 50, 0.003, 39323.021, 0.79, 0.65, 4.60163),
        (600, 0.015, 1.2, 65, 0.003, 44835.142, 0.75, 0.65, 5.24666),
        (540, 0.03, 1.2, 37.5, 0.003, 35250.000, 0.845, 0.65, 4.12500),
        (300, 0.15, 1.35, 52.5, 0.003, 40294.106, 0.776, 0.65, 4.71526),
        (500, 0.1, 1.35, 62.5, 0.003, 43964.474, 0.75, 0.65, 5.14477),
    ],
)
def test_nzs3101_create_prob_os_section(
    yield_strength,
    fracture_strain,
    phi_os,
    compressive_strength,
    ultimate_strain,
    calc_value_e_conc,
    calc_value_alpha_1,
    calc_value_beta_1,
    calc_value_tensile_strength,
):
    """Tests create prob overstrength section."""
    design_code = NZS3101()
    create_dummy_section(design_code)
    # update section properties for scaling to overstrength
    for steel_geom in design_code.concrete_section.reinf_geometries_lumped:
        steel_geom.material.__setattr__("steel_grade", None)
        steel_geom.material.stress_strain_profile.__setattr__(
            "yield_strength", yield_strength
        )
        steel_geom.material.stress_strain_profile.__setattr__(
            "fracture_strain", fracture_strain
        )
        steel_geom.material.__setattr__("phi_os", phi_os)
    for conc_geom in design_code.concrete_section.concrete_geometries:
        conc_geom.material.ultimate_stress_strain_profile.__setattr__(
            "compressive_strength", compressive_strength
        )
        conc_geom.material.ultimate_stress_strain_profile.__setattr__(
            "ultimate_strain", ultimate_strain
        )
        conc_geom.material.__setattr__("density", 2300)

    concrete_prob_section = design_code.create_prob_section(os_design=True)
    # check steel overstrength properties
    for steel_geom in concrete_prob_section.reinf_geometries_lumped:
        assert (
            pytest.approx(steel_geom.material.__getattribute__("steel_grade"))
            == f"user_{yield_strength * phi_os:.0f}"
        )
        assert (
            pytest.approx(
                steel_geom.material.stress_strain_profile.__getattribute__(
                    "yield_strength"
                )
            )
            == yield_strength * phi_os
        )
        assert (
            pytest.approx(
                steel_geom.material.stress_strain_profile.__getattribute__(
                    "fracture_strain"
                )
            )
            == fracture_strain
        )
        assert pytest.approx(steel_geom.material.__getattribute__("phi_os")) == phi_os

    # check concrete overstrength properties
    for conc_geom in concrete_prob_section.concrete_geometries:
        assert (
            pytest.approx(
                conc_geom.material.__getattribute__("flexural_tensile_strength"),
                rel=0.001,
            )
            == calc_value_tensile_strength
        )
        assert pytest.approx(
            conc_geom.material.ultimate_stress_strain_profile.__getattribute__(
                "compressive_strength"
            )
        ) == compressive_strength * (1.5 if compressive_strength <= 40 else 1.4)
        assert (
            pytest.approx(
                conc_geom.material.ultimate_stress_strain_profile.__getattribute__(
                    "alpha"
                )
            )
            == calc_value_alpha_1
        )
        assert (
            pytest.approx(
                conc_geom.material.ultimate_stress_strain_profile.__getattribute__(
                    "gamma"
                )
            )
            == calc_value_beta_1
        )
        assert (
            pytest.approx(
                conc_geom.material.ultimate_stress_strain_profile.__getattribute__(
                    "ultimate_strain"
                )
            )
            == ultimate_strain
        )
        assert pytest.approx(
            conc_geom.material.stress_strain_profile.__getattribute__(
                "compressive_strength"
            )
        ) == compressive_strength * (1.5 if compressive_strength <= 40 else 1.4)
        assert (
            pytest.approx(
                conc_geom.material.stress_strain_profile.__getattribute__(
                    "ultimate_strain"
                )
            )
            == ultimate_strain
        )
        assert (
            pytest.approx(
                conc_geom.material.stress_strain_profile.__getattribute__(
                    "elastic_modulus"
                ),
                rel=0.01,
            )
            == calc_value_e_conc
        )


@pytest.mark.parametrize(
    ("analysis_type", "section_type", "rel_tol", "calc_value"),
    [
        ("nom_chk", "column", 0.1, 15549884.85),
        ("cpe_chk", "column", 0.1, 12805787.52),
        ("os_chk", "column", 0.1, 21347712.25),
        ("prob_chk", "column", 0.1, 22262357.41),
        ("prob_os_chk", "column", 0.1, 22622855.17),
        ("nom_chk", "wall", 0.1, 5940000),
        ("nom_chk", "wall_sr_s", 0.1, 297000),
        ("nom_chk", "wall_sr_m", 0.1, 1188000),
    ],
)
def test_nzs3101_max_comp_strength(analysis_type, section_type, rel_tol, calc_value):
    """Tests max compressive strength."""
    design_code = NZS3101()
    create_dummy_section(design_code, section_type=section_type)
    _, cpe_design, os_design, prob_design = design_code.capacity_reduction_factor(
        analysis_type
    )
    assert (
        pytest.approx(
            design_code.max_comp_strength(cpe_design, os_design, prob_design),
            rel=rel_tol,
        )
        == calc_value
    )


@pytest.mark.parametrize("section_type", [("this_is_not_a_valid_section_type")])
def test_nzs3101_max_comp_strength_valueerror(section_type):
    """Tests max compressive strength ValueError."""
    design_code = NZS3101()
    create_dummy_section(design_code)
    design_code.section_type = section_type

    # raises ValueError in concrete_capacity()
    with pytest.raises(ValueError, match="The specified section type"):
        design_code.max_comp_strength()


@pytest.mark.parametrize(
    ("analysis_type", "rel_tol", "calc_value"),
    [
        ("nom_chk", 0.1, 1570796.33),
        ("cpe_chk", 0.1, 1570796.33),
        ("os_chk", 0.1, 2120575.04),
        ("prob_chk", 0.1, 1696460.03),
        ("prob_os_chk", 0.1, 2120575.04),
    ],
)
def test_nzs3101_max_ten_strength(analysis_type, rel_tol, calc_value):
    """Tests max tensile strength."""
    design_code = NZS3101()
    create_dummy_section(design_code)
    _, _, os_design, prob_design = design_code.capacity_reduction_factor(analysis_type)
    assert (
        pytest.approx(
            design_code.max_ten_strength(os_design, prob_design),
            rel=rel_tol,
        )
        == calc_value
    )


@pytest.mark.parametrize(
    (
        "compressive_strength",
        "steel_grade",
        "pphr_class",
        "analysis_type",
        "theta",
        "phi_mn",
        "d_n",
    ),
    [
        (40, "500e", "LDPR", "nom_chk", 0, 732.7142, 77.4952),
        (40, "500e", "NDPR", "cpe_chk", 0, 862.0167, 77.4952),
        (40, "500e", "DPR", "os_chk", 0, 1164.4308, 88.8817),
        (40, "500e", "NDPR", "prob_chk", 0, 940.6050, 72.1093),
        (40, "500e", "NDPR", "prob_os_chk", 0, 1166.9648, 85.1573),
    ],
)
def test_nzs3101_ultimate_bending_capacity_beam_no_axial(
    compressive_strength,
    steel_grade,
    pphr_class,
    analysis_type,
    theta,
    phi_mn,
    d_n,
):
    """Tests ultimate bending capacity (pure bending)."""
    design_code = NZS3101()
    concrete = design_code.create_concrete_material(compressive_strength)
    steel = design_code.create_steel_material(steel_grade)

    dia_top = 20
    area_top = np.pi * dia_top**2 / 4
    dia_bot = 25
    area_bot = np.pi * dia_bot**2 / 4

    geometry = concrete_rectangular_section(
        b=500,
        d=800,
        dia_top=dia_top,
        area_top=area_top,
        n_top=5,
        c_top=50,
        dia_bot=dia_bot,
        area_bot=area_bot,
        n_bot=5,
        c_bot=50,
        n_circle=16,
        conc_mat=concrete,
        steel_mat=steel,
    )
    n_design = 0
    conc_sec = ConcreteSection(geometry)
    design_code.assign_concrete_section(conc_sec)
    ultimate_results, _, _ = design_code.ultimate_bending_capacity(
        pphr_class,
        analysis_type,
        theta,
        n_design,
    )
    assert pytest.approx(ultimate_results.m_x / 1e6, rel=0.001) == phi_mn
    assert pytest.approx(ultimate_results.m_y / 1e6, rel=0.001) == 0
    # temporarily removed as concreteproperites does not agree with independent analysis
    # assert pytest.approx(ultimate_results.d_n, rel=0.001) == d_n


@pytest.mark.parametrize(
    (
        "n_design",
        "compressive_strength",
        "steel_grade",
        "pphr_class",
        "analysis_type",
        "theta",
        "phi_mn",
        "d_n",
    ),
    [
        (1000, 40, "500e", "LDPR", "nom_chk", 0, 1053.2022, 145.4137),
        (-500, 40, "500e", "NDPR", "cpe_chk", 0, 690.0205, 58.0155),
        (2250, 40, "500e", "DPR", "os_chk", 0, 1852.6218, 217.0735),
        (-1250, 40, "500e", "NDPR", "prob_chk", 0, 502.2989, 38.1082),
        (3400, 40, "500e", "NDPR", "prob_os_chk", 0, 2153.6079, 271.8836),
    ],
)
def test_nzs3101_ultimate_bending_capacity_beam_with_axial(
    n_design,
    compressive_strength,
    steel_grade,
    pphr_class,
    analysis_type,
    theta,
    phi_mn,
    d_n,
):
    """Tests ultimate bending capacity (with axial force)."""
    design_code = NZS3101()
    concrete = design_code.create_concrete_material(compressive_strength)
    steel = design_code.create_steel_material(steel_grade)

    dia_top = 20
    area_top = np.pi * dia_top**2 / 4
    dia_bot = 25
    area_bot = np.pi * dia_bot**2 / 4

    geometry = concrete_rectangular_section(
        b=500,
        d=800,
        dia_top=dia_top,
        area_top=area_top,
        n_top=5,
        c_top=50,
        dia_bot=dia_bot,
        area_bot=area_bot,
        n_bot=5,
        c_bot=50,
        n_circle=16,
        conc_mat=concrete,
        steel_mat=steel,
    )
    conc_sec = ConcreteSection(geometry)
    design_code.assign_concrete_section(conc_sec)
    ultimate_results, _, _ = design_code.ultimate_bending_capacity(
        pphr_class,
        analysis_type,
        theta,
        n_design * 1e3,
    )
    assert pytest.approx(ultimate_results.m_x / 1e6, rel=0.001) == phi_mn
    assert pytest.approx(ultimate_results.m_y / 1e6, rel=0.001) == 0
    assert pytest.approx(ultimate_results.d_n, rel=0.001) == d_n


@pytest.mark.parametrize(
    (
        "compressive_strength",
        "steel_grade",
        "pphr_class",
        "analysis_type",
        "theta",
        "n_design",
        "progress_bar",
        "phi_mn",
        "d_n",
    ),
    [
        (40, "500e", "LDPR", "nom_chk", 0, 1000, False, 742.4384, 140.6187),
        (40, "500e", "NDPR", "cpe_chk", 0, -500, True, 519.5037, 69.7683),
        (40, "500e", "DPR", "os_chk", 0, 2250, False, 1254.3396, 185.8820),
        (40, "500e", "NDPR", "prob_chk", 0, -1250, True, 391.8097, 46.2072),
        (40, "500e", "NDPR", "prob_os_chk", 0, 3400, True, 1424.9388, 219.6921),
    ],
)
def test_nzs3101_moment_interaction_diagram(
    compressive_strength,
    steel_grade,
    pphr_class,
    analysis_type,
    theta,
    n_design,
    progress_bar,
    phi_mn,
    d_n,
):
    """Tests moment interaction diagram."""
    design_code = NZS3101()
    concrete = design_code.create_concrete_material(compressive_strength)
    steel = design_code.create_steel_material(steel_grade)

    dia = 20
    area = np.pi * dia**2 / 4

    geometry = concrete_rectangular_section(
        b=600,
        d=600,
        dia_top=dia,
        area_top=area,
        n_top=5,
        c_top=50,
        dia_bot=dia,
        area_bot=area,
        n_bot=5,
        c_bot=50,
        dia_side=dia,
        area_side=area,
        n_side=3,
        c_side=50,
        n_circle=16,
        conc_mat=concrete,
        steel_mat=steel,
    )
    conc_sec = ConcreteSection(geometry)
    design_code.assign_concrete_section(conc_sec)

    phi, _, _, _ = design_code.capacity_reduction_factor(analysis_type)

    mi_results, _, _ = design_code.moment_interaction_diagram(
        pphr_class=pphr_class,
        analysis_type=analysis_type,
        theta=theta,
        control_points=[("N", n_design / phi * 1e3)],
        progress_bar=progress_bar,
    )

    n_list, m_list = mi_results.get_results_lists(moment="m_x")

    n_list = np.abs(np.asarray(n_list) / 1e3 - n_design)
    assert pytest.approx(m_list[n_list.argmin()] / 1e6, rel=0.001) == phi_mn


@pytest.mark.parametrize(
    (
        "compressive_strength",
        "steel_grade",
        "pphr_class",
        "analysis_type",
        "n_design",
        "progress_bar",
        "phi_mn",
        "d_n",
    ),
    [
        (40, "500e", "LDPR", "nom_chk", 1000, True, 742.4384, 140.6187),
        (40, "500e", "NDPR", "cpe_chk", -500, False, 519.5037, 69.7683),
        (40, "500e", "DPR", "os_chk", 2250, True, 1254.3396, 185.8820),
        (40, "500e", "NDPR", "prob_chk", -1250, True, 391.8097, 46.2072),
        (40, "500e", "NDPR", "prob_os_chk", 3400, True, 1424.9388, 219.6921),
    ],
)
def test_nzs3101_biaxial_bending_diagram(
    compressive_strength,
    steel_grade,
    pphr_class,
    analysis_type,
    n_design,
    progress_bar,
    phi_mn,
    d_n,
):
    """Tests biaxial bending diagram."""
    design_code = NZS3101()
    concrete = design_code.create_concrete_material(compressive_strength)
    steel = design_code.create_steel_material(steel_grade)

    dia = 20
    area = np.pi * dia**2 / 4

    geometry = concrete_rectangular_section(
        b=600,
        d=600,
        dia_top=dia,
        area_top=area,
        n_top=5,
        c_top=50,
        dia_bot=dia,
        area_bot=area,
        n_bot=5,
        c_bot=50,
        dia_side=dia,
        area_side=area,
        n_side=3,
        c_side=50,
        n_circle=16,
        conc_mat=concrete,
        steel_mat=steel,
    )
    conc_sec = ConcreteSection(geometry)
    design_code.assign_concrete_section(conc_sec)

    n_points = 4
    bb_results, _ = design_code.biaxial_bending_diagram(
        pphr_class,
        analysis_type,
        n_design * 1e3,
        n_points,
        progress_bar,
    )

    m_x_list, m_y_list = bb_results.get_results_lists()

    assert pytest.approx(m_x_list[0] / 1e6, rel=0.001) == -phi_mn
    assert pytest.approx(m_y_list[0] / 1e6, rel=0.001) == 0
    assert pytest.approx(m_x_list[1] / 1e6, rel=0.001) == 0
    assert pytest.approx(m_y_list[1] / 1e6, rel=0.001) == phi_mn
    assert pytest.approx(m_x_list[2] / 1e6, rel=0.001) == phi_mn
    assert pytest.approx(m_y_list[2] / 1e6, rel=0.001) == 0
    assert pytest.approx(m_x_list[3] / 1e6, rel=0.001) == 0
    assert pytest.approx(m_y_list[3] / 1e6, rel=0.001) == -phi_mn


def test_nzs3101_create_section_with_non_steelbarnz_material():
    """Tests section creation with non-SteelBarNZ."""
    design_code = NZS3101()
    # create concrete material
    concrete = Concrete(
        name="50 MPa Concrete",
        density=2300,
        stress_strain_profile=ssp.ConcreteLinearNoTension(
            elastic_modulus=design_code.e_conc(50),
            ultimate_strain=0.003,
            compressive_strength=50,
        ),
        ultimate_stress_strain_profile=ssp.RectangularStressBlock(
            compressive_strength=50,
            alpha=design_code.alpha_1(50),
            gamma=design_code.beta_1(50),
            ultimate_strain=0.003,
        ),
        flexural_tensile_strength=design_code.concrete_tensile_strength(50),
        colour="lightgrey",
    )

    # create non SteelBarNZ steel material
    steel = SteelBar(
        name="Meshed Steel",
        density=7850,
        stress_strain_profile=ssp.SteelElasticPlastic(
            yield_strength=300,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="red",
    )

    # create concrete section
    conc = sp_ps.rectangular_section(d=1000, b=1000, material=concrete)

    # create UC section from SteelBar material and centre to concrete section
    uc = sp_ss.i_section(
        d=308,
        b=305,
        t_f=15.4,
        t_w=9.9,
        r=16.5,
        n_r=8,
        material=steel,
    ).align_center(align_to=conc)

    # create geometry
    geom = conc - uc + uc

    concrete_section = ConcreteSection(geom)
    design_code.concrete_section = concrete_section
    design_code.section_type = "column"

    with pytest.raises(ValueError, match="Material must be a SteelBarNZ"):
        design_code.create_os_section()
    with pytest.raises(ValueError, match="Material must be a SteelBarNZ"):
        design_code.create_prob_section()
    with pytest.raises(ValueError, match="Material must be a SteelBarNZ"):
        design_code.capacity_reduction_factor(analysis_type="nom_chk")
    with pytest.raises(ValueError, match="Material must be a SteelBarNZ"):
        design_code.check_f_y_limit()
    with pytest.raises(ValueError, match="Material must be a SteelBarNZ"):
        design_code.steel_capacity()
