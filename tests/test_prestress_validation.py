"""Test examples for prestressed concrete sections from Gilbert et. al."""

import pytest
from sectionproperties.analysis.section import Section
from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.library.bridge_sections import i_girder_section
from sectionproperties.pre.library.primitive_sections import rectangular_section

import concreteproperties.stress_strain_profile as ssp
from concreteproperties.material import Concrete, SteelBar, SteelStrand
from concreteproperties.pre import add_bar, add_bar_rectangular_array
from concreteproperties.prestressed_section import PrestressedSection

# All examples come from:
# Gilbert, R. I., Mickleborough, N. C., & Ranzi, G. (2016). Design of Prestressed
# Concrete to AS3600-2009 (2nd ed.). CRC Press.


def test_example_5_4():
    """Tests Example 5.4."""
    # create materials
    concrete = Concrete(
        name="40 MPa Concrete",
        density=2.4e-6,
        stress_strain_profile=ssp.ConcreteLinear(elastic_modulus=32.0e3),
        ultimate_stress_strain_profile=ssp.RectangularStressBlock(
            compressive_strength=40,
            alpha=0.79,
            gamma=0.87,
            ultimate_strain=0.003,
        ),
        flexural_tensile_strength=3.8,
        colour="lightgrey",
    )

    steel = SteelBar(
        name="500 MPa Steel",
        density=7.85e-6,
        stress_strain_profile=ssp.SteelElasticPlastic(
            yield_strength=500,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="grey",
    )

    strand = SteelStrand(
        name="1830 MPa Strand",
        density=7.85e-6,
        stress_strain_profile=ssp.StrandHardening(
            yield_strength=1500,
            elastic_modulus=200e3,
            fracture_strain=0.035,
            breaking_strength=1830,
        ),
        colour="slategrey",
        prestress_stress=1250,
    )

    # create geometry and check gross section properties
    geom = i_girder_section(girder_type=3)
    geom.create_mesh(mesh_sizes=0)
    gross_sec = Section(geom)
    gross_sec.calculate_geometric_properties()

    assert pytest.approx(gross_sec.get_area(), rel=1e-3) == 317e3
    assert pytest.approx(gross_sec.get_ic()[0], rel=1e-3) == 49.9e9
    assert pytest.approx(gross_sec.get_c()[1], rel=1e-3) == -602

    # recreate geometry with materials
    geom = i_girder_section(girder_type=3, material=concrete)

    # add top steel bars
    geom = add_bar_rectangular_array(
        geometry=geom,
        area=450,
        material=steel,
        n_x=2,
        x_s=120,
        anchor=(-60, -60),
    )

    # add bottom steel bars
    geom = add_bar_rectangular_array(
        geometry=geom,
        area=450,
        material=steel,
        n_x=4,
        x_s=120,
        anchor=(-180, -1090),
    )

    # prestressing
    n_p1 = 3
    n_p2 = 5
    n_p3 = 8
    d_p1 = -880
    d_p2 = -945
    d_p3 = -1010
    x_s = 50

    # add prestressing strand layer 1
    geom = add_bar_rectangular_array(
        geometry=geom,
        area=100,
        material=strand,
        n_x=n_p1,
        x_s=x_s,
        anchor=((n_p1 / 2 - 0.5) * -x_s, d_p1),
    )

    # add prestressing strand layer 2
    geom = add_bar_rectangular_array(
        geometry=geom,
        area=100,
        material=strand,
        n_x=n_p2,
        x_s=x_s,
        anchor=((n_p2 / 2 - 0.5) * -x_s, d_p2),
    )

    # add prestressing strand layer 3
    geom = add_bar_rectangular_array(
        geometry=geom,
        area=100,
        material=strand,
        n_x=n_p3,
        x_s=x_s,
        anchor=((n_p3 / 2 - 0.5) * -x_s, d_p3),
    )

    conc_sec = PrestressedSection(geometry=geom)

    # apply external forces
    d_f = 300
    n_ext = 100e3  # at d_f from top
    m_ext = 1000e6  # at d_f from top
    cy = conc_sec.get_gross_properties().cy
    m_ext_c = m_ext - n_ext * (cy + d_f)

    # calculate elastic stress
    res = conc_sec.calculate_uncracked_stress(n=n_ext, m=m_ext_c)
    min_c, max_c = res.get_concrete_stress_limits()
    max_steel = max(res.lumped_reinforcement_stresses)
    min_steel = min(res.lumped_reinforcement_stresses)
    strand_1 = res.strand_stresses[0]
    strand_2 = res.strand_stresses[3]
    strand_3 = res.strand_stresses[-1]

    # check elastic stresses
    assert pytest.approx(max_c, rel=1e-2) == 9.97
    assert pytest.approx(min_c, rel=1e-2) == 2.91
    assert pytest.approx(max_steel, rel=1e-2) == 60.0
    assert pytest.approx(min_steel, rel=1e-2) == 20.5
    assert pytest.approx(strand_1, rel=1e-3) == -1221.4
    assert pytest.approx(strand_2, rel=1e-3) == -1223.9
    assert pytest.approx(strand_3, rel=1e-3) == -1226.4


def test_example_5_7():
    """Tests Example 5.7."""
    # create materials
    concrete = Concrete(
        name="40 MPa Concrete",
        density=2.4e-6,
        stress_strain_profile=ssp.ConcreteLinear(elastic_modulus=30e3),
        ultimate_stress_strain_profile=ssp.RectangularStressBlock(
            compressive_strength=40,
            alpha=0.79,
            gamma=0.87,
            ultimate_strain=0.003,
        ),
        flexural_tensile_strength=0,
        colour="lightgrey",
    )

    steel = SteelBar(
        name="500 MPa Steel",
        density=7.85e-6,
        stress_strain_profile=ssp.SteelElasticPlastic(
            yield_strength=500,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="grey",
    )

    strand = SteelStrand(
        name="1830 MPa Strand",
        density=7.85e-6,
        stress_strain_profile=ssp.StrandHardening(
            yield_strength=1500,
            elastic_modulus=200e3,
            fracture_strain=0.035,
            breaking_strength=1830,
        ),
        colour="slategrey",
        prestress_stress=1200,
    )

    # create geometry
    geom = rectangular_section(d=750, b=200, material=concrete)

    # add top steel bars
    geom = add_bar_rectangular_array(
        geometry=geom, area=250, material=steel, n_x=2, x_s=100, anchor=(50, 700)
    )

    # add bottom steel bars
    geom = add_bar_rectangular_array(
        geometry=geom, area=250, material=steel, n_x=4, x_s=100 / 3, anchor=(50, 50)
    )

    # add prestressing strand
    geom = add_bar(
        geometry=geom,
        area=750,
        material=strand,
        x=100,
        y=750 - 575,
    )

    conc_sec = PrestressedSection(geometry=geom)

    # calculate cracked stress
    cr = conc_sec.calculate_cracked_properties(m_ext=400e6)
    res = conc_sec.calculate_cracked_stress(cracked_results=cr)
    min_c, max_c = res.get_concrete_stress_limits()
    max_steel = max(res.lumped_reinforcement_stresses)
    min_steel = min(res.lumped_reinforcement_stresses)
    strand = res.strand_stresses[0]

    # check cracked neutral axis
    assert pytest.approx(cr.d_nc, rel=1e-2) == 508.1

    # check cracked stresses
    assert pytest.approx(max_c, rel=1e-3) == 18.0
    assert pytest.approx(min_c, abs=1e-3) == 0.0
    assert pytest.approx(max_steel, rel=1e-3) == 108.1
    assert pytest.approx(min_steel, rel=1e-3) == -45.6
    assert pytest.approx(strand, rel=1e-3) == -1216.0

    # check cracking moment with zero tensile strength
    assert pytest.approx(cr.m_cr[0], rel=1e-3) == 293e6

    # calculate moment curvature
    mk_res = conc_sec.moment_curvature_analysis(kappa_inc=1e-6, delta_m_min=2)

    # check initial curvature
    assert pytest.approx(mk_res.kappa[0], rel=1e-3) == -0.702e-6


def test_example_6_1_2_3():
    """Tests Example 6.1, 6.2 and 6.3."""
    # create materials
    concrete = Concrete(
        name="40 MPa Concrete",
        density=2.4e-6,
        stress_strain_profile=ssp.ConcreteLinearNoTension(
            elastic_modulus=32.75e3,
            ultimate_strain=0.003,
            compressive_strength=40,
        ),
        ultimate_stress_strain_profile=ssp.RectangularStressBlock(
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
        stress_strain_profile=ssp.SteelElasticPlastic(
            yield_strength=500,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="grey",
    )

    strand = SteelStrand(
        name="1910 MPa Strand",
        density=7.85e-6,
        stress_strain_profile=ssp.StrandHardening(
            yield_strength=1770,
            elastic_modulus=195e3,
            fracture_strain=0.035,
            breaking_strength=1910,
        ),
        colour="slategrey",
        prestress_stress=1200,
    )

    # create geometry
    geom = rectangular_section(d=750, b=350, material=concrete)

    # add prestressing strand
    geom = add_bar(
        geometry=geom,
        area=1000,
        material=strand,
        x=175,
        y=100,
    )

    conc_sec = PrestressedSection(geometry=geom)

    # SINGLE TENDON:
    # calculate ultimate bending capacity
    ult_res_1 = conc_sec.ultimate_bending_capacity()

    # check ultimate bending results
    assert pytest.approx(ult_res_1.d_n, rel=1e-2) == 197
    assert pytest.approx(ult_res_1.m_x, rel=1e-2) == 1036e6

    # ADD BOTTOM REINFORCEMENT:
    geom = add_bar_rectangular_array(
        geometry=geom, area=450, material=steel, n_x=3, x_s=115, anchor=(60, 60)
    )

    conc_sec = PrestressedSection(geometry=geom)

    # calculate ultimate bending capacity
    ult_res_2 = conc_sec.ultimate_bending_capacity()

    # check ultimate bending results
    assert pytest.approx(ult_res_2.d_n, rel=1.5e-2) == 264
    assert pytest.approx(ult_res_2.m_x, rel=1.5e-2) == 1353e6

    # ADD TOP REINFORCEMENT:
    geom = add_bar_rectangular_array(
        geometry=geom, area=450, material=steel, n_x=2, x_s=230, anchor=(60, 750 - 60)
    )

    conc_sec = PrestressedSection(geometry=geom)

    # calculate ultimate bending capacity
    ult_res_3 = conc_sec.ultimate_bending_capacity()

    # check ultimate bending results
    assert pytest.approx(ult_res_3.d_n, rel=1.5e-2) == 225
    assert pytest.approx(ult_res_3.m_x, rel=1e-2) == 1422e6


def test_example_6_9():
    """Tests Example 6.9."""
    # create materials
    concrete = Concrete(
        name="40 MPa Concrete",
        density=2.4e-6,
        stress_strain_profile=ssp.ConcreteLinearNoTension(
            elastic_modulus=31.9e3,
            ultimate_strain=0.003,
            compressive_strength=40,
        ),
        ultimate_stress_strain_profile=ssp.RectangularStressBlock(
            compressive_strength=40,
            alpha=0.85,
            gamma=0.77,
            ultimate_strain=0.003,
        ),
        flexural_tensile_strength=3.4,
        colour="lightgrey",
    )

    strand = SteelStrand(
        name="1910 MPa Strand",
        density=7.85e-6,
        stress_strain_profile=ssp.StrandHardening(
            yield_strength=1820,
            elastic_modulus=195e3,
            fracture_strain=0.035,
            breaking_strength=1910,
        ),
        colour="slategrey",
        prestress_stress=1250,
    )

    # create geometry and check gross section properties
    slab = rectangular_section(b=2400, d=50).shift_section(x_offset=-1200, y_offset=-50)
    beam_l = Geometry.from_points(
        points=[(-65, -800), (65, -800), (102.5, -50), (-102.5, -50)],
        facets=[(0, 1), (1, 2), (2, 3), (3, 0)],
        control_points=[(0, -375)],
    ).shift_section(x_offset=-600)
    beam_r = beam_l.shift_section(x_offset=1200)

    geom = slab + beam_l + beam_r
    geom.create_mesh(mesh_sizes=[0, 0, 0])
    gross_sec = Section(geom)
    gross_sec.calculate_geometric_properties()

    assert pytest.approx(gross_sec.get_area(), rel=1e-3) == 371e3
    assert pytest.approx(gross_sec.get_ic()[0], rel=5e-3) == 22.8e9
    assert pytest.approx(gross_sec.get_z()[0], rel=1e-3) == 82.5e6
    assert pytest.approx(gross_sec.get_z()[1], rel=1e-3) == 43.7e6
    assert pytest.approx(gross_sec.get_c()[1], rel=1e-3) == -277

    # recreate geometry with materials
    slab.material = concrete
    beam_l.material = concrete
    beam_r.material = concrete
    geom = slab + beam_l + beam_r

    # add prestressing strands
    geom = add_bar_rectangular_array(
        geometry=geom,
        area=1300,
        material=strand,
        n_x=2,
        x_s=1200,
        anchor=(-600, -685),
    )

    conc_sec = PrestressedSection(geometry=geom)

    # calculate ultimate bending capacity
    ult_res = conc_sec.ultimate_bending_capacity()

    # check ultimate bending results
    assert pytest.approx(ult_res.m_x, rel=1.5e-2) == 3187e6
