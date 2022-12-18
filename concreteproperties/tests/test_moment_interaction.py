import concreteproperties.results as res
import concreteproperties.utils as utils
import numpy as np
import pytest
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
        ("D", 0.8),
        ("d_n", 310),
        ("d_n", 200),
        ("N", 1000.0),
        ("N", 0.0),
    ]

    n_points = 10

    mi_res = conc_sec.moment_interaction_diagram(
        control_points=control_points, n_points=n_points
    )

    d_n_list = []
    n_list = []

    for ult_res in mi_res.results:
        d_n_list.append(ult_res.d_n)
        n_list.append(ult_res.n)

    d_n_list = np.array(d_n_list).astype(int)
    n_list = np.array(n_list).astype(int)

    # check neutral axes in list
    assert int(D * 0.8) in d_n_list
    assert 310 in d_n_list
    assert 200 in d_n_list
    assert int(conc_sec.ultimate_bending_capacity(n=1000).n) in n_list
    assert int(conc_sec.ultimate_bending_capacity(n=0).n) in n_list

    # check length of list
    assert len(d_n_list) == n_points + len(control_points)


limits_list = [
    [("kappa0", 0), ("d_n", 1e-6)],
    [("D", 1), ("d_n", 1e-6)],
    [("D", 0.85), ("N", -1e3)],
]


@pytest.mark.parametrize("limits", limits_list)
def test_limits(limits):
    theta = 0

    # compute extreme tensile fibre
    _, d_t = utils.calculate_extreme_fibre(
        points=conc_sec.compound_geometry.points, theta=theta
    )

    limit_results = []

    # get results for limits
    for lim in limits:
        limit_results.append(
            conc_sec.calculate_ultimate_section_actions(
                d_n=conc_sec.decode_d_n(
                    theta=0,
                    cp=lim,
                    d_t=d_t,
                ),
                ultimate_results=res.UltimateBendingResults(theta=theta),
            )
        )

    mi_res = conc_sec.moment_interaction_diagram(limits=limits)

    for lim_res in limit_results:
        assert lim_res in mi_res.results


def test_sorting_and_duplicates():
    n_points = 10

    limits = [("kappa0", 0), ("d_n", 1e-6)]
    control_points = [
        ("D", 1),
        ("d_n", 310),
        ("d_n", 310),
    ]

    mi_res = conc_sec.moment_interaction_diagram(
        limits=limits, control_points=control_points, n_points=n_points
    )

    # testing sorting
    for idx, res in enumerate(mi_res.results):
        if idx > 0:
            assert res.n < mi_res.results[idx - 1].n

    # 1 duplicate therefore length should be n_points + len(control_points) - 1
    assert len(mi_res.results) == n_points + len(control_points) - 1


def test_limit_validation():
    with pytest.raises(ValueError):
        mi_res = conc_sec.moment_interaction_diagram(
            limits=[("D", 1)],
        )

    with pytest.raises(ValueError):
        mi_res = conc_sec.moment_interaction_diagram(
            limits=[
                ("D", 1),
                ("D", 0.5),
                ("N", 0),
            ],
        )


def test_label_validation():
    mi_res = conc_sec.moment_interaction_diagram(labels=["Hi"])
    mi_res = conc_sec.moment_interaction_diagram(
        labels=["Hi", "Bye"], control_points=[]
    )
    mi_res = conc_sec.moment_interaction_diagram(labels=["A", "B", "C", "D", "E"])
    mi_res = conc_sec.moment_interaction_diagram(
        control_points=[
            ("D", 1),
            ("d_n", 310),
            ("d_n", 310),
        ],
        labels=["Hi"],
    )

    with pytest.raises(ValueError):
        mi_res = conc_sec.moment_interaction_diagram(
            control_points=[
                ("D", 1),
                ("d_n", 310),
                ("d_n", 310),
            ],
            labels=["Hi", "Bye"],
        )

    with pytest.raises(ValueError):
        mi_res = conc_sec.moment_interaction_diagram(
            labels="Hi",  # type: ignore
        )


def test_n_spacing():
    n_spacing = 24
    mi_res = conc_sec.moment_interaction_diagram(n_spacing=n_spacing, control_points=[])

    spacing = 0

    for idx, res in enumerate(mi_res.results):
        if idx == 1:
            spacing = mi_res.results[idx - 1].n - res.n
        elif idx > 1:
            assert pytest.approx(mi_res.results[idx - 1].n - res.n, rel=1e-3) == spacing


def test_max_comp():
    mc = 4000e3  # N.B point chosen to be between first two points on MI diagram
    mi_res_ref = conc_sec.moment_interaction_diagram()
    mi_res_mc = conc_sec.moment_interaction_diagram(max_comp=mc)

    pt1 = mi_res_ref.results[0]
    pt2 = mi_res_ref.results[1]

    # check first point
    assert mi_res_mc.results[0].n == mc
    assert mi_res_mc.results[0].m_x == 0
    assert mi_res_mc.results[0].m_y == 0
    assert mi_res_mc.results[0].m_xy == 0

    # check second point
    factor = (mc - pt2.n) / (pt1.n - pt2.n)

    mx = pt2.m_x + factor * (pt1.m_x - pt2.m_x)
    my = pt2.m_y + factor * (pt1.m_y - pt2.m_y)
    mxy = pt2.m_xy + factor * (pt1.m_xy - pt2.m_xy)

    assert pytest.approx(mi_res_mc.results[1].n) == mc
    # check moment with interpolated values to 3%
    assert pytest.approx(mi_res_mc.results[1].m_x, rel=3e-2) == mx
    assert pytest.approx(mi_res_mc.results[1].m_y, abs=1, rel=3e-2) == my
    assert pytest.approx(mi_res_mc.results[1].m_xy, rel=3e-2) == mxy

    # test max_comp larger than squash load
    with pytest.raises(ValueError):
        mi_res = conc_sec.moment_interaction_diagram(max_comp=6000e3)

    # test control point above max_comp
    mi_res = conc_sec.moment_interaction_diagram(
        control_points=[("N", 4500e3)], max_comp=mc
    )

    # check max axial load (with buffer) is less than or equal to mc
    for res in mi_res.results:
        assert res.n <= (1 + 1e-6) * mc


def test_labels():
    axial_load_list = [0, 1e6, 2e6, 3e6, 4e6]
    limits = [
        ("N", 4000e3),
        ("N", 0.0),
    ]
    control_points = [
        ("N", 3000e3),
        ("N", 2000e3),
        ("N", 1000e3),
    ]

    mi_res = conc_sec.moment_interaction_diagram(
        limits=limits,
        control_points=control_points,
        labels=["HI"],
    )

    # check labels applied to limits/control points
    for res in mi_res.results:
        if any(np.isclose([res.n] * len(axial_load_list), axial_load_list, atol=50)):
            assert res.label == "HI"
        else:
            assert res.label is None

    labels = ["A", "E", "B", "C", "D"]
    labels_ordered = ["A", "B", "C", "D", "E"]
    mi_res = conc_sec.moment_interaction_diagram(
        limits=limits,
        control_points=control_points,
        labels=labels,
    )

    label_counter = 0

    # check labels applied to limits/control points
    for res in mi_res.results:
        if any(np.isclose([res.n] * len(axial_load_list), axial_load_list, atol=50)):
            assert res.label == labels_ordered[label_counter]
            label_counter += 1
        else:
            assert res.label is None

    # check max_comp_labels pt 1
    mc = 4000e3  # N.B point chosen to be between first two points on MI diagram
    mi_res_mc = conc_sec.moment_interaction_diagram(
        max_comp=mc, max_comp_labels=["A", "B"]
    )

    assert mi_res_mc.results[0].label == "A"
    assert mi_res_mc.results[1].label == "B"
    assert mi_res_mc.results[2].label is None
