"""Regression test for degenerate slivers produced by CPGeom.split_section.

See https://github.com/robbievanleeuwen/concrete-properties/issues/153. For
some load angles/curvatures the neutral axis passes so close to an existing
vertex that, after coordinate rounding, the resulting sliver collapses to
fewer than three distinct points or zero area. Passing such a sliver on to
``AnalysisSection`` used to raise a ``ValueError`` from the triangulation
library instead of being silently discarded.
"""

import math

import numpy as np
import pytest
import sectionproperties.pre.library.concrete_sections as sp_cs
from sectionproperties.pre.library.primitive_sections import rectangular_section
from shapely import Polygon

from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.material import Concrete, SteelBar
from concreteproperties.pre import CPGeom
from concreteproperties.stress_strain_profile import (
    BilinearStressStrain,
    ConcreteLinearNoTension,
    EurocodeNonLinear,
    EurocodeParabolicUltimate,
    SteelElasticPlastic,
    SteelProfile,
)


def test_zero_area_geometry_with_three_distinct_points_is_degenerate():
    """Discard a rounded polygon with distinct vertices but no area."""
    # Rounding can also flatten a thin triangle without merging its vertices.
    geom = CPGeom(
        geom=Polygon([(0, 0), (1, 1e-7), (2, 0)]),
        material=None,  # type: ignore[arg-type]
    )

    assert len(set(geom.points)) == 3
    assert geom.is_degenerate()
    assert geom.split_section(point=(0, 1), theta=0) == ([], [])


def test_moment_curvature_analysis_does_not_crash_on_degenerate_split():
    """Complete the reported load case without a triangulation error."""
    fck = 30
    fy = 500
    width = 250
    depth = 250

    elastic_modulus = 5600 * fck**0.5

    concrete = Concrete(
        name="Concrete",
        density=2.5e-6,
        stress_strain_profile=EurocodeNonLinear(
            elastic_modulus=elastic_modulus,
            ultimate_strain=0.0035,
            compressive_strength=fck,
            compressive_strain=0.002,
            tensile_strength=1e-3,
            tension_softening_stiffness=1,
        ),
        ultimate_stress_strain_profile=EurocodeParabolicUltimate(
            compressive_strength=fck,
            compressive_strain=0.002,
            ultimate_strain=0.0035,
            n=2,
        ),
        flexural_tensile_strength=0,
        colour="grey",
    )

    steel = SteelBar(
        name="Bar",
        density=7.85e-6,
        stress_strain_profile=SteelProfile(
            strains=[-0.05, -0.03, -0.02, -fy / 200e3, 0, 1],
            stresses=[-fy, -fy, -fy, -fy, 0, 100],
            yield_strength=fy,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="orange",
    )

    geom = rectangular_section(d=width, b=depth, material=concrete)
    for dia, x, y in [
        (22.225, 35, 35),
        (22.225, 215, 215),
        (22.225, 35, 215),
        (22.225, 215, 35),
    ]:
        geom = sp_cs.add_bar(
            geometry=geom,
            area=math.pi * dia**2 * 0.25,
            material=steel,
            x=x,
            y=y,
            n=6,
        )

    conc_section = ConcreteSection(geom)

    inertia = conc_section.get_transformed_gross_properties(
        elastic_modulus=elastic_modulus
    )

    # angle/curvature combination that used to hit a degenerate split geometry
    mx, mz = -25e6, 1e6
    kappa_x = mx * inertia.ixx_c + mz * inertia.ixy_c
    kappa_y = mz * inertia.iyy_c + mx * inertia.ixy_c
    theta = np.arctan2(kappa_y, kappa_x)

    # should complete without raising
    conc_section.moment_curvature_analysis(theta=theta, n=0, progress_bar=False)


def test_vertices_merged_by_rounding_are_removed():
    """Drop a vertex that becomes identical to its neighbour after rounding.

    See https://github.com/robbievanleeuwen/concrete-properties/issues/155. The
    repeated vertex gave a zero length facet that crashed the triangulation.
    """
    geom = CPGeom(
        geom=Polygon([(300, 1e-6), (300, 0), (0, 1e-9), (0, 0), (0, 1e-6)]),
        material=None,  # type: ignore[arg-type]
    )

    assert len(geom.points) == len(set(geom.points)) == 4
    assert not geom.is_degenerate()


def test_closing_and_hole_vertices_merged_by_rounding_are_removed():
    """Also drop a last vertex that rounds onto the first, and repeats in holes."""
    closing = CPGeom(
        geom=Polygon([(0, 0), (300, 0), (300, 300), (0, 300), (1e-9, 0)]),
        material=None,  # type: ignore[arg-type]
    )

    assert len(closing.points) == len(set(closing.points)) == 4
    assert len(closing.facets) == 4

    with_hole = CPGeom(
        geom=Polygon(
            [(0, 0), (300, 0), (300, 300), (0, 300)],
            [[(100, 100), (200, 100), (200, 200), (200, 200 + 1e-9), (100, 200)]],
        ),
        material=None,  # type: ignore[arg-type]
    )

    assert len(with_hole.points) == len(set(with_hole.points)) == 8
    assert len(with_hole.facets) == 8


def test_sliver_collapsed_by_rounding_is_degenerate():
    """A sliver whose vertices all round to one point is still discarded."""
    geom = CPGeom(
        geom=Polygon([(10, 10), (10 + 1e-9, 10), (10, 10 + 1e-9)]),
        material=None,  # type: ignore[arg-type]
    )

    assert geom.points == [(10, 10)]
    assert geom.is_degenerate()


def test_moment_interaction_diagram_theta_pi():
    """theta=pi gives the same diagram as theta=-pi for the reported section."""
    concrete = Concrete(
        name="C30/37",
        density=2.5e-6,
        stress_strain_profile=ConcreteLinearNoTension(
            elastic_modulus=32.84e3,
            ultimate_strain=0.0035,
            compressive_strength=38,
        ),
        ultimate_stress_strain_profile=BilinearStressStrain(
            compressive_strength=20,
            compressive_strain=0.00175,
            ultimate_strain=0.0035,
        ),
        flexural_tensile_strength=2.9,
        colour="lightgrey",
    )

    steel = SteelBar(
        name="B500B",
        density=7.85e-6,
        stress_strain_profile=SteelElasticPlastic(
            yield_strength=500 / 1.15,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="grey",
    )

    geom = sp_cs.concrete_rectangular_section(
        d=900,
        b=300,
        dia_top=16,
        area_top=np.pi * 8**2,
        n_top=3,
        c_top=42,
        dia_bot=16,
        area_bot=np.pi * 8**2,
        n_bot=3,
        c_bot=42,
        conc_mat=concrete,
        steel_mat=steel,
    )
    conc_section = ConcreteSection(geom)

    res_pos = conc_section.moment_interaction_diagram(
        theta=np.pi, n_points=12, progress_bar=False
    )
    res_neg = conc_section.moment_interaction_diagram(
        theta=-np.pi, n_points=12, progress_bar=False
    )

    for r_pos, r_neg in zip(res_pos.results, res_neg.results, strict=True):
        assert r_pos.n == pytest.approx(r_neg.n)
        assert r_pos.m_xy == pytest.approx(r_neg.m_xy)
