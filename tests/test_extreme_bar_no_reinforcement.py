"""Tests for extreme_bar()/moment_interaction_diagram() on unreinforced sections."""

import pytest
from sectionproperties.pre.geometry import CompoundGeometry
from sectionproperties.pre.library.primitive_sections import rectangular_section

from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.material import Concrete
from concreteproperties.stress_strain_profile import (
    ConcreteLinear,
    RectangularStressBlock,
)


def get_plain_concrete_section() -> ConcreteSection:
    """A plain (unreinforced) rectangular concrete section."""
    material = Concrete(
        name="32 MPa Concrete",
        density=2.4e-6,
        stress_strain_profile=ConcreteLinear(elastic_modulus=30e3),
        ultimate_stress_strain_profile=RectangularStressBlock(
            compressive_strength=32,
            alpha=0.85,
            gamma=0.83,
            ultimate_strain=0.003,
        ),
        flexural_tensile_strength=3.4,
        colour="lightgrey",
    )

    geom = CompoundGeometry([rectangular_section(d=300, b=400, material=material)])

    return ConcreteSection(geom)


def test_extreme_bar_raises_clear_error_without_reinforcement():
    """Regression test: extreme_bar() used to raise an unguarded IndexError
    (list index out of range) on a section with no lumped reinforcement. It
    should raise a descriptive ValueError instead.
    """
    conc_sec = get_plain_concrete_section()
    assert conc_sec.reinf_geometries_lumped == []

    with pytest.raises(ValueError, match="requires at least one lumped"):
        conc_sec.extreme_bar(theta=0)


def test_moment_interaction_diagram_default_control_points_raise_clear_error():
    """moment_interaction_diagram()'s default control points include an 'fy'
    point, which is not meaningful without reinforcement. This should
    surface as the same descriptive ValueError, not an IndexError.
    """
    conc_sec = get_plain_concrete_section()

    with pytest.raises(ValueError, match="requires at least one lumped"):
        conc_sec.moment_interaction_diagram(progress_bar=False, n_points=4)
