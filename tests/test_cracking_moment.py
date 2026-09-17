"""Tests for ConcreteSection.calculate_cracking_moment on composite sections."""

from dataclasses import replace

import pytest
from sectionproperties.pre.library.primitive_sections import rectangular_section

from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.material import Concrete
from concreteproperties.stress_strain_profile import (
    ConcreteLinear,
    RectangularStressBlock,
)


def get_composite_section() -> ConcreteSection:
    """Two-material composite section: a 500x100 topping slab on a 300x500 beam.

    Both materials share the same elastic modulus so the transformed and
    geometric centroids coincide, keeping the analytical check simple.
    """
    beam_material = Concrete(
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

    topping_material = replace(
        beam_material, name="Topping Concrete", flexural_tensile_strength=2.8
    )

    beam = rectangular_section(d=500, b=300, material=beam_material)
    topping = rectangular_section(
        d=100, b=500, material=topping_material
    ).shift_section(x_offset=-100, y_offset=500)

    return ConcreteSection(topping + beam)


def test_cracking_moment_composite_first_geometry_in_compression():
    """The first-added geometry must not zero out the cracking moment.

    Regression test: `calculate_cracking_moment` used `idx == 0` to decide
    whether to initialise the running minimum. If the geometry at index 0
    happened to be entirely in compression for the given `theta` (skipped
    via `continue`), the running minimum was never initialised and the
    function returned 0 for every subsequent geometry (`min(0, m_c_geom)`),
    regardless of the actual tension-side cracking moment.
    """
    conc_sec = get_composite_section()

    # geometry order as added: topping (index 0) is in compression at
    # theta=0, beam (index 1) is in tension -- this is exactly the ordering
    # that triggered the bug.
    assert conc_sec.concrete_geometries[0].material.name == "Topping Concrete"
    assert conc_sec.concrete_geometries[1].material.name == "32 MPa Concrete"

    m_c = conc_sec.calculate_cracking_moment(theta=0)

    # Analytical cracking moment, computed independently of the library:
    # centroid from the bottom of the beam (both materials share E, so the
    # transformed centroid is the plain area-weighted centroid):
    #   y_bar = (A_beam * 250 + A_top * 550) / (A_beam + A_top)
    #         = (150_000 * 250 + 50_000 * 550) / 200_000 = 325 mm
    # second moment of area about the centroid (parallel axis theorem):
    #   I = (300 * 500^3 / 12 + 150_000 * (325 - 250)^2)
    #     + (500 * 100^3 / 12 + 50_000 * (550 - 325)^2)
    #     = 6_541_666_666.67 mm^4
    # distance to the extreme tensile fibre (bottom of the beam) is y_bar:
    #   d = 325 mm
    # m_c = (f_t / E) * (E * I / d) = f_t * I / d
    f_t = 3.4
    i_total = 6_541_666_666.666666
    d = 325.0
    m_c_expected = f_t * i_total / d

    assert m_c == pytest.approx(m_c_expected, rel=1e-6)
    assert m_c > 0
