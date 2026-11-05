"""Tests for prestressed concrete sections."""

import numpy as np
import pytest
from sectionproperties.pre.library.primitive_sections import rectangular_section

import concreteproperties.stress_strain_profile as ssp
import concreteproperties.utils as utils
from concreteproperties.material import Concrete, Steel, SteelStrand
from concreteproperties.pre import add_bar
from concreteproperties.prestressed_section import PrestressedSection

# material properties
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
    prestress_stress=1000,
)


def test_asymmetric_section():
    """Tests asymmetric section detection."""
    geom = rectangular_section(b=100, d=100, material=concrete)
    geom = add_bar(
        geometry=geom,
        area=100,
        material=strand,
        x=40,
        y=50,
    )

    with pytest.raises(
        ValueError, match=r"PrestressedSection must be symmetric about y-axis."
    ):
        PrestressedSection(geom)


def test_meshed_sections():
    """Tests meshed steel section detection."""
    steel = Steel(
        name="500 MPa Steel",
        density=7.85e-6,
        stress_strain_profile=ssp.SteelElasticPlastic(
            yield_strength=500,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="grey",
    )

    geom = rectangular_section(b=100, d=100, material=concrete)  #
    geom = add_bar(
        geometry=geom,
        area=100,
        material=strand,
        x=50,
        y=50,
    )
    geom = add_bar(
        geometry=geom,
        area=100,
        material=steel,
        x=50,
        y=80,
    )

    msg = "Meshed reinforcement geometries are not permitted in PrestressedSection."
    with pytest.raises(ValueError, match=msg):
        PrestressedSection(geom)


# define section to be used in cracked analysis
geom_bot = rectangular_section(d=400, b=300, material=concrete)
geom_top = rectangular_section(d=400, b=300, material=concrete).shift_section(
    y_offset=400
)
geom = geom_top + geom_bot
geom = add_bar(
    geometry=geom,
    area=1000,
    material=strand,
    x=150,
    y=80,
)
conc_sec = PrestressedSection(geom)


def test_cracked_full_compression():
    """Tests cracked analysis when section in compression."""
    with pytest.raises(utils.AnalysisError):
        conc_sec.calculate_cracked_properties(m_ext=300e6)


def test_cracked_multiple_sections():
    """Tests cracked analysis on a combined section is equal to a single section."""
    # calculate with two separate sections
    cr = conc_sec.calculate_cracked_properties(m_ext=500e6)

    # calculate with one section
    geom2 = rectangular_section(d=800, b=300, material=concrete)
    geom2 = add_bar(
        geometry=geom2,
        area=1000,
        material=strand,
        x=150,
        y=80,
    )
    conc_sec2 = PrestressedSection(geom2)
    cr2 = conc_sec2.calculate_cracked_properties(m_ext=500e6)

    if isinstance(cr.m_cr, tuple) and isinstance(cr2.m_cr, tuple):
        assert pytest.approx(cr.m_cr[0]) == cr2.m_cr[0]
        assert pytest.approx(cr.m_cr[1]) == cr2.m_cr[1]


def test_moment_interaction():
    """Tests NotImplementedError for moment interaction diagram."""
    with pytest.raises(NotImplementedError):
        conc_sec.moment_interaction_diagram()


def test_biaxial_bending():
    """Tests NotImplementedError for biaxial bending diagram."""
    with pytest.raises(NotImplementedError):
        conc_sec.biaxial_bending_diagram()


def _service_stress_at_extremes(positive: bool) -> tuple[float, float]:
    """Returns the mean concrete stress of the lowest and the highest section piece."""
    mk = conc_sec.moment_curvature_analysis(positive=positive, progress_bar=False)
    stress = conc_sec.calculate_service_stress(moment_curvature_results=mk, m=100e6)

    pieces = sorted(
        (sec.geometry.calculate_centroid()[1], float(np.mean(sig)))
        for sec, sig in zip(
            stress.concrete_analysis_sections, stress.concrete_stresses, strict=True
        )
    )

    return pieces[0][1], pieces[-1][1]


def test_service_stress_negative_moment():
    """Tests the service stresses follow the bending direction of the analysis.

    The strand sits below the centroid, so a negative (hogging) moment adds to the
    eccentric prestress, compressing the bottom and tensioning the top more than the
    same moment applied as a positive (sagging) moment.
    """
    bot_pos, top_pos = _service_stress_at_extremes(positive=True)
    bot_neg, top_neg = _service_stress_at_extremes(positive=False)

    assert bot_neg > bot_pos
    assert top_neg < top_pos
