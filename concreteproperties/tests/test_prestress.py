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
    prestress_force=1000e3,
)


def test_asymmetric_section():
    geom = rectangular_section(b=100, d=100, material=concrete)  # type: ignore
    geom = add_bar(
        geometry=geom,
        area=100,
        material=strand,
        x=40,
        y=50,
    )

    with pytest.raises(
        ValueError, match="PrestressedSection must be symmetric about y-axis."
    ):
        conc_sec = PrestressedSection(geom)


def test_meshed_sections():
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

    geom = rectangular_section(b=100, d=100, material=concrete)  # type: ignore
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
        material=steel,  # type: ignore
        x=50,
        y=80,
    )

    with pytest.raises(
        ValueError,
        match="Meshed reinforcement geometries are not permitted in PrestressedSection.",
    ):
        conc_sec = PrestressedSection(geom)


# define section to be used in cracked analysis
geom_bot = rectangular_section(d=400, b=300, material=concrete)  # type: ignore
geom_top = rectangular_section(
    d=400, b=300, material=concrete  # type: ignore
).shift_section(y_offset=400)
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
    with pytest.raises(utils.AnalysisError):
        conc_sec.calculate_cracked_properties(m_ext=300e6)


def test_cracked_multiple_sections():
    # calculate with two separate sections
    cr = conc_sec.calculate_cracked_properties(m_ext=500e6)

    # calculate with one section
    geom2 = rectangular_section(d=800, b=300, material=concrete)  # type: ignore
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
    with pytest.raises(NotImplementedError):
        conc_sec.moment_interaction_diagram()


def test_biaxial_bending():
    with pytest.raises(NotImplementedError):
        conc_sec.biaxial_bending_diagram()
