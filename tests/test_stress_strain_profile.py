"""Tests for stress-strain profiles."""

import pytest

import concreteproperties.stress_strain_profile as ssp


def test_whitney():
    """Tests the Whitney stress block."""
    profile = ssp.RectangularStressBlock(40, 0.85, 0.77, 0.003)

    assert pytest.approx(profile.get_stress(0)) == 0
    assert pytest.approx(profile.get_stress(0.003)) == 0.85 * 40
    assert pytest.approx(profile.get_stress(0.1)) == 0.85 * 40
    assert pytest.approx(profile.get_stress(-0.001)) == 0
    assert pytest.approx(profile.get_stress(-0.003)) == 0
    assert pytest.approx(profile.get_stress(-0.1)) == 0
    assert pytest.approx(profile.get_stress(0.00068998)) == 0
    assert pytest.approx(profile.get_stress(0.00069)) == 0.85 * 40
    assert pytest.approx(profile.get_stress(0.001)) == 0.85 * 40


def test_piecewise_linear():
    """Tests the piecewise linear profile."""
    with pytest.raises(ValueError, match="must be greater than 1"):
        profile = ssp.StressStrainProfile([0], [0])

    with pytest.raises(ValueError, match="Length of strains must equal length of"):
        profile = ssp.StressStrainProfile([-1, 0, 1], [0, 2])

    with pytest.raises(ValueError, match="Strains must contain increasing or equal"):
        profile = ssp.StressStrainProfile([0, 1, 0.5], [0, 3, 5])

    profile = ssp.StressStrainProfile([-0.05, 0, 0.0025, 0.05], [0, 0, 500, 600])

    assert pytest.approx(profile.get_stress(0)) == 0
    assert pytest.approx(profile.get_stress(0.0025)) == 500
    assert pytest.approx(profile.get_stress(0.05)) == 600
    assert pytest.approx(profile.get_stress(0.001)) == 200
    assert pytest.approx(profile.get_stress(0.01)) == 515.789
    assert pytest.approx(profile.get_stress(0.1)) == 705.263
    assert pytest.approx(profile.get_stress(-0.0025)) == 0


def test_modifiedmander_invalid_sect_type():
    """Tests the modified mander profile (invalid section type)."""
    with pytest.raises(ValueError, match="The specified section type"):
        ssp.ModifiedMander(
            elastic_modulus=30e3,
            compressive_strength=30,
            tensile_strength=4.5,
            sect_type="this_is_an_incorrect_sect_type",
            conc_confined=True,
        )


def test_modifiedmander_confined_warning():
    """Tests the modified mander profile (confined warning)."""
    with pytest.warns(UserWarning):
        ssp.ModifiedMander(
            elastic_modulus=30e3,
            compressive_strength=30,
            tensile_strength=4.5,
            sect_type="rect",
            conc_confined=True,
        )


@pytest.mark.parametrize(
    (
        "elastic_modulus",
        "compressive_strength",
        "tensile_strength",
        "stress_index",
        "stress",
        "strain_index",
        "strain",
    ),
    [
        (30e3, 30, 4.5, 27, 29.9936, 45, 0.00334694),
        (40e3, 40, 5.5, 34, 39.1935, 22, 0.00155102),
        (25e3, 25, 2.5, 11, 14.7535, 35, 0.00253061),
    ],
)
def test_modifiedmander_unconfined_stress_strain(
    elastic_modulus,
    compressive_strength,
    tensile_strength,
    stress_index,
    stress,
    strain_index,
    strain,
):
    """Tests the modified mander profile (unconfined stress-strain)."""
    stress_strain_profile = ssp.ModifiedMander(
        elastic_modulus=elastic_modulus,
        compressive_strength=compressive_strength,
        tensile_strength=tensile_strength,
        conc_tension=True,
        conc_spalling=True,
    )

    assert pytest.approx(max(stress_strain_profile.stresses)) == compressive_strength
    assert pytest.approx(min(stress_strain_profile.stresses)) == -tensile_strength
    assert (
        pytest.approx(stress_strain_profile.strains[1])
        == -tensile_strength / elastic_modulus
    )
    assert (
        pytest.approx(stress_strain_profile.stresses[stress_index], rel=0.001) == stress
    )
    assert (
        pytest.approx(stress_strain_profile.strains[strain_index], rel=0.000001)
        == strain
    )


@pytest.mark.parametrize(
    (
        "sect_type",
        "elastic_modulus",
        "compressive_strength",
        "tensile_strength",
        "max_fc",
        "stress_index",
        "stress",
        "strain_index",
        "strain",
    ),
    [
        ("rect", 30e3, 30, 4.5, 37.5859, 27, 37.0829, 41, 0.00957687),
        ("circ_hoop", 40e3, 40, 5.5, 47.3521, 22, 45.4338, 25, 0.00292869),
        ("circ_spiral", 25e3, 25, 2.5, 32.6504, 36, 32.6504, 13, 0.00156307),
    ],
)
def test_modifiedmander_confined_stress_strain(
    sect_type,
    elastic_modulus,
    compressive_strength,
    tensile_strength,
    max_fc,
    stress_index,
    stress,
    strain_index,
    strain,
):
    """Tests the modified mander profile (confined stress-strain)."""
    stress_strain_profile = ssp.ModifiedMander(
        elastic_modulus=elastic_modulus,
        compressive_strength=compressive_strength,
        tensile_strength=tensile_strength,
        sect_type=sect_type,
        conc_confined=True,
        conc_tension=True,
        d=800,
        b=500,
        long_reinf_area=12 * 314,
        w_dash=[150.0] * 12,
        cvr=30 + 10,
        trans_spacing=125,
        trans_d_b=10,
        trans_num_d=4,
        trans_num_b=4,
        trans_f_y=500,
        eps_su=0.15,
    )

    assert pytest.approx(max(stress_strain_profile.stresses), rel=0.001) == max_fc
    assert pytest.approx(min(stress_strain_profile.stresses)) == -tensile_strength
    assert (
        pytest.approx(stress_strain_profile.strains[1])
        == -tensile_strength / elastic_modulus
    )
    assert (
        pytest.approx(stress_strain_profile.stresses[stress_index], rel=0.001) == stress
    )
    assert (
        pytest.approx(stress_strain_profile.strains[strain_index], rel=0.000001)
        == strain
    )
