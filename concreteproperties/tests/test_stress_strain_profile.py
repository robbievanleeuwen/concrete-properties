import pytest
from concreteproperties.stress_strain_profile import *


def test_whitney():
    profile = WhitneyStressBlock(0.85, 0.77, 40, 0.003)

    assert pytest.approx(profile.get_stress(0)) == 0
    assert pytest.approx(profile.get_stress(0.003)) == 0.85 * 40
    assert pytest.approx(profile.get_stress(0.1)) == 0.85 * 40
    assert pytest.approx(profile.get_stress(-0.001)) == 0
    assert pytest.approx(profile.get_stress(-0.003)) == 0
    assert pytest.approx(profile.get_stress(-0.1)) == 0
    assert pytest.approx(profile.get_stress(0.00069)) == 0
    assert pytest.approx(profile.get_stress(0.0007)) == 0.85 * 40
    assert pytest.approx(profile.get_stress(0.001)) == 0.85 * 40


def test_piecewise_linear():
    with pytest.raises(ValueError):
        profile = PiecewiseLinearProfile([0], [0])

    with pytest.raises(ValueError):
        profile = PiecewiseLinearProfile([-1, 0, 1], [1, 2, 3])

    with pytest.raises(ValueError):
        profile = PiecewiseLinearProfile([-1, 0, 1], [0, 2])

    with pytest.raises(ValueError):
        profile = PiecewiseLinearProfile([0, 1, 0.5], [0, 3, 5])

    with pytest.raises(ValueError):
        profile = PiecewiseLinearProfile([0, 1, 2], [0, 3, 2])

    profile = PiecewiseLinearProfile([0, 0.0025, 0.05], [0, 500, 600])

    assert pytest.approx(profile.get_stress(0)) == 0
    assert pytest.approx(profile.get_stress(0.0025)) == 500
    assert pytest.approx(profile.get_stress(0.05)) == 600
    assert pytest.approx(profile.get_stress(0.001)) == 200
    assert pytest.approx(profile.get_stress(0.01)) == 515.789
    assert pytest.approx(profile.get_stress(0.1)) == 705.263
    assert pytest.approx(profile.get_stress(-0.0025)) == -500
    assert pytest.approx(profile.get_stress(-0.05)) == -600
    assert pytest.approx(profile.get_stress(-0.001)) == -200
    assert pytest.approx(profile.get_stress(-0.01)) == -515.789
    assert pytest.approx(profile.get_stress(-0.1)) == -705.263


def test_bilinear():
    with pytest.raises(ValueError):
        profile = BilinearProfile(0, 0, 0, 0)

    with pytest.raises(ValueError):
        profile = BilinearProfile(0, -1, 1, 2)

    with pytest.raises(ValueError):
        profile = BilinearProfile(1, 0.5, 3, 5)

    with pytest.raises(ValueError):
        profile = BilinearProfile(1, 2, 3, 2)

    profile = BilinearProfile(0.0025, 0.05, 500, 600)

    assert pytest.approx(profile.get_stress(0)) == 0
    assert pytest.approx(profile.get_stress(0.0025)) == 500
    assert pytest.approx(profile.get_stress(0.05)) == 600
    assert pytest.approx(profile.get_stress(0.001)) == 200
    assert pytest.approx(profile.get_stress(0.01)) == 515.789
    assert pytest.approx(profile.get_stress(0.1)) == 705.263
    assert pytest.approx(profile.get_stress(-0.0025)) == -500
    assert pytest.approx(profile.get_stress(-0.05)) == -600
    assert pytest.approx(profile.get_stress(-0.001)) == -200
    assert pytest.approx(profile.get_stress(-0.01)) == -515.789
    assert pytest.approx(profile.get_stress(-0.1)) == -705.263
