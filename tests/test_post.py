"""Tests the post methods."""

import math

import pytest

from concreteproperties.post import si_kn_m, si_n_mm, string_formatter


def test_string_formatter():
    """Tests the string formatter."""
    assert string_formatter(value=2704.897111, eng=False, prec=3) == "2704.897"
    assert string_formatter(value=2704.897111, eng=False, prec=0) == "2705"
    assert string_formatter(value=2704.897111, eng=False, prec=10) == "2704.8971110000"
    assert string_formatter(value=0, eng=True, prec=2) == "0.00"
    assert string_formatter(value=2704.897111, eng=True, prec=4) == "2.7049 x 10^3"
    assert string_formatter(value=0.0034563, eng=True, prec=2) == "3.46 x 10^-3"
    assert string_formatter(value=0.034563, eng=True, prec=3) == "34.56 x 10^-3"
    assert string_formatter(value=15, eng=True, prec=2) == "15.0"
    assert string_formatter(value=14435.654, eng=True, prec=4) == "14.436 x 10^3"
    assert (
        string_formatter(value=14435.654, eng=True, prec=4, scale=10) == "144.36 x 10^3"
    )
    assert (
        string_formatter(value=14435.654, eng=True, prec=3, scale=1000)
        == "14.44 x 10^6"
    )
    assert string_formatter(value=14435.654, eng=True, prec=3, scale=1e-3) == "14.44"


def test_unit_display():
    """Tests the unit display class."""
    assert si_n_mm.length_unit == " mm"
    assert si_kn_m.length_unit == " m"
    assert si_n_mm.length_scale == pytest.approx(1)
    assert si_kn_m.length_scale == pytest.approx(0.001)
    assert si_n_mm.force_unit == " N"
    assert si_kn_m.force_unit == " kN"
    assert si_n_mm.force_scale == pytest.approx(1)
    assert si_kn_m.force_scale == pytest.approx(0.001)
    assert si_n_mm.mass_unit == " kg"
    assert si_kn_m.mass_unit == " kg"
    assert si_n_mm.mass_scale == pytest.approx(1)
    assert si_kn_m.mass_scale == pytest.approx(1)
    assert si_n_mm.angle_unit == " rads"
    assert si_kn_m.angle_unit == " rads"
    assert si_n_mm.angle_scale == pytest.approx(1)
    assert si_kn_m.angle_scale == pytest.approx(1)
    si_n_mm.radians = False
    assert si_n_mm.angle_unit == " degs"
    assert si_n_mm.angle_scale == pytest.approx(180 / math.pi)
    assert si_n_mm.area_unit == " mm^2"
    assert si_kn_m.area_unit == " m^2"
    assert si_n_mm.area_scale == pytest.approx(1)
    assert si_kn_m.area_scale == pytest.approx(1e-6)
    assert si_n_mm.mass_per_length_unit == " kg/mm"
    assert si_kn_m.mass_per_length_unit == " kg/m"
    assert si_n_mm.mass_per_length_scale == pytest.approx(1)
    assert si_kn_m.mass_per_length_scale == pytest.approx(1e3)
    assert si_n_mm.moment_unit == " N.mm"
    assert si_kn_m.moment_unit == " kN.m"
    assert si_n_mm.moment_scale == pytest.approx(1)
    assert si_kn_m.moment_scale == pytest.approx(1e-6)
    assert si_n_mm.flex_rig_unit == " N.mm^2"
    assert si_kn_m.flex_rig_unit == " kN.m^2"
    assert si_n_mm.flex_rig_scale == pytest.approx(1)
    assert si_kn_m.flex_rig_scale == pytest.approx(1e-9)
    assert si_n_mm.stress_unit == " MPa"
    assert si_kn_m.stress_unit == " kPa"
    assert si_n_mm.stress_scale == pytest.approx(1)
    assert si_kn_m.stress_scale == pytest.approx(1e3)
    si_n_mm.length = "um"
    assert si_n_mm.stress_unit == " N/um^2"
    si_n_mm.length = "mm"
    assert si_n_mm.length_3_unit == " mm^3"
    assert si_kn_m.length_3_unit == " m^3"
    assert si_n_mm.length_3_scale == pytest.approx(1)
    assert si_kn_m.length_3_scale == pytest.approx(1e-9)
    assert si_n_mm.length_4_unit == " mm^4"
    assert si_kn_m.length_4_unit == " m^4"
    assert si_n_mm.length_4_scale == pytest.approx(1)
    assert si_kn_m.length_4_scale == pytest.approx(1e-12)
