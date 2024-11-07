"""Tests the post methods."""

from concreteproperties.post import string_formatter


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


# TODO: unit display tests
