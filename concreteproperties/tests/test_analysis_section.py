import pytest

from concreteproperties.analysis_section import *

import sectionproperties.pre.library.primitive_sections as sp_ps


def test_second_moment_of_area():
    d = 600
    b = 100

    rect = sp_ps.rectangular_section(d=d, b=b)
    sec = AnalysisSection(geometry=rect, order=1)

    ixx_g = 0
    iyy_g = 0
    ixy_g = 0

    for el in sec.elements:
        el_ixx_g, el_iyy_g, el_ixy_g = el.second_moments_of_area()

        ixx_g += el_ixx_g
        iyy_g += el_iyy_g
        ixy_g += el_ixy_g

    ixx_g_exp = b * d * d * d / 3
    iyy_g_exp = d * b * b * b / 3

    print(f"Ixx_g: calc = {ixx_g}; expected = {ixx_g_exp}")
    print(f"Iyy_g: calc = {iyy_g}; expected = {iyy_g_exp}")


if __name__ == "__main__":
    test_second_moment_of_area()
