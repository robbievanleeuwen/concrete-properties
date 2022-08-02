import pytest
import sectionproperties.pre.library.primitive_sections as sp_ps
from concreteproperties.analysis_section import AnalysisSection


def test_rectangle_second_moment_of_area():
    d = 100
    b = 100

    rect = sp_ps.rectangular_section(d=d, b=b)
    sec = AnalysisSection(geometry=rect)  # type: ignore

    ixx_g = 0
    iyy_g = 0
    ixy_g = 0

    area = rect.calculate_area()
    centroid = rect.calculate_centroid()
    qx = area * centroid[1]
    qy = area * centroid[0]

    for el in sec.elements:
        el_ixx_g, el_iyy_g, el_ixy_g = el.second_moments_of_area()

        ixx_g += el_ixx_g
        iyy_g += el_iyy_g
        ixy_g += el_ixy_g

    assert pytest.approx(ixx_g) == b * d * d * d / 3
    assert pytest.approx(iyy_g) == d * b * b * b / 3

    ixx_c = ixx_g - qx**2 / area
    iyy_c = iyy_g - qy**2 / area
    ixy_c = ixy_g - qx * qy / area

    assert pytest.approx(ixx_c) == b * d * d * d / 12
    assert pytest.approx(iyy_c) == d * b * b * b / 12
    assert pytest.approx(ixy_c, abs=1e-6) == 0
