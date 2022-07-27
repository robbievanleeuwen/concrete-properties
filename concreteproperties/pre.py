from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import sectionproperties.pre.library.primitive_sections as sp_ps

if TYPE_CHECKING:
    from sectionproperties.pre.geometry import CompoundGeometry, Geometry

    from concreteproperties.material import Steel


def add_bar(
    geometry: Union[Geometry, CompoundGeometry],
    area: float,
    material: Steel,
    x: float,
    y: float,
    n: int = 4,
) -> Union[Geometry, CompoundGeometry]:
    """Adds a reinforcing bar to a *sectionproperties* geometry.

    Bars are discretised by four points by default.

    :param geometry: Reinforced concrete geometry to which the new bar will be added
    :param area: Bar cross-sectional area
    :param material: Material object for the bar
    :param x: x-position of the bar
    :param y: y-position of the bar
    :param n: Number of points to discretise the bar circle

    :return: Reinforced concrete geometry with added bar
    """

    bar = sp_ps.circular_section_by_area(
        area=area, n=n, material=material  # type: ignore
    ).shift_section(x_offset=x, y_offset=y)

    return (geometry - bar) + bar


def add_bar_rectangular_array(
    geometry: Union[Geometry, CompoundGeometry],
    area: float,
    material: Steel,
    n_x: int,
    x_s: float,
    n_y: int = 1,
    y_s: float = 0,
    anchor: Tuple[float, float] = (0, 0),
    exterior_only: bool = False,
    n: int = 4,
) -> Union[Geometry, CompoundGeometry]:
    """Adds a rectangular array of reinforcing bars to a *sectionproperties* geometry.

    Bars are discretised by four points by default.

    :param geometry: Reinforced concrete geometry to which the new bar will be added
    :param area: Bar cross-sectional area
    :param material: Material object for the bar
    :param n_x: Number of bars in the x-direction
    :param x_s: Spacing in the x-direction
    :param n_y: Number of bars in the y-direction
    :param y_s: Spacing in the y-direction
    :param anchor: Coordinates of the bottom left hand bar in the rectangular array
    :param exterior_only: If set to True, only returns bars on the external perimeter
    :param n: Number of points to discretise the bar circle

    :return: Reinforced concrete geometry with added bar
    """

    for j_idx in range(n_y):
        for i_idx in range(n_x):
            # check to see if we are adding a bar
            if exterior_only:
                if i_idx != 0 and i_idx != n_x - 1 and j_idx != 0 and j_idx != n_y - 1:
                    add_bar = False
                else:
                    add_bar = True
            else:
                add_bar = True

            if add_bar:
                bar = sp_ps.circular_section_by_area(area=area, n=n, material=material)  # type: ignore
                x = anchor[0] + i_idx * x_s
                y = anchor[1] + j_idx * y_s
                bar = bar.shift_section(x_offset=x, y_offset=y)
                geometry = (geometry - bar) + bar

    return geometry


def add_bar_circular_array(
    geometry: Union[Geometry, CompoundGeometry],
    area: float,
    material: Steel,
    n_bar: int,
    r_array: float,
    theta_0: float = 0,
    ctr: Tuple[float, float] = (0, 0),
    n: int = 4,
) -> Union[Geometry, CompoundGeometry]:
    """Adds a circular array of reinforcing bars to a *sectionproperties* geometry.

    Bars are discretised by four points by default.

    :param geometry: Reinforced concrete geometry to which the news bar will be added
    :param area: Bar cross-sectional area
    :param material: Material object for the bar
    :param n_bar: Number of bars in the array
    :param r_array: Radius of the circular array
    :param theta_0: Initial angle (in radians) that the first bar makes with the
        horizontal axis in the circular array
    :param ctr: Centre of the circular array
    :param n: Number of points to discretise the bar circle

    :return: Reinforced concrete geometry with added bar
    """

    d_theta = 2 * np.pi / n_bar

    for idx in range(n_bar):
        bar = sp_ps.circular_section_by_area(area=area, n=n, material=material)  # type: ignore
        theta = theta_0 + idx * d_theta
        x = ctr[0] + r_array * np.cos(theta)
        y = ctr[1] + r_array * np.sin(theta)
        bar = bar.shift_section(x_offset=x, y_offset=y)
        geometry = (geometry - bar) + bar

    return geometry
