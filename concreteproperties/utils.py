from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from sectionproperties.analysis.fea import principal_coordinate, global_coordinate

if TYPE_CHECKING:
    from sectionproperties.pre.geometry import Geometry, CompoundGeometry


def get_strain(
    point: Tuple[float, float],
    point_na: Tuple[float, float],
    d_n: float,
    theta: float,
    ultimate_strain: float,
) -> float:
    """Determines the strain at point `point` given neutral axis depth `d_n` and
    neutral axis angle `theta`. Positive strain is compression.

    :param point: Point at which to evaluate the strain
    :type point: Tuple[float, float]
    :param point_na: Point on the neutral axis
    :type point_na: Tuple[float, float]
    :param float d_n: Depth of the neutral axis from the extreme compression fibre
    :param float theta: Angle the neutral axis makes with the horizontal axis
    :param float ultimate_strain: Strain at the extreme compression fibre

    :return: Strain
    :rtype: float
    """

    # convert point to local coordinates
    u, v = principal_coordinate(phi=theta * 180 / np.pi, x=point[0], y=point[1])

    # convert point_na to local coordinates
    u_na, v_na = principal_coordinate(
        phi=theta * 180 / np.pi, x=point_na[0], y=point_na[1]
    )

    # calculate distance between NA and point in `v` direction
    d = v - v_na

    return d / d_n * ultimate_strain


def get_point_from_strain(
    strain: float,
    point_na: Tuple[float, float],
    d_n: float,
    theta: float,
    ultimate_strain: float,
) -> Tuple[float, float]:
    """Returns a point experiencing a certain `strain`, given neutral axis depth
    `d_n` and neutral axis angle `theta`.

    :param float strain: Strain at which to get point
    :param point_na: Point on the neutral axis
    :type point_na: Tuple[float, float]
    :param float d_n: Depth of the neutral axis from the extreme compression fibre
    :param float theta: Angle the neutral axis makes with the horizontal axis
    :param float ultimate_strain: Strain at the extreme compression fibre

    :return: Point experiencing `strain`
    :rtype: Tuple[float, float]
    """

    # depth to point from NA
    d = strain / ultimate_strain * d_n

    # convert depth to global coordinates
    dx, dy = global_coordinate(phi=theta * 180 / np.pi, x11=0, y22=d)

    return point_na[0] + dx, point_na[1] + dy


def point_on_neutral_axis(
    extreme_fibre: Tuple[float, float],
    d_n: float,
    theta: float,
) -> Tuple[float, float]:
    """Returns a point on the neutral axis given an extreme fibre, a depth to the
    neutral axis and a neutral axis angle.

    :param extreme_fibre: Global coordinate of the extreme compression fibre
    :type extreme_fibre: Tuple[float, float]
    :param float d_n: Depth of the neutral axis from the extreme compression fibre
    :param float theta: Angle the neutral axis makes with the horizontal axis

    :return: Point on the neutral axis in global coordinates `(x, y)`
    :rtype: Tuple[float, float]
    """

    # determine the coordinate of the point wrt the local axis
    (u, v) = principal_coordinate(
        phi=theta * 180 / np.pi, x=extreme_fibre[0], y=extreme_fibre[1]
    )

    # subtract the neutral axis depth
    v -= d_n

    # convert point back to global coordinates
    return global_coordinate(phi=theta * 180 / np.pi, x11=u, y22=v)


def split_section(
    geometry: CompoundGeometry,
    point: Tuple[float, float],
    theta: float,
) -> Tuple[List[Geometry], List[Geometry]]:
    """Splits the geometry along a line defined by a `point` and rotation angle
    `theta`.

    :param geometry: Geometry to split
    :type geometry: :class:`sectionproperties.pre.geometry.CompoundGeometry`
    :param point: Point at which to split the geometry `(x, y)`
    :type point: Tuple[float, float]
    :param float theta: Angle the neutral axis makes with the horizontal axis

    :return: Split geometry above and below the line
    :rtype:
        Tuple[List[:class:`sectionproperties.pre.geometry.Geometry],
        List[sectionproperties.pre.geometry.Geometry]]
    """

    # split the section using the sectionproperties method
    top_geoms, bot_geoms = geometry.split_section(
        point_i=point, vector=(np.cos(theta), np.sin(theta))
    )

    # ensure top geoms is in compression
    # sectionproperties definition is based on global coordinate system only
    if theta < np.pi / 2 and theta > -np.pi / 2:
        return top_geoms, bot_geoms
    else:
        return bot_geoms, top_geoms


def calculate_extreme_fibre(
    points: List[List[float]],
    theta: float,
) -> Tuple[Tuple[float, float], float]:
    """Calculates the locations of the extreme compression fibre in global
    coordinates given a neutral axis angle `theta`.

    :param points: Points over which to search for an extreme fibre
    :type points: List[List[float, float]]
    :param float theta: Angle the neutral axis makes with the horizontal axis

    :return: Global coordinate of the extreme compression fibre `(x, y)` and the
        neutral axis depth at the extreme tensile fibre
    :rtype: Tuple[Tuple[float, float], float]
    """

    # loop through all points in the geometry
    for (idx, point) in enumerate(points):
        # determine the coordinate of the point wrt the local axis
        (u, v) = principal_coordinate(phi=theta * 180 / np.pi, x=point[0], y=point[1])

        # initialise min/max variable & point
        if idx == 0:
            v_min = v
            min_pt = point
            v_max = v
            max_pt = point

        # update the min/max & point where necessary
        if v < v_min:
            v_min = v
            min_pt = point

        if v > v_max:
            v_max = v
            max_pt = point

    # calculate depth of neutral axis at tensile fibre
    d_t = v_max - v_min

    return max_pt, d_t


def gauss_points(
    n: float,
):
    """x

    x
    """

    if n == 1:
        return [[0.5, 1.0 / 3, 1.0 / 3]]
    elif n == 3:
        return [
            [1.0 / 6, 0, 0.5],
            [1.0 / 6, 0.5, 0],
            [1.0 / 6, 0.5, 0.5],
        ]
    else:
        raise ValueError(f"{n} gauss points not implemented.")


def shape_function(
    coords,
    gauss_point,
):
    """x

    x
    """

    xi = gauss_point[1]
    eta = gauss_point[2]

    N = np.array([1 - xi - eta, xi, eta])
    dN = [[-1, -1], [1, 0], [0, 1]]

    # calculate jacobian
    J_mat = np.matmul(coords, dN)
    j = np.linalg.det(J_mat)

    return N, j
