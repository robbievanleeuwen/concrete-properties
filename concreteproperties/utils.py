from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from rich.progress import BarColumn, Progress, ProgressColumn, SpinnerColumn, TextColumn
from rich.table import Column
from rich.text import Text

from concreteproperties.pre import CPGeomConcrete

if TYPE_CHECKING:
    from sectionproperties.pre.geometry import CompoundGeometry

    from concreteproperties.pre import CPGeom


def get_service_strain(
    point: Tuple[float, float],
    point_na: Tuple[float, float],
    theta: float,
    kappa: float,
) -> float:
    r"""Determines the strain at point `point` given curvature `kappa` and neutral axis
    angle `theta`. Positive strain is compression.

    :param point: Point at which to evaluate the strain
    :param point_na: Point on the neutral axis
    :param theta: Angle (in radians) the neutral axis makes with the horizontal
        axis (:math:`-\pi \leq \theta \leq \pi`)
    :param kappa: Curvature

    :return: Strain
    """

    # convert point to local coordinates
    _, v = global_to_local(theta=theta, x=point[0], y=point[1])

    # convert point_na to local coordinates
    _, v_na = global_to_local(theta=theta, x=point_na[0], y=point_na[1])

    # calculate distance between NA and point in `v` direction
    d = v - v_na

    return kappa * d


def get_ultimate_strain(
    point: Tuple[float, float],
    point_na: Tuple[float, float],
    d_n: float,
    theta: float,
    ultimate_strain: float,
) -> float:
    r"""Determines the strain at point `point` given neutral axis depth `d_n` and
    neutral axis angle `theta`. Positive strain is compression.

    :param point: Point at which to evaluate the strain
    :param point_na: Point on the neutral axis
    :param d_n: Depth of the neutral axis from the extreme compression fibre
    :param theta: Angle (in radians) the neutral axis makes with the horizontal
        axis (:math:`-\pi \leq \theta \leq \pi`)
    :param ultimate_strain: Concrete strain at failure

    :return: Strain
    """

    # convert point to local coordinates
    _, v = global_to_local(theta=theta, x=point[0], y=point[1])

    # convert point_na to local coordinates
    _, v_na = global_to_local(theta=theta, x=point_na[0], y=point_na[1])

    # calculate distance between NA and point in `v` direction
    d = v - v_na

    return d / d_n * ultimate_strain


def point_on_neutral_axis(
    extreme_fibre: Tuple[float, float],
    d_n: float,
    theta: float,
) -> Tuple[float, float]:
    r"""Returns a point on the neutral axis given an extreme fibre, a depth to the
    neutral axis and a neutral axis angle.

    :param extreme_fibre: Global coordinate of the extreme compression fibre
    :param d_n: Depth of the neutral axis from the extreme compression fibre
    :param theta: Angle (in radians) the neutral axis makes with the horizontal
        axis (:math:`-\pi \leq \theta \leq \pi`)

    :return: Point on the neutral axis in global coordinates `(x, y)`
    """

    # determine the coordinate of the point wrt the local axis
    u, v = global_to_local(theta=theta, x=extreme_fibre[0], y=extreme_fibre[1])

    # subtract the neutral axis depth
    v -= d_n

    # convert point back to global coordinates
    return local_to_global(theta=theta, u=u, v=v)


def split_geom_at_strains(
    geom: Union[CPGeom, CPGeomConcrete],
    theta: float,
    point_na: Tuple[float, float],
    ultimate: bool,
    ultimate_strain: Optional[float] = None,
    d_n: Optional[float] = None,
    kappa: Optional[float] = None,
) -> Union[List[CPGeom], List[CPGeomConcrete]]:
    r"""Splits geometries at discontinuities in its stress-strain profile.

    :param geom: Geometry to split
    :param theta: Angle (in radians) the neutral axis makes with the horizontal
        axis (:math:`-\pi \leq \theta \leq \pi`)
    :param point_na: Point on the neutral axis
    :param ultimate: If set to True, uses ultimate stress-strain profile for concrete
        geometries
    :param ultimate_strain: Concrete strain at failure (required for ``ultimate=True``
        only)
    :param d_n: Depth of the neutral axis from the extreme compression fibre (required
        for ``ultimate=True`` only)
    :param kappa: Curvature

    :return: List of split geometries
    """

    # handle zero curvature
    if kappa == 0:
        return [geom]

    # create splits in concrete geometries at points in stress-strain profiles
    split_geoms: Union[List[CPGeom], List[CPGeomConcrete]] = []

    if ultimate and isinstance(geom, CPGeomConcrete):
        strains = geom.material.ultimate_stress_strain_profile.get_unique_strains()
    else:
        strains = geom.material.stress_strain_profile.get_unique_strains()

    # make geom a list of geometries
    geom_list = [geom]

    # initialise top_geoms in case of two unique strains
    top_geoms = geom_list
    continuing_geoms = []

    # loop through intermediate points on stress-strain profile
    for strain in strains[1:-1]:
        # depth to points of *strain* from NA
        # ultimate case
        if ultimate and ultimate_strain and d_n:
            d = strain / ultimate_strain * d_n
        # service case
        elif kappa:
            d = strain / kappa
        else:
            raise ValueError("Not enough arguments provided.")

        # convert depth to global coordinates
        dx, dy = local_to_global(theta=theta, u=0, v=d)

        # calculate location of point
        pt = point_na[0] + dx, point_na[1] + dy

        # make list of geometries that will need to continue to be split after the
        # split operation, i.e. those above the split
        continuing_geoms = []

        # split concrete geometries (from bottom up)
        for g in geom_list:
            top_geoms, bot_geoms = g.split_section(
                point=pt,
                theta=theta,
            )

            # save bottom geoms
            split_geoms.extend(bot_geoms)

            # save continuing geoms
            continuing_geoms.extend(top_geoms)

        # update geom_list for next strain
        geom_list = continuing_geoms

    # save final top geoms
    split_geoms.extend(continuing_geoms)

    return split_geoms


def calculate_extreme_fibre(
    points: List[Tuple[float, float]],
    theta: float,
) -> Tuple[Tuple[float, float], float]:
    r"""Calculates the locations of the extreme compression fibre in global
    coordinates given a neutral axis angle `theta`.

    :param points: Points over which to search for an extreme fibre
    :param theta: Angle (in radians) the neutral axis makes with the horizontal
        axis (:math:`-\pi \leq \theta \leq \pi`)

    :return: Global coordinate of the extreme compression fibre `(x, y)` and the
        neutral axis depth at the extreme tensile fibre
    """

    # initialise min/max variable & point
    max_pt = points[0]
    _, v = global_to_local(theta=theta, x=points[0][0], y=points[0][1])
    v_min = v
    v_max = v

    # loop through all points
    for idx, point in enumerate(points[1:]):
        # determine the coordinate of the point wrt the local axis
        _, v = global_to_local(theta=theta, x=point[0], y=point[1])

        # update the min/max & point where necessary
        if v < v_min:
            v_min = v

        if v > v_max:
            v_max = v
            max_pt = point

    # calculate depth of neutral axis at tensile fibre
    d_t = v_max - v_min

    return max_pt, d_t


def calculate_max_bending_depth(
    points: List[Tuple[float, float]],
    c_local_v: float,
    theta: float,
) -> float:
    r"""Calculates the maximum distance from the centroid to an extreme fibre when
    bending about an axis `theta`.

    :param points: Points over which to search for a bending depth
    :param c_local_v: Centroid coordinate in the local v-direction
    :param theta: Angle (in radians) the bending axis makes with the horizontal
        axis (:math:`-\pi \leq \theta \leq \pi`)

    :return: Maximum bending depth
    """

    max_bending_depth = 0

    # loop through all points
    for idx, point in enumerate(points):
        # determine the coordinate of the point wrt the local axis
        _, v = global_to_local(theta=theta, x=point[0], y=point[1])

        max_bending_depth = max(c_local_v - v, max_bending_depth)

    return max_bending_depth


def gauss_points(
    n: float,
) -> List[List[float]]:
    """Returns the Gaussian weights and locations for *n* point Gaussian integration of
    a linear triangular element.

    :param n: Number of Gauss points (1, 3 or 6)

    :return: An *n x 3* matrix consisting of the integration weight and the xi and eta
        locations for *n* Gauss points
    """

    if n == 1:
        return [[0.5, 1.0 / 3, 1.0 / 3]]
    elif n == 3:
        return [
            [1.0 / 6, 1.0 / 6, 1.0 / 6],
            [1.0 / 6, 2.0 / 3, 1.0 / 6],
            [1.0 / 6, 1.0 / 6, 2.0 / 3],
        ]
    else:
        raise ValueError(f"{n} gauss points not implemented.")


def shape_function(
    coords: np.ndarray,
    gauss_point: List[float],
) -> Tuple[np.ndarray, float]:
    """Computes shape functions and the determinant of the Jacobian matrix for a
    linear triangular element at a given Gauss point.

    :param coords: Global coordinates of the linear triangle vertices [2 x 3]
    :param gauss_point: Gaussian weight and isoparametric location of the Gauss point

    :return: The value of the shape functions *N(i)* at the given Gauss point [1 x 3]
        and the determinant of the Jacobian matrix *j*
    """

    xi = gauss_point[1]
    eta = gauss_point[2]

    N = np.array([1 - xi - eta, xi, eta])
    dN = np.array([[-1, -1], [1, 0], [0, 1]])

    # calculate jacobian
    J_mat = np.matmul(coords, dN)
    j = np.linalg.det(J_mat)

    return N, j


def calculate_local_extents(
    geometry: CompoundGeometry,
    cx: float,
    cy: float,
    theta: float,
) -> Tuple[float, float, float, float]:
    r"""Calculates the local extents of a geometry given a centroid and axis angle.

    :param geometry: Geometry over which to calculate extents
    :param cx: x-location of the centroid
    :param cy: y-location of the centroid
    :param theta: Angle (in radians) the neutral axis makes with the horizontal
        axis (:math:`-\pi \leq \theta \leq \pi`)

    :return: Local extents *(x11_max, x11_min, y22_max, y22_min)*
    """

    # initialise min, max variables
    pt0 = geometry.points[0]
    x11, y22 = global_to_local(theta=theta, x=pt0[0] - cx, y=pt0[1] - cy)
    x11_max = x11
    x11_min = x11
    y22_max = y22
    y22_min = y22

    # loop through all points in geometry
    for idx, pt in enumerate(geometry.points[1:]):
        # determine the coordinate of the point wrt the principal axis
        x11, y22 = global_to_local(theta=theta, x=pt[0] - cx, y=pt[1] - cy)

        # update the mins and maxes where necessary
        x11_max = max(x11_max, x11)
        x11_min = min(x11_min, x11)
        y22_max = max(y22_max, y22)
        y22_min = min(y22_min, y22)

    return x11_max, x11_min, y22_max, y22_min


def global_to_local(
    theta: float,
    x: float,
    y: float,
) -> Tuple[float, float]:
    r"""Determines the local coordinates of the global point (``x``, ``y``) given local
    axis angle ``theta``.

    :param theta: Angle (in radians) the local axis makes with the horizontal axis
        (:math:`-\pi \leq \theta \leq \pi`)
    :param x: x-coordinate of the point in the global axis
    :param y: y-coordinate of the point in the global axis

    :return: Local axis coordinates (``u``, ``v``)
    """

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return x * cos_theta + y * sin_theta, y * cos_theta - x * sin_theta


def local_to_global(
    theta: float,
    u: float,
    v: float,
) -> Tuple[float, float]:
    r"""Determines the global coordinates of the local point (``u``, ``v``) given local
    axis angle ``theta``.

    :param theta: Angle (in radians) the local axis makes with the horizontal axis
        (:math:`-\pi \leq \theta \leq \pi`)
    :param u: u-coordinate of the point in the local axis
    :param v: v-coordinate of the point in the local axis

    :return: Global axis coordinates (``x``, ``y``)
    """

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return u * cos_theta - v * sin_theta, u * sin_theta + v * cos_theta


class CustomTimeElapsedColumn(ProgressColumn):
    """Renders time elapsed in milliseconds."""

    def render(
        self,
        task: str = "Task",
    ) -> Text:
        """Show time remaining.

        :param task: Task string

        :return: Rich text object
        """

        elapsed = task.finished_time if task.finished else task.elapsed  # type: ignore

        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")

        elapsed_string = "[ {0:.4f} s ]".format(elapsed)

        return Text(elapsed_string, style="progress.elapsed")


def create_known_progress() -> Progress:
    """Returns a Rich Progress class for a known number of iterations.

    :return: Rich progress object
    """

    return Progress(
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description}", table_column=Column(ratio=1)
        ),
        BarColumn(bar_width=None, table_column=Column(ratio=1)),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        CustomTimeElapsedColumn(),
        expand=True,
    )


def create_unknown_progress() -> Progress:
    """Returns a Rich Progress class for an unknown number of iterations.

    :return: Rich progress object
    """

    return Progress(
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description}", table_column=Column(ratio=1)
        ),
        BarColumn(bar_width=None, table_column=Column(ratio=1)),
        CustomTimeElapsedColumn(),
        expand=True,
    )


class AnalysisError(Exception):
    pass
