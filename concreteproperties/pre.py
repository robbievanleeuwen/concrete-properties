from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np
import sectionproperties.pre.library.primitive_sections as sp_ps
from more_itertools import peekable
from sectionproperties.pre.geometry import Geometry
from shapely.geometry import LineString, Polygon
from shapely.ops import split

from concreteproperties.material import Concrete

if TYPE_CHECKING:
    import matplotlib
    from sectionproperties.pre.geometry import CompoundGeometry

    from concreteproperties.material import Material, SteelBar


class CPGeom:
    """Watered down implementation of the *sectionproperties* Geometry object, optimised
    for *concreteproperties*.
    """

    def __init__(
        self,
        geom: Polygon,
        material: Material,
    ):
        """Inits the CPGeom class.

        :param geom: Shapely polygon defining the geometry
        :param material: Material to apply to the geometry
        """

        # round polygon points and save geometry
        self.geom = self.round_geometry(geometry=geom, tol=6)

        # store material
        self.material = material

        # create points and facets
        self.points, self.facets = self.create_points_and_facets(geometry=self.geom)

        # create holes
        self.holes: List[Tuple[float, float]] = []

        for hole in self.geom.interiors:
            hole_polygon = Polygon(hole)
            self.holes += tuple(hole_polygon.representative_point().coords)

    def round_geometry(
        self,
        geometry: Polygon,
        tol: int,
    ) -> Polygon:
        """Rounds the coordinates in ``geometry`` to tolerance ``tol``.

        :param geometry: Geometry to round
        :param tol: Number of decimal places to round

        :return: Rounded geometry
        """

        if geometry.exterior:
            rounded_exterior = np.round(geometry.exterior.coords, tol)
        else:
            rounded_exterior = np.array([None])

        rounded_interiors = []
        for interior in geometry.interiors:
            rounded_interiors.append(np.round(interior.coords, tol))

        if not rounded_exterior.any():
            return Polygon()
        else:
            return Polygon(rounded_exterior, rounded_interiors)

    def create_points_and_facets(
        self,
        geometry: Polygon,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
        """Creates a list of points and facets from a shapely polygon.

        :param geometry: Shapely polygon from which to create points and facets

        :return: Points and facets
        """

        master_count = 0
        points: List[Tuple[float, float]] = []
        facets: List[Tuple[int, int]] = []

        # perimeter, note in shapely last point == first point
        if geometry.exterior:
            for coords in list(geometry.exterior.coords[:-1]):
                points.append(coords)
                master_count += 1

        facets += self.create_facets(points)
        exterior_count = master_count

        # holes
        for idx, hole in enumerate(geometry.interiors):
            break_count = master_count
            int_points = []

            for coords in hole.coords[:-1]:
                int_points.append(coords)
                master_count += 1

            # (idx > 0), (idx < 1) are like a 'step functions'
            offset = break_count * (idx > 0) + exterior_count * (idx < 1)
            facets += self.create_facets(int_points, offset=offset)
            points += int_points

        return points, facets

    def create_facets(
        self, points_list: List[Tuple[float, float]], offset: int = 0
    ) -> List[Tuple[int, int]]:
        """Generates a list of facets given a list of points and a facet offset.

        :param points_list: List of ordered points to create facets from
        :param offset: Facet offset integer

        :return: List of facets
        """

        idx_peeker = peekable([idx + offset for idx, _ in enumerate(points_list)])
        return [(item, idx_peeker.peek(offset)) for item in idx_peeker]

    def calculate_area(
        self,
    ) -> float:
        """Calculates the area of the geometry.

        :return: Geometry area
        """

        return self.geom.area

    def calculate_centroid(
        self,
    ) -> Tuple[float, float]:
        """Calculates the centroid of the geometry.

        :return: Geometry centroid
        """

        return self.geom.centroid.coords[0]

    def calculate_extents(
        self,
    ) -> Tuple[float, float, float, float]:
        """Calculates the minimum and maximum ``x`` and ``y`` values among the points
        describing the geometry.

        :return: Extents (``x_min``, ``x_max``, ``y_min``, ``y_max``)
        """

        min_x, min_y, max_x, max_y = self.geom.bounds  # type: ignore

        return min_x, max_x, min_y, max_y

    def split_section(
        self,
        point: Tuple[float, float],
        theta: float,
    ) -> Tuple[List[CPGeom], List[CPGeom]]:
        """Splits the geometry about a line.

        :param point: Point on line
        :param theta: Angle line makes with horizontal axis

        :return: Geometries above and below the line
        """

        # round point
        point = np.round(point, 6)

        # generate unit vector
        vector = np.cos(theta), np.sin(theta)

        # calculate bounds of geometry
        bounds = self.calculate_extents()

        # generate line segment that matches bounds of geometry object
        line_seg = self.create_line_segment(point=point, vector=vector, bounds=bounds)

        # check to see if line intersects geometry
        if line_seg.intersects(self.geom):
            # split geometries
            polys = split(geom=self.geom, splitter=line_seg).geoms
        else:
            polys = [self.geom]

        # sort geometries
        top_polys, bot_polys = self.sort_polys(polys=polys, point=point, vector=vector)  # type: ignore

        # assign material properties and create cp geometry objects
        top_geoms = [
            CPGeomConcrete(geom=poly, material=self.material)
            if isinstance(self.material, Concrete)
            else CPGeom(geom=poly, material=self.material)
            for poly in top_polys
        ]
        bot_geoms = [
            CPGeomConcrete(geom=poly, material=self.material)
            if isinstance(self.material, Concrete)
            else CPGeom(geom=poly, material=self.material)
            for poly in bot_polys
        ]

        # ensure top geoms is in compression
        if theta <= np.pi / 2 and theta >= -np.pi / 2:
            return top_geoms, bot_geoms
        else:
            return bot_geoms, top_geoms

    def create_line_segment(
        self,
        point: Tuple[float, float],
        vector: Tuple[float, float],
        bounds: Tuple[float, float, float, float],
    ) -> LineString:
        """Creates a shapely line string defined by a ``point`` and ``vector`` and
        bounded by ``bounds``.

        :param point: Point on line
        :param vector: Vector defining direction of line
        :param bounds: Bounds of the geometry

        :return: Shapely line string
        """

        tol = 1e-6  # distance to displace start of line from bounds

        # not a vertical line
        if abs(vector[0]) > 1e-12:
            v_ratio = vector[1] / vector[0]
            x1 = bounds[0] - tol
            x2 = bounds[1] + tol
            y1 = v_ratio * (x1 - point[0]) + point[1]
            y2 = v_ratio * (x2 - point[0]) + point[1]

        # vertical line
        else:
            v_ratio = vector[0] / vector[1]
            y1 = bounds[2] - tol
            y2 = bounds[3] + tol
            x1 = v_ratio * (y1 - point[1]) + point[0]
            x2 = v_ratio * (y2 - point[1]) + point[0]

        return LineString([(x1, y1), (x2, y2)])

    def sort_polys(
        self,
        polys: List[Polygon],
        point: Tuple[float, float],
        vector: Tuple[float, float],
    ) -> Tuple[List[Polygon], List[Polygon]]:
        """Sorts polygons that are above and below the line.

        :param polys: Polygons to sort
        :param point: Point on line
        :param vector: Vector defining direction of line

        :return: Polygons above and below the line
        """

        top_polys: List[Polygon] = []
        bot_polys: List[Polygon] = []
        v_ratio = vector[1] / vector[0]

        for poly in polys:
            # get point inside polygon
            px, py = poly.representative_point().coords[0]

            # not a vertical line
            if abs(vector[0]) > 1e-12:
                # get point on line at x-coordinate of representative point
                y_line = point[1] + (px - point[0]) * v_ratio

                # if we are below the line
                if py < y_line:
                    bot_polys.append(poly)
                # if we are above the line
                else:
                    top_polys.append(poly)

            # vertical line
            else:
                # if we are to the right of the line
                if px < point[0]:
                    bot_polys.append(poly)
                # if we are to the left of the line
                else:
                    top_polys.append(poly)

        return top_polys, bot_polys

    def plot_geometry(
        self,
        title: str = "Cross-Section Geometry",
        **kwargs,
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots the geometry.

        :param title: Plot title
        :param kwargs: Passed to
            :meth:`~sectionproperties.pre.geometry.Geometry.plot_geometry`

        :return: Matplotlib axes object
        """

        return self.to_sp_geom().plot_geometry(title=title, **kwargs)

    def to_sp_geom(
        self,
    ) -> Geometry:
        """Converts self to a *sectionproperties* geometry object.

        :return: *sectionproperties* geometry object
        """

        return Geometry(geom=self.geom, material=self.material)  # type: ignore


class CPGeomConcrete(CPGeom):
    """*concreteproperties* Geometry class for concrete geometries."""

    def __init__(
        self,
        geom: Polygon,
        material: Concrete,
    ):
        """Inits the CPGeomConcrete class.

        :param geom: Shapely polygon defining the geometry
        :param material: Material to apply to the geometry
        """

        super().__init__(
            geom=geom,
            material=material,
        )

        # ensure material is a Concrete object
        self.material = material


def add_bar(
    geometry: Union[Geometry, CompoundGeometry],
    area: float,
    material: SteelBar,
    x: float,
    y: float,
    n: int = 4,
) -> CompoundGeometry:
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

    return (geometry - bar) + bar  # type: ignore


def add_bar_rectangular_array(
    geometry: Union[Geometry, CompoundGeometry],
    area: float,
    material: SteelBar,
    n_x: int,
    x_s: float,
    n_y: int = 1,
    y_s: float = 0,
    anchor: Tuple[float, float] = (0, 0),
    exterior_only: bool = False,
    n: int = 4,
) -> CompoundGeometry:
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
                geometry = (geometry - bar) + bar  # type: ignore

    return geometry  # type: ignore


def add_bar_circular_array(
    geometry: Union[Geometry, CompoundGeometry],
    area: float,
    material: SteelBar,
    n_bar: int,
    r_array: float,
    theta_0: float = 0,
    ctr: Tuple[float, float] = (0, 0),
    n: int = 4,
) -> CompoundGeometry:
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
        geometry = (geometry - bar) + bar  # type: ignore

    return geometry  # type: ignore
