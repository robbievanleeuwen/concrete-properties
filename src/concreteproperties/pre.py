"""Contains doctored versions of the sectionproperties Geometry objects for concrete.

Also contains helper methods for adding bars to a geometry objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from more_itertools import peekable
from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.library.primitive_sections import circular_section_by_area
from shapely import LineString, Polygon
from shapely.ops import split

from concreteproperties.material import Concrete


if TYPE_CHECKING:
    import matplotlib.axes
    from sectionproperties.pre.geometry import CompoundGeometry

    from concreteproperties.material import Material, SteelBar, SteelStrand


class CPGeom:
    """A ``concreteproperties`` geometry object.

    A watered down implementation of the ``sectionproperties`` ``Geometry`` object,
    optimised for ``concreteproperties``.
    """

    def __init__(
        self,
        geom: Polygon,
        material: Material,
    ) -> None:
        """Inits the CPGeom class.

        Args:
            geom: Shapely polygon defining the geometry
            material: Material to apply to the geometry
        """
        # round polygon points and save geometry
        self.geom = self.round_geometry(geometry=geom, tol=6)

        # store material
        self.material = material

        # create points and facets
        self.points, self.facets = self.create_points_and_facets(geometry=self.geom)

        # create holes
        self.holes: list[tuple[float, float]] = []

        for hole in self.geom.interiors:
            hole_polygon = Polygon(hole)
            self.holes += tuple(hole_polygon.representative_point().coords)

    def round_geometry(
        self,
        geometry: Polygon,
        tol: int,
    ) -> Polygon:
        """Rounds the coordinates in ``geometry`` to tolerance ``tol``.

        Args:
            geometry: Geometry to round
            tol: Number of decimal places to round

        Returns:
            Rounded geometry
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
    ) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
        """Creates a list of points and facets from a shapely polygon.

        Args:
            geometry: Shapely polygon from which to create points and facets

        Returns:
            Points and facets
        """
        master_count = 0
        points: list[tuple[float, float]] = []
        facets: list[tuple[int, int]] = []

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
        self,
        points_list: list[tuple[float, float]],
        offset: int = 0,
    ) -> list[tuple[int, int]]:
        """Generates a list of facets given a list of points and a facet offset.

        Args:
            points_list: List of ordered points to create facets from
            offset: Facet offset integer

        Returns:
            List of facets
        """
        idx_peeker = peekable([idx + offset for idx, _ in enumerate(points_list)])
        return [(item, idx_peeker.peek(offset)) for item in idx_peeker]

    def calculate_area(self) -> float:
        """Calculates the area of the geometry.

        Returns:
            Geometry area
        """
        return self.geom.area

    def calculate_centroid(self) -> tuple[float, float]:
        """Calculates the centroid of the geometry.

        Returns:
            Geometry centroid
        """
        return self.geom.centroid.coords[0]

    def calculate_extents(self) -> tuple[float, float, float, float]:
        """Calculates the extents of the geometry.

        Calculates the minimum and maximum ``x`` and ``y`` values among the points
        describing the geometry.

        Returns:
            Extents (``x_min``, ``x_max``, ``y_min``, ``y_max``)
        """
        min_x, min_y, max_x, max_y = self.geom.bounds

        return min_x, max_x, min_y, max_y

    def split_section(
        self,
        point: tuple[float, float],
        theta: float,
    ) -> tuple[list[CPGeom], list[CPGeom]]:
        """Splits the geometry about a line.

        Args:
            point: Point on line
            theta: Angle line makes with horizontal axis

        Returns:
            Geometries above and below the line
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
        top_polys, bot_polys = self.sort_polys(polys=polys, point=point, vector=vector)

        # assign material properties and create cp geometry objects
        top_geoms = [
            (
                CPGeomConcrete(geom=poly, material=self.material)
                if isinstance(self.material, Concrete)
                else CPGeom(geom=poly, material=self.material)
            )
            for poly in top_polys
        ]
        bot_geoms = [
            (
                CPGeomConcrete(geom=poly, material=self.material)
                if isinstance(self.material, Concrete)
                else CPGeom(geom=poly, material=self.material)
            )
            for poly in bot_polys
        ]

        # ensure top geoms is in compression
        if theta <= np.pi / 2 and theta >= -np.pi / 2:
            return top_geoms, bot_geoms
        else:
            return bot_geoms, top_geoms

    def create_line_segment(
        self,
        point: tuple[float, float],
        vector: tuple[float, float],
        bounds: tuple[float, float, float, float],
    ) -> LineString:
        """Creates a ``shapely`` line.

        Creates a ``shapely`` line string defined by a ``point`` and ``vector`` and
        bounded by ``bounds``.

        Args:
            point: Point on line
            vector: Vector defining direction of line
            bounds: Bounds of the geometry

        Returns:
            Shapely line string
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
        polys: list[Polygon],
        point: tuple[float, float],
        vector: tuple[float, float],
    ) -> tuple[list[Polygon], list[Polygon]]:
        """Sorts polygons that are above and below the line.

        Args:
            polys: Polygons to sort
            point: Point on line
            vector: Vector defining direction of line

        Returns:
            Polygons above and below the line
        """
        top_polys: list[Polygon] = []
        bot_polys: list[Polygon] = []
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
    ) -> matplotlib.axes.Axes:
        """Plots the geometry.

        Args:
            title: Plot title
            kwargs: Passed to
                :meth:`~sectionproperties.pre.geometry.Geometry.plot_geometry`

        Returns:
            Matplotlib axes object
        """
        return self.to_sp_geom().plot_geometry(title=title, **kwargs)

    def to_sp_geom(self) -> Geometry:
        """Converts self to a *sectionproperties* geometry object.

        Returns:
            ``sectionproperties`` geometry object
        """
        return Geometry(geom=self.geom, material=self.material)


class CPGeomConcrete(CPGeom):
    """A ``concreteproperties`` Geometry object for concrete geometries."""

    def __init__(
        self,
        geom: Polygon,
        material: Concrete,
    ) -> None:
        """Inits the CPGeomConcrete class.

        Args:
            geom: Shapely polygon defining the geometry
            material: Material to apply to the geometry
        """
        super().__init__(
            geom=geom,
            material=material,
        )

        # ensure material is a Concrete object
        self.material = material


def add_bar(
    geometry: Geometry | CompoundGeometry,
    area: float,
    material: SteelBar | SteelStrand,
    x: float,
    y: float,
    n: int = 4,
) -> CompoundGeometry:
    """Adds a reinforcing bar to a ``sectionproperties`` geometry.

    Bars are discretised by four points by default.

    Args:
        geometry: Reinforced concrete geometry to which the new bar will be added
        area: Bar cross-sectional area
        material: Material object for the bar
        x: x-position of the bar
        y: y-position of the bar
        n: Number of points to discretise the bar circle

    Returns:
        Reinforced concrete geometry with added bar
    """
    bar = circular_section_by_area(area=area, n=n, material=material).shift_section(
        x_offset=x, y_offset=y
    )

    return (geometry - bar) + bar


def add_bar_rectangular_array(
    geometry: Geometry | CompoundGeometry,
    area: float,
    material: SteelBar | SteelStrand,
    n_x: int,
    x_s: float,
    n_y: int = 1,
    y_s: float = 0,
    anchor: tuple[float, float] = (0, 0),
    exterior_only: bool = False,
    n: int = 4,
) -> CompoundGeometry:
    """Adds a rectangular array of reinforcing bars to a ``sectionproperties`` geometry.

    Bars are discretised by four points by default.

    Args:
        geometry: Reinforced concrete geometry to which the new bar will be added
        area: Bar cross-sectional area
        material: Material object for the bar
        n_x: Number of bars in the x-direction
        x_s: Spacing in the x-direction
        n_y: Number of bars in the y-direction
        y_s: Spacing in the y-direction
        anchor: Coordinates of the bottom left hand bar in the rectangular array
        exterior_only: If set to True, only returns bars on the external perimeter
        n: Number of points to discretise the bar circle

    Returns:
        Reinforced concrete geometry with added bar
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
                bar = circular_section_by_area(area=area, n=n, material=material)
                x = anchor[0] + i_idx * x_s
                y = anchor[1] + j_idx * y_s
                bar = bar.shift_section(x_offset=x, y_offset=y)
                geometry = (geometry - bar) + bar

    return geometry


def add_bar_circular_array(
    geometry: Geometry | CompoundGeometry,
    area: float,
    material: SteelBar | SteelStrand,
    n_bar: int,
    r_array: float,
    theta_0: float = 0,
    ctr: tuple[float, float] = (0, 0),
    n: int = 4,
) -> CompoundGeometry:
    """Adds a circular array of reinforcing bars to a ``sectionproperties`` geometry.

    Bars are discretised by four points by default.

    Args:
        geometry: Reinforced concrete geometry to which the news bar will be added
        area: Bar cross-sectional area
        material: Material object for the bar
        n_bar: Number of bars in the array
        r_array: Radius of the circular array
        theta_0: Initial angle (in radians) that the first bar makes with the
            horizontal axis in the circular array
        ctr: Centre of the circular array
        n: Number of points to discretise the bar circle

    Returns:
        Reinforced concrete geometry with added bar
    """
    d_theta = 2 * np.pi / n_bar

    for idx in range(n_bar):
        bar = circular_section_by_area(area=area, n=n, material=material)
        theta = theta_0 + idx * d_theta
        x = ctr[0] + r_array * np.cos(theta)
        y = ctr[1] + r_array * np.sin(theta)
        bar = bar.shift_section(x_offset=x, y_offset=y)
        geometry = (geometry - bar) + bar

    return geometry
