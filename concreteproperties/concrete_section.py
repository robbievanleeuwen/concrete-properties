from typing import List, Tuple, Union
import numpy as np
from concreteproperties.material import Concrete
from concreteproperties.stress_strain_profile import WhitneyStressBlock
from sectionproperties.pre.geometry import Geometry, CompoundGeometry
from sectionproperties.analysis.section import Section
from sectionproperties.analysis.fea import principal_coordinate, global_coordinate


class ConcreteSection:
    """Class for a reinforced concrete section."""

    def __init__(
        self,
        concrete_section: Section,
    ):
        """Inits the ConcreteSection class.

        :param concrete_section: *sectionproperties* section object describing the
            reinforced concrete section
        :type concrete_section: :class:`sectionproperties.analysis.section.Section`
        """

        self.concrete_section = concrete_section

    def calculate_section_actions(
        self,
        d_n: float,
        theta: float,
    ) -> List[float]:
        """Given a neutral axis depth `d_n` and neutral axis angle `theta`, calculates
        the resultant bending moments `m_x` and `m_y` and the net axial force `n`.

        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle the neutral axis makes with the horizontal axis
        """

        # calculate extreme fibre in global coordinates
        extreme_fibre = self.calculate_extreme_fibre(theta=theta)

        # find point on neutral axis by shifting by d_n
        pt_na = self.point_on_neutral_axis(
            extreme_fibre=extreme_fibre, d_n=d_n, theta=theta
        )

        # split the section at the neutral axis
        top_geoms, bot_geoms = self.split_section(
            geometry=self.concrete_section.geometry, point=pt_na, theta=theta,
        )

        # check to see if the concrete has a whitney stress block
        whitney = False  # start assuming False

        for material in self.concrete_section.materials:
            # if the material is concrete
            if isinstance(material, Concrete):
                # if the concrete has a WhitneyStressBlock
                if isinstance(material.stress_strain_profile, WhitneyStressBlock):
                    whitney = True
                    conc_mat = material

        # if we have a WhitneyStressBlock create an additional split in the section
        if whitney:
            # find point on whiney axis by shifting by gamma * d_n
            pt_whitney = self.point_on_neutral_axis(
                extreme_fibre=extreme_fibre,
                d_n=conc_mat.stress_strain_profile.gamma * d_n,
                theta=theta,
            )

            # combine top geometries into one CompoundGeometry
            top_geoms = CompoundGeometry(top_geoms)

            # split the section at the neutral axis
            top_geoms1, top_geoms2 = self.split_section(
                geometry=top_geoms,
                point=pt_whitney,
                theta=theta,
            )

            # combine geometries
            top_geoms = top_geoms1 + top_geoms2

        # combine geometries back into a new CompoundGeometry object
        new_geom = CompoundGeometry(top_geoms + bot_geoms)
        # new_geom.plot_geometry()

        for geom in new_geom.geoms:
            geom.plot_geometry()

        # generate a mesh (refinement not important)
        # new_geom.create_mesh(0)

        # create new section object
        # new_section = Section(new_geom)
        # new_section.plot_mesh()

    def calculate_extreme_fibre(
        self,
        theta: float,
    ) -> Tuple[float, float]:
        """Calculates the locations of the extreme compression fibre in global
        coordinates given a neutral axis angle `theta`.

        :param float theta: Angle the neutral axis makes with the horizontal axis

        :return: Global coordinate of the extreme compression fibre `(x, y)`
        :rtype: Tuple[float, float]
        """

        # loop through all points in the geometry
        for (idx, pt) in enumerate(self.concrete_section.geometry.points):
            # determine the coordinate of the point wrt the new axis
            (u, v) = principal_coordinate(phi=theta * 180 / np.pi, x=pt[0], y=pt[1])

            # initialise max variable & point
            if idx == 0:
                v_max = v
                max_pt = pt

            # update the max & pt where necessary
            if v > v_max:
                v_max = v
                max_pt = pt

        return max_pt

    def point_on_neutral_axis(
        self,
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

        # determine the coordinate of the point wrt the new axis
        (u, v) = principal_coordinate(
            phi=theta * 180 / np.pi, x=extreme_fibre[0], y=extreme_fibre[1]
        )

        # subtract the neutral axis depth
        v -= d_n

        # convert point back to global coordinates
        return global_coordinate(phi=theta * 180 / np.pi, x11=u, y22=v)

    def split_section(
        self,
        geometry: Union[Geometry, CompoundGeometry],
        point: Tuple[float, float],
        theta: float,
    ) -> Tuple[List[Geometry], List[Geometry]]:
        """Splits the section..."""

        # split the section using the sectionproperties method
        top_geoms, bot_geoms = geometry.split_section(
            point_i=point, vector=(np.cos(theta), np.sin(theta))
        )

        # ensure top geoms is in compression
        # sectionproperties definition is based on global coordinate system only
        if theta < np.pi / 2 and theta > -np.pi / 2:
            return (top_geoms, bot_geoms)
        else:
            return (bot_geoms, top_geoms)
