from typing import List, Tuple, Union
import numpy as np
from scipy.optimize import brentq
from concreteproperties.material import Concrete, Steel
from concreteproperties.stress_strain_profile import WhitneyStressBlock
from sectionproperties.pre.geometry import Geometry, CompoundGeometry
from sectionproperties.analysis.section import Section
from sectionproperties.analysis.fea import (
    principal_coordinate,
    global_coordinate,
    gauss_points,
    shape_function,
)


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

    def ultimate_bending_capacity(
        self,
        theta: float,
        n: float,
    ) -> Tuple[float, float]:
        """Given a neutral axis angle `theta` and an axial force `n`, calculates the
        ultimate bending capacity.

        :param float theta: Angle the neutral axis makes with the horizontal axis
        :param float n: Axial force

        :return: Ultimate bending capacity about the x & y axes `(mx, my)`
        :rtype: Tuple[float, float]
        """

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        d_t = self.calculate_extreme_fibre(theta=theta, min=True)[1]

        # TODO: figure out how to calculate strain at d_n = 0 & d_n = d_t ?
        a = 1e-6 * d_t
        b = (1 - 1e-6) * d_t

        (d_n, r) = brentq(
            f=self.normal_force_convergence,
            a=a,
            b=b,
            args=(theta, n),
            xtol=1e-3,
            rtol=1e-6,
            full_output=True,
            disp=False,
        )

        print(d_n)

        return self.calculate_section_actions(d_n=d_n, theta=theta)
        # return self.calculate_section_actions(d_n=d_n, theta=theta)[1:]

    def calculate_section_actions(
        self,
        d_n: float,
        theta: float,
    ) -> Tuple[float, float, float]:
        """Given a neutral axis depth `d_n` and neutral axis angle `theta`, calculates
        the resultant bending moments `mx` and `my` and the net axial force `n`.

        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle the neutral axis makes with the horizontal axis

        :return: Section actions `(n, mx, my)`
        :rtype: Tuple[float, float, float]
        """

        # calculate extreme fibre in global coordinates
        extreme_fibre = self.calculate_extreme_fibre(theta=theta)

        # find point on neutral axis by shifting by d_n
        point_na = self.point_on_neutral_axis(
            extreme_fibre=extreme_fibre, d_n=d_n, theta=theta
        )

        # extract concrete section
        geom = self.extract_concrete(self.concrete_section.geometry)

        # split the section at the neutral axis
        top_geoms, bot_geoms = self.split_section(
            geometry=geom,
            point=point_na,
            theta=theta,
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

        # generate a mesh (refinement not important)
        new_geom.create_mesh(0, True)

        # create new section object
        new_section = Section(new_geom)

        # calculate section actions
        n, m_v = self.stress_analysis(
            conc_only_section=new_section, point_na=point_na, d_n=d_n, theta=theta
        )

        # convert mv to mx & my
        (my, mx) = global_coordinate(phi=theta * 180 / np.pi, x11=0, y22=m_v)

        return n, mx, my

    def normal_force_convergence(
        self,
        d_n: float,
        theta: float,
        n: float,
    ) -> float:
        """Given a neutral axis depth `d_n` and neutral axis angle `theta`, calculates
        the difference between the desired net axial force `n` and the axial force
        given `d_n` & `theta`.

        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle the neutral axis makes with the horizontal axis

        :return: Axial force convergence
        :rtype: float
        """

        # calculate convergence
        conv = n - self.calculate_section_actions(d_n=d_n, theta=theta)[0]

        return conv

    def calculate_extreme_fibre(
        self,
        theta: float,
        min: bool= False,
    ) -> Union[Tuple[float, float], Tuple[Tuple[float, float], float]]:
        """Calculates the locations of the extreme compression fibre in global
        coordinates given a neutral axis angle `theta`.

        :param float theta: Angle the neutral axis makes with the horizontal axis
        :param bool min: If true, returns the neutral axis depth at the extreme tensile
            fibre

        :return: Global coordinate of the extreme compression fibre `(x, y)`, if
            min=True also returns the neutral axis depth at the extreme tensile fibre
        :rtype: Union[Tuple[float, float], Tuple[Tuple[float, float], float]]
        """

        # loop through all points in the geometry
        for (idx, point) in enumerate(self.concrete_section.geometry.points):
            # determine the coordinate of the point wrt the new axis
            (u, v) = principal_coordinate(
                phi=theta * 180 / np.pi, x=point[0], y=point[1]
            )

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

        if min:
            # calculate depth of neutral axis at tensile fibre
            d_t = v_max - v_min

            return (max_pt, d_t)
        else:
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

    def extract_concrete(
        self,
        geometry: CompoundGeometry,
    ) -> CompoundGeometry:
        """Extracts only the concrete geometries from the cross-section."""

        geom_idx = 0

        for idx, geom in enumerate(geometry.geoms):
            if isinstance(geom.material, Concrete):
                if geom_idx == 0:
                    conc_geoms = geom
                else:
                    conc_geoms += geom

                geom_idx += 1

        return conc_geoms

    def split_section(
        self,
        geometry: CompoundGeometry,
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

    def stress_analysis(
        self,
        conc_only_section: Section,
        point_na: Tuple[float, float],
        d_n: float,
        theta: float,
    ) -> Tuple[float, float]:
        """Determines the section actions."""

        # initialise section actions
        n = 0
        n_steel = 0
        m_v = 0

        # Gauss points for 6 point Gaussian integration
        gps = gauss_points(6)

        # loop through all concrete elements
        for element in conc_only_section.elements:
            conc_mat = element.material

            # loop through each Gauss point
            for gp in gps:
                # determine shape function and jacobian
                (N, _, j) = shape_function(element.coords, gp)

                # get coordinates of the gauss point
                x = np.dot(N, np.transpose(element.coords[0, :]))
                y = np.dot(N, np.transpose(element.coords[1, :]))

                # get strain at gauss point
                d, strain = self.get_strain(
                    point=(x, y),
                    point_na=point_na,
                    d_n=d_n,
                    theta=theta,
                    ultimate_strain=conc_mat.stress_strain_profile.ultimate_strain,
                )

                # get stress at gauss point
                stress = conc_mat.stress_strain_profile.get_stress(strain=strain)

                n_el = gp[0] * stress * j
                n += n_el
                m_v += n_el * d

        # loop through all steel elements
        for element in self.concrete_section.elements:
            if isinstance(element.material, Steel):
                steel_mat = element.material

                # loop through each Gauss point
                for gp in gps:
                    # determine shape function and jacobian
                    (N, _, j) = shape_function(element.coords, gp)

                    # get coordinates of the gauss point
                    x = np.dot(N, np.transpose(element.coords[0, :]))
                    y = np.dot(N, np.transpose(element.coords[1, :]))

                    # get strain at gauss point
                    d, strain = self.get_strain(
                        point=(x, y),
                        point_na=point_na,
                        d_n=d_n,
                        theta=theta,
                        ultimate_strain=conc_mat.stress_strain_profile.ultimate_strain,
                    )

                    # get stress at gauss point
                    stress = steel_mat.stress_strain_profile.get_stress(strain=strain)

                    n_el = gp[0] * stress * j
                    n += n_el
                    n_steel += n_el
                    m_v += n_el * d

        print(n_steel)

        return n, m_v

    def get_strain(
        self,
        point: Tuple[float, float],
        point_na: float,
        d_n: float,
        theta: float,
        ultimate_strain: float,
    ) -> Tuple[float, float]:
        """Determines the strain at point `point` given neutral axis depth `d_n` and
        neutral axis angle `theta`. Positive strain is compression.

        :param point: Point at which to evaluate the strain
        :type point: Tuple[float, float]
        :param point_na: Point on the neutral axis
        :type point_na: Tuple[float, float]
        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle the neutral axis makes with the horizontal axis
        :param float ultimate_strain: Strain at the extreme compression fibre

        :return: xxx
        :rtype: Tuple[float, float]
        """

        # convert point to principal coordinates
        (u, v) = principal_coordinate(phi=theta * 180 / np.pi, x=point[0], y=point[1])

        # convert point_na to principal coordinates
        (u_na, v_na) = principal_coordinate(
            phi=theta * 180 / np.pi, x=point_na[0], y=point_na[1]
        )

        # calculate distance between NA and point in `v` direction
        d = v - v_na

        return d, d / d_n * ultimate_strain
