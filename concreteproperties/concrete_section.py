from typing import List, Tuple, Union
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from concreteproperties.material import Concrete, Steel
from concreteproperties.stress_strain_profile import WhitneyStressBlock
from concreteproperties.post import plotting_context
from sectionproperties.pre.geometry import Geometry, CompoundGeometry
from sectionproperties.analysis.section import Section
from sectionproperties.analysis.fea import (
    principal_coordinate,
    global_coordinate,
    gauss_points,
    shape_function,
)
import progress.bar as prog_bar


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

        # initialise class variables
        self.squash_load = 0  # squash load (positive = compression)
        self.tensile_load = 0  # tension load (negative = tension)
        self.axial_pc = [0, 0]  # axial plastic centroid (global coordinates)

        # calculate the plastic centroid (& squash load)
        self.calculate_plastic_centroid()

        # assign ultimate concrete strain
        for material in self.concrete_section.materials:
            if isinstance(material, Concrete):
                self.conc_ultimate_strain = (
                    material.stress_strain_profile.ultimate_strain
                )
                return

    def calculate_plastic_centroid(
        self,
    ):
        """Calculates the plastic centroid of the section assuming all steel is at
        yield and the concrete experiences a stress of alpha_1 * f'c. Stores the
        plastic centroid in the class variable
        """

        # initialise the squash load, tensile load and squash moment variables
        squash_load = 0
        tensile_load = 0
        squash_moment_x = 0
        squash_moment_y = 0

        # Gauss points for 1 point Gaussian integration (area)
        gps = gauss_points(n=1)

        # loop through all elements in the mesh
        for element in self.concrete_section.elements:
            mat = element.material  # get material

            # initialise element results
            area_e = 0
            qx_e = 0
            qy_e = 0
            force_e = 0
            force_t_e = 0

            # loop through each Gauss point
            for gp in gps:
                # determine shape function and jacobian
                (N, _, j) = shape_function(coords=element.coords, gauss_point=gp)

                # get coordinates of the gauss point
                x = np.dot(N, np.transpose(element.coords[0, :]))
                y = np.dot(N, np.transpose(element.coords[1, :]))

                # calculate area properties
                area_e += gp[0] * j
                qx_e += gp[0] * y * j
                qy_e += gp[0] * x * j

                # calculate force
                if isinstance(mat, Concrete):
                    force_e += gp[0] * j * mat.alpha_1 * mat.compressive_strength
                elif isinstance(mat, Steel):
                    n = gp[0] * j * mat.yield_strength
                    force_e += n
                    force_t_e -= n

            # calculate element centroid
            cx_e, cy_e = qy_e / area_e, qx_e / area_e

            # add to totals
            squash_load += force_e
            tensile_load += force_t_e
            squash_moment_x += force_e * cx_e
            squash_moment_y += force_e * cy_e

        # store squash load, tensile load and plastic centroid
        self.squash_load = squash_load
        self.tensile_load = tensile_load
        self.axial_pc = [squash_moment_x / squash_load, squash_moment_y / squash_load]

    def get_pc_local(
        self,
        theta: float,
    ) -> Tuple[float, float]:
        """Returns the plastic centroid location in local coordinates.

        :param float theta: Angle the neutral axis makes with the horizontal axis

        :return: Plastic centroid in local coordinates `(pc_u, pc_v)`
        :rtype: Tuple[float, float]
        """

        return principal_coordinate(
            phi=theta * 180 / np.pi, x=self.axial_pc[0], y=self.axial_pc[1]
        )

    def moment_interaction_diagram(
        self,
        theta: float,
        n_points: int = 24,
        n_scale: float = 1e-3,
        m_scale: float = 1e-6,
        plot: bool = True,
        **kwargs,
    ) -> Tuple[List[float], List[float]]:
        """Generates a moment interaction diagram given a neutral axis angle `theta`
        and `n_points` calculation points between the decompression case and the pure
        bending case.

        :param float theta: Angle the neutral axis makes with the horizontal axis
        :param int n_points: Number of calculation points between the decompression
            case and the pure bending case.
        :param float n_scale: Scaling factor to apply to axial force
        :param float m_scale: Scaling factor to apply to bending moment
        :param bool plot: If set to true, displays a plot of the moment interaction
            diagram
        :param \**kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: A list of the points on the moment interaction diagram `(n, m)`
        :rtype: Tuple[List[float], List[float]]
        """

        # initialise variables
        n_curve = []
        m_curve = []

        # add squash load
        n_curve.append(self.squash_load * n_scale)
        m_curve.append(0)

        # compute extreme tensile fibre
        _, d_t = self.calculate_extreme_fibre(theta=theta)

        # compute neutral axis depth for pure bending case
        _, _, _, _, d_nb = self.ultimate_bending_capacity(theta=theta, n=0)

        # generate list of neutral axes
        d_n_list = np.linspace(start=d_t, stop=d_nb, num=n_points)

        # generate progress bar
        with prog_bar.IncrementalBar(
            message="Generating M-N diagram...",
            max=n_points,
            suffix="%(percent)d%% [ %(elapsed)ds ]",
        ) as progress_bar:
            # loop through each neutral axis and calculate actions
            for d_n in d_n_list:
                n, _, _, mv = self.calculate_section_actions(d_n=d_n, theta=theta)
                n_curve.append(n * n_scale)
                m_curve.append(mv * m_scale)
                progress_bar.next()

        # add tensile load
        n_curve.append(self.tensile_load * n_scale)
        m_curve.append(0)

        if plot:
            self.plot_moment_interaction_diagram(
                n_i=[n_curve], m_i=[m_curve], labels=["Concrete Section"], **kwargs
            )

        return n_curve, m_curve

    def plot_moment_interaction_diagram(
        self,
        n_i: List[List[float]],
        m_i: List[List[float]],
        labels: List[str],
        **kwargs,
    ):
        """Plots a number of moment interaction diagrams.

        :param n_i: List containing outputs of axial force from moment interaction
            diagrams.
        :type n_i: List[List[float]]
        :param m_i: List containing outputs of bending moment from moment interaction
            diagrams.
        :type m_i: List[List[float]]
        :param labels: List of labels for each moment interaction diagram
        :type labels: List[str]
        :param \**kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes`
        """

        # create plot and setup the plot
        with plotting_context(title="Moment Interaction Diagram", **kwargs) as (
            fig,
            ax,
        ):
            # for each M-N curve
            for idx in range(len(n_i)):
                ax.plot(m_i[idx], n_i[idx], "o-", label=labels[idx], markersize=3)

            plt.xlabel("Bending Moment")
            plt.ylabel("Axial Force")
            plt.grid(True)

            if idx > 0:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return ax

    def ultimate_bending_capacity(
        self,
        theta: float,
        n: float,
    ) -> Tuple[float, float, float, float, float]:
        """Given a neutral axis angle `theta` and an axial force `n`, calculates the
        ultimate bending capacity.

        :param float theta: Angle the neutral axis makes with the horizontal axis
        :param float n: Axial force

        :return: Axial force, ultimate bending capacity about the x & y axes, resultant
            moment and the depth to the neutral axis `(n, mx, my, mv, d_n)`
        :rtype: Tuple[float, float, float, float, float]
        """

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        _, d_t = self.calculate_extreme_fibre(theta=theta)

        a = 1e-6 * d_t  # sufficiently small depth of compressive zone
        b = d_t  # neutral axis at extreme tensile fibre

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

        n, mx, my, mv = self.calculate_section_actions(d_n=d_n, theta=theta)

        return n, mx, my, mv, d_n

    def normal_force_convergence(
        self,
        d_n: float,
        theta: float,
        n: float,
    ) -> float:
        """Given a neutral axis depth `d_n` and neutral axis angle `theta`, calculates
        the difference between the target net axial force `n` and the axial force
        given `d_n` & `theta`.

        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle the neutral axis makes with the horizontal axis
        :param float n: Target axial force

        :return: Axial force convergence
        :rtype: float
        """

        # calculate convergence
        conv = n - self.calculate_section_actions(d_n=d_n, theta=theta)[0]

        return conv

    def calculate_section_actions(
        self,
        d_n: float,
        theta: float,
    ) -> Tuple[float, float, float, float]:
        """Given a neutral axis depth `d_n` and neutral axis angle `theta`, calculates
        the resultant bending moments `mx`, `my`, `mv` and the net axial force `n`.

        TODO - don't count area of concrete in steel areas! (currently counted)

        :param float d_n: Depth of the neutral axis from the extreme compression fibre,
            0 < d_n <= d_t, where d_t is the depth of the extreme tensile fibre, i.e.
            d_n must be within the section and not equal to zero
        :param float theta: Angle the neutral axis makes with the horizontal axis

        :return: Section actions `(n, mx, my, mv)`
        :rtype: Tuple[float, float, float, float]
        """

        # calculate extreme fibre in global coordinates
        extreme_fibre, d_t = self.calculate_extreme_fibre(theta=theta)

        # validate d_n input
        if d_n <= 0:
            raise ValueError("d_n must be positive.")
        elif d_n > d_t:
            raise ValueError("d_n must lie within the section, i.e. d_n <= d_t")

        # find point on neutral axis by shifting by d_n
        point_na = self.point_on_neutral_axis(
            extreme_fibre=extreme_fibre, d_n=d_n, theta=theta
        )

        # # extract concrete section
        # geom = self.extract_concrete(self.concrete_section.geometry)

        # split the section at the neutral axis
        top_geoms, bot_geoms = self.split_section(
            geometry=self.concrete_section.geometry,
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
            top_geoms = CompoundGeometry(geoms=top_geoms)

            # split the section at the whitney axis
            top_geoms1, top_geoms2 = self.split_section(
                geometry=top_geoms,
                point=pt_whitney,
                theta=theta,
            )

            # combine all top geometries (above and below whitney axis)
            top_geoms = top_geoms1 + top_geoms2

        # combine geometries back into a new CompoundGeometry object
        new_geom = CompoundGeometry(geoms=top_geoms + bot_geoms)

        # generate a mesh (refinement not important)
        new_geom.create_mesh(mesh_sizes=0, coarse=True)

        # create new section object
        new_section = Section(geometry=new_geom)

        # calculate section actions
        n, mv = self.stress_analysis(
            section=new_section, point_na=point_na, d_n=d_n, theta=theta
        )

        # convert mv to mx & my
        (my, mx) = global_coordinate(phi=theta * 180 / np.pi, x11=0, y22=mv)

        return n, mx, my, mv

    def calculate_extreme_fibre(
        self,
        theta: float,
    ) -> Tuple[Tuple[float, float], float]:
        """Calculates the locations of the extreme compression fibre in global
        coordinates given a neutral axis angle `theta`.

        :param float theta: Angle the neutral axis makes with the horizontal axis

        :return: Global coordinate of the extreme compression fibre `(x, y)` and the
            neutral axis depth at the extreme tensile fibre
        :rtype: Tuple[Tuple[float, float], float]
        """

        # loop through all points in the geometry
        for (idx, point) in enumerate(self.concrete_section.geometry.points):
            # determine the coordinate of the point wrt the local axis
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

        # calculate depth of neutral axis at tensile fibre
        d_t = v_max - v_min

        return max_pt, d_t

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

        # determine the coordinate of the point wrt the local axis
        (u, v) = principal_coordinate(
            phi=theta * 180 / np.pi, x=extreme_fibre[0], y=extreme_fibre[1]
        )

        # subtract the neutral axis depth
        v -= d_n

        # convert point back to global coordinates
        return global_coordinate(phi=theta * 180 / np.pi, x11=u, y22=v)

    # def extract_concrete(
    #     self,
    #     geometry: CompoundGeometry,
    # ) -> CompoundGeometry:
    #     """Extracts only the concrete geometries from the cross-section.
    #
    #     :param geometry: Reinforced concrete geometry
    #     :type geometry: :class:`sectionproperties.pre.geometry.CompoundGeometry`
    #
    #     :return: Concrete geometries
    #     :type geometry: :class:`sectionproperties.pre.geometry.CompoundGeometry`
    #     """
    #
    #     geom_idx = 0
    #
    #     for idx, geom in enumerate(geometry.geoms):
    #         if isinstance(geom.material, Concrete):
    #             if geom_idx == 0:
    #                 conc_geoms = geom
    #             else:
    #                 conc_geoms += geom
    #
    #             geom_idx += 1
    #
    #     return conc_geoms

    def split_section(
        self,
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

    def stress_analysis(
        self,
        section: Section,
        point_na: Tuple[float, float],
        d_n: float,
        theta: float,
    ) -> Tuple[float, float]:
        """Calculate the net axial force and moment within the section given the
        netural axis and netural axis rotation.

        :param section: *sectionproperties* section object on which to perform the
            stress analysis
        :type section: :class:`sectionproperties.analysis.section.Section`
        :param point_na: Point on the neutral axis in global coordinates
        :type point_na: Tuple[float, float]
        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle the neutral axis makes with the horizontal axis

        :return: Section actions - axial force and moment about netural axis `(n, mv)`
        :rtype: Tuple[float, float]
        """

        # initialise section actions
        n = 0
        mv = 0

        # Gauss points for 3 point Gaussian integration
        gps = gauss_points(n=3)

        # loop through all elements (concrete & steel)
        for element in section.elements:
            # get current element material
            mat = element.material

            # initialise element results
            area_e = 0
            qx_e = 0
            qy_e = 0
            force_e = 0

            # loop through each Gauss point
            for gp in gps:
                # determine shape function and jacobian
                (N, _, j) = shape_function(coords=element.coords, gauss_point=gp)

                # get coordinates of the gauss point
                x = np.dot(N, np.transpose(element.coords[0, :]))
                y = np.dot(N, np.transpose(element.coords[1, :]))

                # calculate area properties
                area_e += gp[0] * j
                qx_e += gp[0] * y * j
                qy_e += gp[0] * x * j

                # get strain at gauss point
                d, strain = self.get_strain(
                    point=(x, y),
                    point_na=point_na,
                    d_n=d_n,
                    theta=theta,
                    ultimate_strain=self.conc_ultimate_strain,
                )

                # get stress at gauss point
                stress = mat.stress_strain_profile.get_stress(strain=strain)

                # calculate force (stress * area)
                force_e += gp[0] * stress * j

            # calculate element centroid
            cx_e, cy_e = qy_e / area_e, qx_e / area_e

            # convert centroid to local coordinates
            (c_u, c_v) = principal_coordinate(phi=theta * 180 / np.pi, x=cx_e, y=cy_e)

            # add to totals
            n += force_e
            mv += force_e * (c_v - self.get_pc_local(theta=theta)[1])

        return n, mv

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

        # convert point to local coordinates
        (u, v) = principal_coordinate(phi=theta * 180 / np.pi, x=point[0], y=point[1])

        # convert point_na to local coordinates
        (u_na, v_na) = principal_coordinate(
            phi=theta * 180 / np.pi, x=point_na[0], y=point_na[1]
        )

        # calculate distance between NA and point in `v` direction
        d = v - v_na

        return d, d / d_n * ultimate_strain
