from typing import List, Tuple
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from concreteproperties.material import Concrete, Steel
from concreteproperties.analysis_section import AnalysisSection
import concreteproperties.utils as utils
from concreteproperties.post import plotting_context

from sectionproperties.pre.geometry import CompoundGeometry
from sectionproperties.analysis.fea import principal_coordinate, global_coordinate

import progress.bar as prog_bar
from rich.pretty import pprint


class ConcreteSection:
    """Class for a reinforced concrete section."""

    def __init__(
        self,
        concrete_geometry: CompoundGeometry,
    ):
        """Inits the ConcreteSection class.

        :param concrete_section: *sectionproperties* compound geometry object describing
            the reinforced concrete section
        :type concrete_section: :class:`sectionproperties.pre.geometry.CompoundGeometry`
        """

        self.concrete_geometry = concrete_geometry

        # initialise class variables
        self.squash_load = 0  # squash load (positive = compression)
        self.tensile_load = 0  # tension load (negative = tension)
        self.axial_pc = [0, 0]  # axial plastic centroid (global coordinates)

        # calculate the plastic centroid (& squash load)
        self.calculate_plastic_centroid()

        # check there is concrete & steel, and assign ultimate concrete strain
        conc = False
        steel = False

        for geom in self.concrete_geometry.geoms:
            if isinstance(geom.material, Concrete):
                conc = True
                self.conc_ultimate_strain = (
                    geom.material.stress_strain_profile.get_ultimate_strain()
                )
            if isinstance(geom.material, Steel):
                steel = True

        if not conc or not steel:
            raise ValueError("Geometry must contain Concrete and Steel.")

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

        # loop through all geometries in the CompoundGeometry
        for geom in self.concrete_geometry.geoms:
            mat = geom.material  # get material
            area = geom.calculate_area()  # calculate area
            centroid = geom.calculate_centroid()  # calculate centroid

            # calculate plastic forces
            if isinstance(mat, Concrete):
                force_c = area * mat.alpha_1 * mat.compressive_strength
                force_t = 0
            elif isinstance(mat, Steel):
                force_c = area * mat.yield_strength
                force_t = -force_c
            else:
                raise ValueError("Material is not Concrete or Steel!")

            # add to totals
            squash_load += force_c
            tensile_load += force_t
            squash_moment_x += force_c * centroid[0]
            squash_moment_y += force_c * centroid[1]

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
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

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
        _, d_t = utils.calculate_extreme_fibre(
            points=self.concrete_geometry.points, theta=theta
        )

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
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

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
        _, d_t = utils.calculate_extreme_fibre(
            points=self.concrete_geometry.points, theta=theta
        )

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
        extreme_fibre, d_t = utils.calculate_extreme_fibre(
            points=self.concrete_geometry.points, theta=theta
        )

        # validate d_n input
        if d_n <= 0:
            raise ValueError("d_n must be positive.")
        elif d_n > d_t:
            raise ValueError("d_n must lie within the section, i.e. d_n <= d_t")

        # find point on neutral axis by shifting by d_n
        point_na = utils.point_on_neutral_axis(
            extreme_fibre=extreme_fibre, d_n=d_n, theta=theta
        )

        # get principal coordinates of plastic centroid
        pc_local = self.get_pc_local(theta=theta)

        # get all concrete & steel geometries
        concrete_geoms = []
        steel_geoms = []

        for geom in self.concrete_geometry.geoms:
            if isinstance(geom.material, Concrete):
                concrete_geoms.append(geom)

            if isinstance(geom.material, Steel):
                steel_geoms.append(geom)

        # create splits in concrete geometries at points in stress strain profiles
        concrete_split_geoms = []

        for conc_geom in concrete_geoms:
            strains = conc_geom.material.stress_strain_profile.get_unique_strains()

            # loop through intermediate points on stress strain profile
            for idx, strain in enumerate(strains[1:-1]):
                pt = utils.get_point_from_strain(
                    strain=strain,
                    point_na=point_na,
                    d_n=d_n,
                    theta=theta,
                    ultimate_strain=self.conc_ultimate_strain,
                )

                # split concrete geometry (from bottom up)
                top_geoms, bot_geoms = utils.split_section(
                    geometry=conc_geom,
                    point=pt,
                    theta=theta,
                )

                # save bottom geoms
                concrete_split_geoms.extend(bot_geoms)

                # continue to split top geoms
                conc_geom = CompoundGeometry(geoms=top_geoms)

            # save final top geoms
            concrete_split_geoms.extend(top_geoms)

        # initialise results
        n = 0
        mv = 0

        # calculate concrete actions
        for conc_geom in concrete_split_geoms:
            sec = AnalysisSection(conc_geom)
            n_sec, mv_sec = sec.ultimate_stress_analysis(
                point_na=point_na,
                d_n=d_n,
                theta=theta,
                ultimate_strain=self.conc_ultimate_strain,
                pc_local=pc_local[1],
            )

            n += n_sec
            mv += mv_sec

        # calculate steel actions
        for steel_geom in steel_geoms:
            # calculate area and centroid
            area = steel_geom.calculate_area()
            centroid = steel_geom.calculate_centroid()

            # get strain at centroid of steel
            strain = utils.get_strain(
                point=(centroid[0], centroid[1]),
                point_na=point_na,
                d_n=d_n,
                theta=theta,
                ultimate_strain=self.conc_ultimate_strain,
            )

            # calculate stress and force
            stress = steel_geom.material.stress_strain_profile.get_stress(strain=strain)
            force = stress * area
            n += force

            # convert centroid to local coordinates
            _, c_v = principal_coordinate(
                phi=theta * 180 / np.pi, x=centroid[0], y=centroid[1]
            )

            # calculate moment
            mv += force * (c_v - pc_local[1])

        # convert mv to mx & my
        (my, mx) = global_coordinate(phi=theta * 180 / np.pi, x11=0, y22=mv)

        return n, mx, my, mv
