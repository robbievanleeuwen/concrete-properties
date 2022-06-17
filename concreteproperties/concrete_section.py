from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from concreteproperties.material import Concrete, Steel
from concreteproperties.analysis_section import AnalysisSection
import concreteproperties.utils as utils
from concreteproperties.post import plotting_context

from sectionproperties.pre.geometry import CompoundGeometry
from sectionproperties.analysis.fea import principal_coordinate, global_coordinate

if TYPE_CHECKING:
    import matplotlib.axes

from rich.pretty import pprint


class ConcreteSection:
    """Class for a reinforced concrete section."""

    def __init__(
        self,
        geometry: CompoundGeometry,
    ):
        """Inits the ConcreteSection class.

        :param geometry: *sectionproperties* compound geometry object describing the
            reinforced concrete section
        :type geometry: :class:`sectionproperties.pre.geometry.CompoundGeometry`
        """

        self.geometry = geometry

        # sort into concrete and steel geometries
        self.concrete_geometries = []
        self.steel_geometries = []

        for geom in self.geometry.geoms:
            if isinstance(geom.material, Concrete):
                self.concrete_geometries.append(geom)
            if isinstance(geom.material, Steel):
                self.steel_geometries.append(geom)

        # validate reinforced concrete input
        if len(self.concrete_geometries) == 0 or len(self.steel_geometries) == 0:
            raise ValueError(
                "geometry must contain both Concrete and Steel geometries."
            )

        # initialise gross properties results class
        self.gross_properties = ConcreteProperties()

        # calculate gross area properties
        self.calculate_gross_area_properties()

        # calculate gross plastic properties
        self.calculate_gross_plastic_properties()

    def calculate_gross_area_properties(
        self,
    ):
        """Calculates and stores gross section area properties."""

        # concrete areas
        for conc_geom in self.concrete_geometries:
            # area and centroid of geometry
            area = conc_geom.calculate_area()
            centroid = conc_geom.calculate_centroid()

            self.gross_properties.concrete_area += area
            self.gross_properties.e_a += area * conc_geom.material.elastic_modulus
            self.gross_properties.mass += area * conc_geom.material.density
            self.gross_properties.e_qx += (
                area * conc_geom.material.elastic_modulus * centroid[1]
            )
            self.gross_properties.e_qy += (
                area * conc_geom.material.elastic_modulus * centroid[0]
            )

        # steel area
        for steel_geom in self.steel_geometries:
            # area and centroid of geometry
            area = steel_geom.calculate_area()
            centroid = steel_geom.calculate_centroid()

            self.gross_properties.steel_area += area
            self.gross_properties.e_a += area * steel_geom.material.elastic_modulus
            self.gross_properties.mass += area * steel_geom.material.density
            self.gross_properties.e_qx += (
                area * steel_geom.material.elastic_modulus * centroid[1]
            )
            self.gross_properties.e_qy += (
                area * steel_geom.material.elastic_modulus * centroid[0]
            )

        # total area
        self.gross_properties.total_area = (
            self.gross_properties.concrete_area + self.gross_properties.steel_area
        )

        # perimeter
        self.gross_properties.perimeter = self.geometry.calculate_perimeter()

        # centroids
        self.gross_properties.cx = (
            self.gross_properties.e_qy / self.gross_properties.e_a
        )
        self.gross_properties.cy = (
            self.gross_properties.e_qx / self.gross_properties.e_a
        )

        # global second moments of area
        # concrete geometries
        for conc_geom in self.concrete_geometries:
            conc_sec = AnalysisSection(geometry=conc_geom)

            for conc_el in conc_sec.elements:
                el_e_ixx_g, el_e_iyy_g, el_e_ixy_g = conc_el.second_moments_of_area()
                self.gross_properties.e_ixx_g += el_e_ixx_g
                self.gross_properties.e_iyy_g += el_e_iyy_g
                self.gross_properties.e_ixy_g += el_e_ixy_g

        # steel geometries
        for steel_geom in self.steel_geometries:
            # area, diameter and centroid of geometry
            area = steel_geom.calculate_area()
            diam = np.sqrt(4 * area / np.pi)
            centroid = steel_geom.calculate_centroid()

            self.gross_properties.e_ixx_g += steel_geom.material.elastic_modulus * (
                np.pi * pow(diam, 4) / 64 + area * centroid[1] * centroid[1]
            )
            self.gross_properties.e_iyy_g += steel_geom.material.elastic_modulus * (
                np.pi * pow(diam, 4) / 64 + area * centroid[0] * centroid[0]
            )
            self.gross_properties.e_ixy_g += steel_geom.material.elastic_modulus * (
                area * centroid[0] * centroid[1]
            )

        # centroidal second moments of area
        self.gross_properties.e_ixx_c = (
            self.gross_properties.e_ixx_g
            - self.gross_properties.e_qx**2 / self.gross_properties.e_a
        )
        self.gross_properties.e_iyy_c = (
            self.gross_properties.e_iyy_g
            - self.gross_properties.e_qy**2 / self.gross_properties.e_a
        )
        self.gross_properties.e_ixy_c = (
            self.gross_properties.e_ixy_g
            - self.gross_properties.e_qx
            * self.gross_properties.e_qy
            / self.gross_properties.e_a
        )

        # principal 2nd moments of area about the centroidal xy axis
        Delta = (
            ((self.gross_properties.e_ixx_c - self.gross_properties.e_iyy_c) / 2) ** 2
            + self.gross_properties.e_ixy_c**2
        ) ** 0.5
        self.gross_properties.e_i11_c = (
            self.gross_properties.e_ixx_c + self.gross_properties.e_iyy_c
        ) / 2 + Delta
        self.gross_properties.e_i22_c = (
            self.gross_properties.e_ixx_c + self.gross_properties.e_iyy_c
        ) / 2 - Delta

        # principal axis angle
        if (
            abs(self.gross_properties.e_ixx_c - self.gross_properties.e_i11_c)
            < 1e-12 * self.gross_properties.e_i11_c
        ):
            self.gross_properties.phi = 0
        else:
            self.gross_properties.phi = (
                np.arctan2(
                    self.gross_properties.e_ixx_c - self.gross_properties.e_i11_c,
                    self.gross_properties.e_ixy_c,
                )
                * 180
                / np.pi
            )

        # centroidal section moduli
        x_min, x_max, y_min, y_max = self.geometry.calculate_extents()
        self.gross_properties.e_zxx_plus = self.gross_properties.e_ixx_c / abs(
            y_max - self.gross_properties.cy
        )
        self.gross_properties.e_zxx_minus = self.gross_properties.e_ixx_c / abs(
            y_min - self.gross_properties.cy
        )
        self.gross_properties.e_zyy_plus = self.gross_properties.e_iyy_c / abs(
            x_max - self.gross_properties.cx
        )
        self.gross_properties.e_zyy_minus = self.gross_properties.e_iyy_c / abs(
            x_min - self.gross_properties.cx
        )

        # principal section moduli
        x11_max, x11_min, y22_max, y22_min = utils.calculate_local_extents(
            geometry=self.geometry,
            cx=self.gross_properties.cx,
            cy=self.gross_properties.cy,
            theta=self.gross_properties.phi,
        )

        # evaluate principal section moduli
        self.gross_properties.e_z11_plus = self.gross_properties.e_i11_c / abs(y22_max)
        self.gross_properties.e_z11_minus = self.gross_properties.e_i11_c / abs(y22_min)
        self.gross_properties.e_z22_plus = self.gross_properties.e_i22_c / abs(x11_max)
        self.gross_properties.e_z22_minus = self.gross_properties.e_i22_c / abs(x11_min)

    def get_transformed_gross_properties(
        self,
        elastic_modulus: float,
    ) -> TransformedConcreteProperties:
        """Transforms gross section properties given a reference elastic modulus.

        :param float elastic_modulus: Reference elastic modulus

        :return: Transformed concrete properties object
        :rtype:
            :class:`~concreteproperties.concrete_section.TransformedConcreteProperties`
        """

        return TransformedConcreteProperties(
            concrete_properties=self.gross_properties, elastic_modulus=elastic_modulus
        )

    def calculate_gross_plastic_properties(
        self,
    ):
        """Calculates and stores gross section plastic properties.

        Calculates the plastic centroid and squash load assuming all steel is at yield
        and the concrete experiences a stress of alpha_1 * f'c.

        Calculates tensile load assuming all steel is at yield and the concrete is
        entirely cracked.
        """

        # initialise the squash load, tensile load and squash moment variables
        squash_load = 0
        tensile_load = 0
        squash_moment_x = 0
        squash_moment_y = 0

        # loop through all concrete geometries
        for conc_geom in self.concrete_geometries:
            # calculate area and centroid
            area = conc_geom.calculate_area()
            centroid = conc_geom.calculate_centroid()

            # calculate compressive force
            force_c = (
                area
                * conc_geom.material.alpha_1
                * conc_geom.material.ultimate_stress_strain_profile.get_compressive_strength()
            )

            # add to totals
            squash_load += force_c
            squash_moment_x += force_c * centroid[0]
            squash_moment_y += force_c * centroid[1]

        # loop through all steel geometries
        for steel_geom in self.steel_geometries:
            # calculate area and centroid
            area = steel_geom.calculate_area()
            centroid = steel_geom.calculate_centroid()

            # calculate compressive and tensile force
            force_c = area * steel_geom.material.yield_strength
            force_t = -force_c

            # add to totals
            squash_load += force_c
            tensile_load += force_t
            squash_moment_x += force_c * centroid[0]
            squash_moment_y += force_c * centroid[1]

        # store squash load, tensile load and plastic centroid
        self.gross_properties.squash_load = squash_load
        self.gross_properties.tensile_load = tensile_load
        self.gross_properties.axial_pc_x = squash_moment_x / squash_load
        self.gross_properties.axial_pc_y = squash_moment_y / squash_load

        # store ultimate concrete strain (get from first concrete geometry)
        # note this MUST not vary between different concrete materials
        self.gross_properties.conc_ultimate_strain = self.concrete_geometries[
            0
        ].material.ultimate_stress_strain_profile.get_ultimate_strain()

    def calculate_cracking_moment(
        self,
        theta: float = 0,
    ) -> float:
        """Calculates the cracking moment given a bending angle `theta`.

        :param float theta: Angle the bending axis makes with the horizontal axis

        :return: Cracking moment
        :rtype: float
        """

        # get centroidal second moments of area
        e_ixx = self.gross_properties.e_ixx_c
        e_iyy = self.gross_properties.e_iyy_c
        e_ixy = self.gross_properties.e_ixy_c

        # determine rotated second moment of area
        e_iuu = (
            e_iyy * (np.sin(theta)) ** 2
            + e_ixx * (np.cos(theta)) ** 2
            - 2 * e_ixy * np.sin(theta) * np.cos(theta)
        )

        # loop through all concrete geometries to find lowest cracking moment
        for idx, conc_geom in enumerate(self.concrete_geometries):
            # get distance from centroid to extreme tensile fibre
            d = utils.calculate_max_bending_depth(
                points=conc_geom.points,
                c_local_v=self.get_c_local(theta=theta)[1],
                theta=theta,
            )

            # if no part of the section is in tension, go to next geometry
            if d == 0:
                continue

            # cracking moment for this geometry
            f_t = (
                conc_geom.material.flexural_tensile_strength
                - conc_geom.material.residual_shrinkage_stress
            )
            m_c_geom = (f_t / conc_geom.material.elastic_modulus) * (e_iuu / d)

            # if first geometry, initialise cracking moment
            if idx == 0:
                m_c = m_c_geom
            # otherwise take smallest cracking moment
            else:
                m_c = min(m_c, m_c_geom)

        return m_c

    def calculate_cracked_properties(
        self,
    ):
        pass

    def ultimate_bending_capacity(
        self,
        theta: float = 0,
        n: float = 0,
    ) -> Tuple[float]:
        """Given a neutral axis angle `theta` and an axial force `n`, calculates the
        ultimate bending capacity.

        :param float theta: Angle the neutral axis makes with the horizontal axis
        :param float n: Axial force

        :return: Axial force, ultimate bending capacity about the x & y axes, resultant
            moment and the depth to the neutral axis `(n, mx, my, mv, d_n)`
        :rtype: Tuple[float]
        """

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(points=self.geometry.points, theta=theta)

        a = 1e-6 * d_t  # sufficiently small depth of compressive zone
        b = d_t  # neutral axis at extreme tensile fibre

        # initialise ultimate bending results
        self._ult_bend_res = [None, None, None, None]

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

        # unpack ultimate bending results from last run of brentq
        n, mx, my, mv = self._ult_bend_res

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
        conv = n - self.calculate_ultimate_section_actions(d_n=d_n, theta=theta)[0]

        return conv

    def calculate_ultimate_section_actions(
        self,
        d_n: float,
        theta: float,
    ) -> Tuple[float]:
        """Given a neutral axis depth `d_n` and neutral axis angle `theta`, calculates
        the resultant bending moments `mx`, `my`, `mv` and the net axial force `n`.

        :param float d_n: Depth of the neutral axis from the extreme compression fibre,
            0 < d_n <= d_t, where d_t is the depth of the extreme tensile fibre, i.e.
            d_n must be within the section and not equal to zero
        :param float theta: Angle the neutral axis makes with the horizontal axis

        :return: Section actions `(n, mx, my, mv)`
        :rtype: Tuple[float]
        """

        # calculate extreme fibre in global coordinates
        extreme_fibre, d_t = utils.calculate_extreme_fibre(
            points=self.geometry.points, theta=theta
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

        # create splits in concrete geometries at points in stress strain profiles
        concrete_split_geoms = []

        for conc_geom in self.concrete_geometries:
            strains = (
                conc_geom.material.ultimate_stress_strain_profile.get_unique_strains()
            )

            # loop through intermediate points on stress strain profile
            for idx, strain in enumerate(strains[1:-1]):
                pt = utils.get_point_from_strain(
                    strain=strain,
                    point_na=point_na,
                    d_n=d_n,
                    theta=theta,
                    ultimate_strain=self.gross_properties.conc_ultimate_strain,
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
            sec = AnalysisSection(geometry=conc_geom)
            n_sec, mv_sec = sec.ultimate_stress_analysis(
                point_na=point_na,
                d_n=d_n,
                theta=theta,
                ultimate_strain=self.gross_properties.conc_ultimate_strain,
                pc_local=pc_local[1],
            )

            n += n_sec
            mv += mv_sec

        # calculate steel actions
        for steel_geom in self.steel_geometries:
            # calculate area and centroid
            area = steel_geom.calculate_area()
            centroid = steel_geom.calculate_centroid()

            # get strain at centroid of steel
            strain = utils.get_strain(
                point=(centroid[0], centroid[1]),
                point_na=point_na,
                d_n=d_n,
                theta=theta,
                ultimate_strain=self.gross_properties.conc_ultimate_strain,
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

        # save results
        self._ult_bend_res = [n, mx, my, mv]

        return n, mx, my, mv

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
        n_curve.append(self.gross_properties.squash_load * n_scale)
        m_curve.append(0)

        # compute extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(points=self.geometry.points, theta=theta)

        # compute neutral axis depth for pure bending case
        _, _, _, _, d_nb = self.ultimate_bending_capacity(theta=theta, n=0)

        # generate list of neutral axes
        d_n_list = np.linspace(start=d_t, stop=d_nb, num=n_points)

        # create progress bar
        with utils.create_progress() as progress:
            task = progress.add_task(
                description="[red]Generating M-N diagram",
                total=n_points,
            )

            for d_n in d_n_list:
                n, _, _, mv = self.calculate_ultimate_section_actions(
                    d_n=d_n, theta=theta
                )
                n_curve.append(n * n_scale)
                m_curve.append(mv * m_scale)
                progress.update(task, advance=1)

            progress.update(
                task,
                description="[bold green]:white_check_mark: M-N diagram generated",
            )

        # add tensile load
        n_curve.append(self.gross_properties.tensile_load * n_scale)
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
    ) -> matplotlib.axes._subplots.AxesSubplot:
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
        :rtype: :class:`matplotlib.axes._subplots.AxesSubplot`
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

            # if there is more than one curve show legend
            if idx > 0:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return ax

    def get_c_local(
        self,
        theta: float,
    ) -> Tuple[float]:
        """Returns the elastic centroid location in local coordinates.

        :param float theta: Angle the neutral axis makes with the horizontal axis

        :return: Elastic centroid in local coordinates `(c_u, c_v)`
        :rtype: Tuple[float]
        """

        return principal_coordinate(
            phi=theta * 180 / np.pi,
            x=self.gross_properties.cx,
            y=self.gross_properties.cy,
        )

    def get_pc_local(
        self,
        theta: float,
    ) -> Tuple[float]:
        """Returns the plastic centroid location in local coordinates.

        :param float theta: Angle the neutral axis makes with the horizontal axis

        :return: Plastic centroid in local coordinates `(pc_u, pc_v)`
        :rtype: Tuple[float]
        """

        return principal_coordinate(
            phi=theta * 180 / np.pi,
            x=self.gross_properties.axial_pc_x,
            y=self.gross_properties.axial_pc_y,
        )


@dataclass
class ConcreteProperties:
    """Class for storing gross concrete section properties."""

    # section areas
    total_area: float = 0
    concrete_area: float = 0
    steel_area: float = 0
    e_a: float = 0

    # section mass
    mass: float = 0

    # section perimeter
    perimeter: float = 0

    # first moments of area
    e_qx: float = 0
    e_qy: float = 0

    # centroids
    cx: float = 0
    cy: float = 0

    # second moments of area
    e_ixx_g: float = 0
    e_iyy_g: float = 0
    e_ixy_g: float = 0
    e_ixx_c: float = 0
    e_iyy_c: float = 0
    e_ixy_c: float = 0
    e_i11_c: float = 0
    e_i22_c: float = 0

    # principal axis angle
    phi: float = 0

    # section moduli
    e_zxx_plus: float = 0
    e_zxx_minus: float = 0
    e_zyy_plus: float = 0
    e_zyy_minus: float = 0
    e_z11_plus: float = 0
    e_z11_minus: float = 0
    e_z22_plus: float = 0
    e_z22_minus: float = 0

    # plastic properties
    squash_load: float = 0
    tensile_load: float = 0
    axial_pc_x: float = 0
    axial_pc_y: float = 0
    conc_ultimate_strain: float = 0


@dataclass
class TransformedConcreteProperties:
    """Class for storing transformed gross concrete section properties.

    :param concrete_properties: Concrete properties object
    :type concrete_properties:
        :class:`~concreteproperties.concrete_section.ConcreteProperties`
    :param float elastic_modulus: Reference elastic modulus
    """

    concrete_properties: ConcreteProperties = field(repr=False)
    elastic_modulus: float

    # first moments of area
    qx: float = 0
    qy: float = 0

    # second moments of area
    ixx_g: float = 0
    iyy_g: float = 0
    ixy_g: float = 0
    ixx_c: float = 0
    iyy_c: float = 0
    ixy_c: float = 0
    i11_c: float = 0
    i22_c: float = 0

    # section moduli
    zxx_plus: float = 0
    zxx_minus: float = 0
    zyy_plus: float = 0
    zyy_minus: float = 0
    z11_plus: float = 0
    z11_minus: float = 0
    z22_plus: float = 0
    z22_minus: float = 0

    def __post_init__(
        self,
    ):

        self.qx = self.concrete_properties.e_qx / self.elastic_modulus
        self.qy = self.concrete_properties.e_qy / self.elastic_modulus
        self.ixx_g = self.concrete_properties.e_ixx_g / self.elastic_modulus
        self.iyy_g = self.concrete_properties.e_iyy_g / self.elastic_modulus
        self.ixy_g = self.concrete_properties.e_ixy_g / self.elastic_modulus
        self.ixx_c = self.concrete_properties.e_ixx_c / self.elastic_modulus
        self.iyy_c = self.concrete_properties.e_iyy_c / self.elastic_modulus
        self.ixy_c = self.concrete_properties.e_ixy_c / self.elastic_modulus
        self.i11_c = self.concrete_properties.e_i11_c / self.elastic_modulus
        self.i22_c = self.concrete_properties.e_i22_c / self.elastic_modulus
        self.zxx_plus = self.concrete_properties.e_zxx_plus / self.elastic_modulus
        self.zxx_minus = self.concrete_properties.e_zxx_minus / self.elastic_modulus
        self.zyy_plus = self.concrete_properties.e_zyy_plus / self.elastic_modulus
        self.zyy_minus = self.concrete_properties.e_zyy_minus / self.elastic_modulus
        self.z11_plus = self.concrete_properties.e_z11_plus / self.elastic_modulus
        self.z11_minus = self.concrete_properties.e_z11_minus / self.elastic_modulus
        self.z22_plus = self.concrete_properties.e_z22_plus / self.elastic_modulus
        self.z22_minus = self.concrete_properties.e_z22_minus / self.elastic_modulus
