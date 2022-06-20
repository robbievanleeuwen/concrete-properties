from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from concreteproperties.material import Concrete, Steel
from concreteproperties.analysis_section import AnalysisSection
import concreteproperties.utils as utils
from concreteproperties.post import plotting_context
import concreteproperties.results as res

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
        self.gross_properties = res.ConcreteProperties()

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
    ) -> res.TransformedConcreteProperties:
        """Transforms gross section properties given a reference elastic modulus.

        :param float elastic_modulus: Reference elastic modulus

        :return: Transformed concrete properties object
        :rtype:
            :class:`~concreteproperties.results.TransformedConcreteProperties`
        """

        return res.TransformedConcreteProperties(
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

    def calculate_cracked_properties(
        self,
        theta: float = 0,
    ) -> res.CrackedResults:
        """Calculates cracked section properties given a neutral axis angle `theta`.

        :param float theta: Neutral axis angle about which bending is taking place

        :return: Cracked results object
        :rtype: :class:`~concreteproperties.results.CrackedResults`
        """

        cracked_results = res.CrackedResults(theta=theta)
        cracked_results.m_cr = self.calculate_cracking_moment(theta=theta)

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(points=self.geometry.points, theta=theta)
        a = 1e-6 * d_t  # sufficiently small depth of compressive zone
        b = d_t  # neutral axis at extreme tensile fibre

        # find neutral axis that gives convergence of the the cracked neutral axis
        (cracked_results.d_nc, r) = brentq(
            f=self.cracked_neutral_axis_convergence,
            a=a,
            b=b,
            args=(cracked_results),
            xtol=1e-3,
            rtol=1e-6,
            full_output=True,
            disp=False,
        )

        # calculate cracked section properties
        # axial rigidity & first moments of area
        for geom in cracked_results.cracked_geometries:
            area = geom.calculate_area()
            centroid = geom.calculate_centroid()

            cracked_results.e_a_cr += area * geom.material.elastic_modulus
            cracked_results.e_qx_cr += (
                area * geom.material.elastic_modulus * centroid[1]
            )
            cracked_results.e_qy_cr += (
                area * geom.material.elastic_modulus * centroid[0]
            )

        # global second moments of area
        for geom in cracked_results.cracked_geometries:
            # if concrete
            if isinstance(geom.material, Concrete):
                conc_sec = AnalysisSection(geometry=geom)

                for conc_el in conc_sec.elements:
                    (
                        el_e_ixx_g,
                        el_e_iyy_g,
                        el_e_ixy_g,
                    ) = conc_el.second_moments_of_area()
                    cracked_results.e_ixx_g_cr += el_e_ixx_g
                    cracked_results.e_iyy_g_cr += el_e_iyy_g
                    cracked_results.e_ixy_g_cr += el_e_ixy_g

            elif isinstance(geom.material, Steel):
                # area, diameter and centroid of geometry
                area = geom.calculate_area()
                diam = np.sqrt(4 * area / np.pi)
                centroid = geom.calculate_centroid()

                cracked_results.e_ixx_g_cr += geom.material.elastic_modulus * (
                    np.pi * pow(diam, 4) / 64 + area * centroid[1] * centroid[1]
                )
                cracked_results.e_iyy_g_cr += geom.material.elastic_modulus * (
                    np.pi * pow(diam, 4) / 64 + area * centroid[0] * centroid[0]
                )
                cracked_results.e_ixy_g_cr += geom.material.elastic_modulus * (
                    area * centroid[0] * centroid[1]
                )

        # centroidal second moments of area
        cracked_results.e_ixx_c_cr = (
            cracked_results.e_ixx_g_cr
            - cracked_results.e_qx_cr**2 / cracked_results.e_a_cr
        )
        cracked_results.e_iyy_c_cr = (
            cracked_results.e_iyy_g_cr
            - cracked_results.e_qy_cr**2 / cracked_results.e_a_cr
        )
        cracked_results.e_ixy_c_cr = (
            cracked_results.e_ixy_g_cr
            - cracked_results.e_qx_cr * cracked_results.e_qy_cr / cracked_results.e_a_cr
        )
        cracked_results.e_iuu_cr = (
            cracked_results.e_iyy_c_cr * (np.sin(theta)) ** 2
            + cracked_results.e_ixx_c_cr * (np.cos(theta)) ** 2
            - 2 * cracked_results.e_ixy_c_cr * np.sin(theta) * np.cos(theta)
        )

        return cracked_results

    def calculate_cracking_moment(
        self,
        theta: float,
    ) -> float:
        """Calculates the cracking moment given a bending angle `theta`.

        :param float theta: Neutral axis angle about which bending is taking place

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

    def cracked_neutral_axis_convergence(
        self,
        d_nc: float,
        cracked_results: res.CrackedResults,
    ) -> float:
        """Given a trial cracked neutral axis depth `d_nc`, determines the difference
        between the first moments of area above and below the trial axis.

        :param float d_nc: Trial cracked neutral axis
        :param cracked_results: Cracked results object
        :type cracked_results: :class:`~concreteproperties.results.CrackedResults`

        :return: Cracked neutral axis convergence
        :rtype: float
        """

        # calculate extreme fibre in global coordinates
        extreme_fibre, d_t = utils.calculate_extreme_fibre(
            points=self.geometry.points, theta=cracked_results.theta
        )

        # validate d_nc input
        if d_nc <= 0:
            raise ValueError("d_nc must be positive.")
        elif d_nc > d_t:
            raise ValueError("d_nc must lie within the section, i.e. d_nc <= d_t")

        # find point on neutral axis by shifting by d_nc
        point_na = utils.point_on_neutral_axis(
            extreme_fibre=extreme_fibre, d_n=d_nc, theta=cracked_results.theta
        )

        # get principal coordinates of neutral axis
        na_local = principal_coordinate(
            phi=cracked_results.theta * 180 / np.pi, x=point_na[0], y=point_na[1]
        )

        # split concrete geometries above and below d_nc, discard below
        cracked_geoms = []

        for conc_geom in self.concrete_geometries:
            top_geoms, _ = utils.split_section(
                geometry=conc_geom,
                point=point_na,
                theta=cracked_results.theta,
            )

            # save compression geometries
            cracked_geoms.extend(top_geoms)

        # determine moment of area equilibrium about neutral axis
        e_qu = 0  # initialise first moment of area

        # add steel geometries to list
        cracked_geoms.extend(self.steel_geometries)

        # concrete
        for conc_geom in cracked_geoms:
            ea = conc_geom.calculate_area() * conc_geom.material.elastic_modulus
            centroid = conc_geom.calculate_centroid()

            # convert centroid to local coordinates
            _, c_v = principal_coordinate(
                phi=cracked_results.theta * 180 / np.pi, x=centroid[0], y=centroid[1]
            )

            # calculate first moment of area
            e_qu += ea * (c_v - na_local[1])

        cracked_results.cracked_geometries = cracked_geoms

        return e_qu

    def moment_curvature_diagram(
        self,
        theta: float = 0,
        kappa_inc: float = 1e-7,
        delta_m_min: float = 0.15,
        delta_m_max: float = 0.3,
    ) -> res.MomentCurvatureResults:
        """Generates a moment curvature diagram given a bending angle `theta`.

        Analysis continues until the steel reaches fracture strain.

        :param float theta: Neutral axis angle about which bending is taking place
        :param float kappa_inc: Initial curvature increment
        :param float delta_m_min: Relative change in moment at which to double step
            size
        :param float delta_m_max: Relative change in moment at which to halve step
            size

        :return: Moment curvature results object
        :rtype: :class:`~concreteproperties.results.MomentCurvatureResults`
        """

        # initiliase variables
        moment_curvature = res.MomentCurvatureResults(theta=theta)
        iter = 0

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(points=self.geometry.points, theta=theta)
        a = 1e-6 * d_t  # sufficiently small depth of compressive zone
        b = d_t  # neutral axis at extreme tensile fibre

        # create progress bar
        with utils.create_unknown_progress() as progress:
            task = progress.add_task(
                description="[red]Generating M-K diagram",
                total=None,
            )

            # while there hasn't been a failure in the steel
            while not moment_curvature._failure:
                # calculate adaptive step size for curvature
                if iter > 1:
                    moment_diff = (
                        abs(moment_curvature.kappa[-1] - moment_curvature.kappa[-2])
                        / moment_curvature.kappa[-1]
                    )
                    if moment_diff <= delta_m_min:
                        kappa_inc *= 2
                    elif moment_diff >= delta_m_max:
                        kappa_inc *= 0.5

                kappa = moment_curvature.kappa[-1] + kappa_inc

                # find neutral axis that gives convergence of the axial force
                (d_n, r) = brentq(
                    f=self.service_normal_force_convergence,
                    a=a,
                    b=b,
                    args=(kappa, moment_curvature),
                    xtol=1e-3,
                    rtol=1e-6,
                    full_output=True,
                    disp=False,
                )
                text_update = "[red]Generating M-K diagram: "
                text_update += f"M={moment_curvature._m_i:.3e}"

                progress.update(task, description=text_update)

                # save results
                moment_curvature.kappa.append(kappa)
                moment_curvature.moment.append(moment_curvature._m_i)
                iter += 1

            progress.update(
                task,
                description="[bold green]:white_check_mark: M-K diagram generated",
            )

        return moment_curvature

    def service_normal_force_convergence(
        self,
        d_n: float,
        kappa: float,
        moment_curvature: res.MomentCurvatureResults,
    ) -> float:
        """Given a trial neutral axis depth `d_n` and curvature `kappa`, determines the
        difference between the net axial force and the desired axial force.

        :param float d_nc: Trial cracked neutral axis
        :param float kappa: Curvature
        :param moment_curvature: Moment curvature results object
        :type moment_curvature:
            :class:`~concreteproperties.results.MomentCurvatureResults`

        :return: Service normal force convergence
        :rtype: float
        """

        # calculate convergence
        return self.calculate_service_section_actions(
            d_n=d_n, kappa=kappa, moment_curvature=moment_curvature
        )._n_i

    def calculate_service_section_actions(
        self,
        d_n: float,
        kappa: float,
        moment_curvature: res.MomentCurvatureResults = res.MomentCurvatureResults(),
    ) -> res.MomentCurvatureResults:
        """Given a neutral axis depth `d_n` and curvature `kappa`, calculates the
        resultant axial force and bending moment.

        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float kappa: Curvature
        :param moment_curvature: Moment curvature results object
        :type moment_curvature:
            :class:`~concreteproperties.results.MomentCurvatureResults`

        :return: Moment curvature results object
        :rtype: :class:`~concreteproperties.results.MomentCurvatureResults`
        """

        # calculate extreme fibre in global coordinates
        extreme_fibre, d_t = utils.calculate_extreme_fibre(
            points=self.geometry.points, theta=moment_curvature.theta
        )

        # validate d_n input
        if d_n <= 0:
            raise ValueError("d_n must be positive.")
        elif d_n > d_t:
            raise ValueError("d_n must lie within the section, i.e. d_n <= d_t")

        # find point on neutral axis by shifting by d_n
        point_na = utils.point_on_neutral_axis(
            extreme_fibre=extreme_fibre, d_n=d_n, theta=moment_curvature.theta
        )

        # get principal coordinates of neutral axis
        na_local = principal_coordinate(
            phi=moment_curvature.theta * 180 / np.pi, x=point_na[0], y=point_na[1]
        )

        # create splits in concrete geometries at points in stress strain profiles
        concrete_split_geoms = []

        for conc_geom in self.concrete_geometries:
            strains = conc_geom.material.stress_strain_profile.get_unique_strains()

            # loop through intermediate points on stress strain profile
            for idx, strain in enumerate(strains[1:-1]):
                # depth to point with `strain` from NA
                d = strain / kappa

                # convert depth to global coordinates
                dx, dy = global_coordinate(
                    phi=moment_curvature.theta * 180 / np.pi, x11=0, y22=d
                )

                # calculate location of point with `strain`
                pt = point_na[0] + dx, point_na[1] + dy

                # split concrete geometry (from bottom up)
                top_geoms, bot_geoms = utils.split_section(
                    geometry=conc_geom,
                    point=pt,
                    theta=moment_curvature.theta,
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
            n_sec, mv_sec = sec.service_stress_analysis(
                point_na=point_na,
                d_n=d_n,
                theta=moment_curvature.theta,
                kappa=kappa,
                na_local=na_local[1],
            )

            n += n_sec
            mv += mv_sec

        # calculate steel actions
        for steel_geom in self.steel_geometries:
            # calculate area and centroid
            area = steel_geom.calculate_area()
            centroid = steel_geom.calculate_centroid()

            # get strain at centroid of steel
            strain = utils.get_service_strain(
                point=(centroid[0], centroid[1]),
                point_na=point_na,
                theta=moment_curvature.theta,
                kappa=kappa,
            )

            # check for steel failure
            if (
                abs(strain)
                > steel_geom.material.stress_strain_profile.get_ultimate_strain()
            ):
                moment_curvature._failure = True

            # calculate stress and force
            stress = steel_geom.material.stress_strain_profile.get_stress(strain=strain)
            force = stress * area
            n += force

            # convert centroid to local coordinates
            _, c_v = principal_coordinate(
                phi=moment_curvature.theta * 180 / np.pi, x=centroid[0], y=centroid[1]
            )

            # calculate moment
            mv += force * (c_v - na_local[1])

        moment_curvature._n_i = n
        moment_curvature._m_i = mv

        return moment_curvature

    def ultimate_bending_capacity(
        self,
        theta: float = 0,
        n: float = 0,
    ) -> results.UltimateBendingResults:
        """Given a neutral axis angle `theta` and an axial force `n`, calculates the
        ultimate bending capacity.

        :param float theta: Angle the neutral axis makes with the horizontal axis
        :param float n: Net axial force

        :return: Ultimate bending results object
        :rtype: :class:`~concreteproperties.results.UltimateBendingResults`
        """

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(points=self.geometry.points, theta=theta)
        a = 1e-6 * d_t  # sufficiently small depth of compressive zone
        b = d_t  # neutral axis at extreme tensile fibre

        # initialise ultimate bending results
        ultimate_results = res.UltimateBendingResults()

        # find neutral axis that gives convergence of the axial force
        (d_n, r) = brentq(
            f=self.ultimate_normal_force_convergence,
            a=a,
            b=b,
            args=(n, ultimate_results),
            xtol=1e-3,
            rtol=1e-6,
            full_output=True,
            disp=False,
        )

        return ultimate_results

    def ultimate_normal_force_convergence(
        self,
        d_n: float,
        n: float,
        ultimate_results: results.UltimateBendingResults,
    ) -> float:
        """Given a neutral axis depth `d_n` and neutral axis angle `theta`, calculates
        the difference between the target net axial force `n` and the axial force
        given `d_n` & `theta`.

        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float n: Net axial force
        :param ultimate_results: Ultimate bending results object
        :type ultimate_results:
            :class:`~concreteproperties.results.UltimateBendingResults`

        :return: Axial force convergence
        :rtype: float
        """

        # calculate convergence
        return (
            n
            - self.calculate_ultimate_section_actions(
                d_n=d_n, ultimate_results=ultimate_results
            ).n
        )

    def calculate_ultimate_section_actions(
        self,
        d_n: float,
        ultimate_results: results.UltimateBendingResults,
    ) -> results.UltimateBendingResults:
        """Given a neutral axis depth `d_n` and neutral axis angle `theta`, calculates
        the resultant bending moments `mx`, `my`, `mv` and the net axial force `n`.

        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param ultimate_results: Ultimate bending results object
        :type ultimate_results:
            :class:`~concreteproperties.results.UltimateBendingResults`

        :return: Ultimate bending results object
        :rtype: :class:`~concreteproperties.results.UltimateBendingResults`
        """

        # calculate extreme fibre in global coordinates
        extreme_fibre, d_t = utils.calculate_extreme_fibre(
            points=self.geometry.points, theta=ultimate_results.theta
        )

        # validate d_n input
        if d_n <= 0:
            raise ValueError("d_n must be positive.")
        elif d_n > d_t:
            raise ValueError("d_n must lie within the section, i.e. d_n <= d_t")

        # find point on neutral axis by shifting by d_n
        point_na = utils.point_on_neutral_axis(
            extreme_fibre=extreme_fibre, d_n=d_n, theta=ultimate_results.theta
        )

        # get principal coordinates of plastic centroid
        pc_local = self.get_pc_local(theta=ultimate_results.theta)

        # create splits in concrete geometries at points in stress strain profiles
        concrete_split_geoms = []

        for conc_geom in self.concrete_geometries:
            strains = (
                conc_geom.material.ultimate_stress_strain_profile.get_unique_strains()
            )

            # loop through intermediate points on stress strain profile
            for idx, strain in enumerate(strains[1:-1]):
                # depth to point with `strain` from NA
                d = strain / self.gross_properties.conc_ultimate_strain * d_n

                # convert depth to global coordinates
                dx, dy = global_coordinate(
                    phi=ultimate_results.theta * 180 / np.pi, x11=0, y22=d
                )

                # calculate location of point with `strain`
                pt = point_na[0] + dx, point_na[1] + dy

                # split concrete geometry (from bottom up)
                top_geoms, bot_geoms = utils.split_section(
                    geometry=conc_geom,
                    point=pt,
                    theta=ultimate_results.theta,
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
                theta=ultimate_results.theta,
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
            strain = utils.get_ultimate_strain(
                point=(centroid[0], centroid[1]),
                point_na=point_na,
                d_n=d_n,
                theta=ultimate_results.theta,
                ultimate_strain=self.gross_properties.conc_ultimate_strain,
            )

            # calculate stress and force
            stress = steel_geom.material.stress_strain_profile.get_stress(strain=strain)
            force = stress * area
            n += force

            # convert centroid to local coordinates
            _, c_v = principal_coordinate(
                phi=ultimate_results.theta * 180 / np.pi, x=centroid[0], y=centroid[1]
            )

            # calculate moment
            mv += force * (c_v - pc_local[1])

        # convert mv to mx & my
        (my, mx) = global_coordinate(
            phi=ultimate_results.theta * 180 / np.pi, x11=0, y22=mv
        )

        # save results
        ultimate_results.d_n = d_n
        ultimate_results.n = n
        ultimate_results.mx = mx
        ultimate_results.my = my
        ultimate_results.mv = mv

        return ultimate_results

    def moment_interaction_diagram(
        self,
        theta: float = 0,
        n_points: int = 24,
    ) -> res.MomentInteractionResults:
        """Generates a moment interaction diagram given a neutral axis angle `theta`
        and `n_points` calculation points between the decompression case and the pure
        bending case.

        :param float theta: Angle the neutral axis makes with the horizontal axis
        :param int n_points: Number of calculation points between the decompression
            case and the pure bending case.

        :return: Moment interaction results object
        :rtype: :class:`~concreteproperties.results.MomentInteractionResults`
        """

        # initialise results
        mi_results = res.MomentInteractionResults()

        # add squash load
        mi_results.n.append(self.gross_properties.squash_load)
        mi_results.m.append(0)

        # compute extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(points=self.geometry.points, theta=theta)

        # compute neutral axis depth for pure bending case
        ultimate_results = self.ultimate_bending_capacity(theta=theta, n=0)

        # generate list of neutral axes
        d_n_list = np.linspace(start=d_t, stop=ultimate_results.d_n, num=n_points)

        # create progress bar
        with utils.create_known_progress() as progress:
            task = progress.add_task(
                description="[red]Generating M-N diagram",
                total=n_points,
            )

            for d_n in d_n_list:
                ultimate_results = self.calculate_ultimate_section_actions(
                    d_n=d_n, ultimate_results=ultimate_results
                )
                mi_results.n.append(ultimate_results.n)
                mi_results.m.append(ultimate_results.mv)
                progress.update(task, advance=1)

            progress.update(
                task,
                description="[bold green]:white_check_mark: M-N diagram generated",
            )

        # add tensile load
        mi_results.n.append(self.gross_properties.tensile_load)
        mi_results.m.append(0)

        return mi_results

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
