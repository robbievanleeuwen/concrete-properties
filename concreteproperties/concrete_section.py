from __future__ import annotations

from typing import List, Tuple, Optional, TYPE_CHECKING
import warnings
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rich.live import Live

from concreteproperties.material import Concrete, Steel
from concreteproperties.analysis_section import AnalysisSection
import concreteproperties.utils as utils
from concreteproperties.post import plotting_context
import concreteproperties.results as res

import sectionproperties.pre.geometry as sp_geom
from sectionproperties.analysis.fea import principal_coordinate, global_coordinate

if TYPE_CHECKING:
    import matplotlib


class ConcreteSection:
    """Class for a reinforced concrete section."""

    def __init__(
        self,
        geometry: sp_geom.CompoundGeometry,
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

        # check overlapping regions
        polygons = [sec_geom.geom for sec_geom in self.geometry.geoms]
        overlapped_regions = sp_geom.check_geometry_overlaps(polygons)
        if overlapped_regions:
            warnings.warn(
                "The provided geometry contains overlapping regions, results may be incorrect."
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
        self.gross_properties.e_i11 = (
            self.gross_properties.e_ixx_c + self.gross_properties.e_iyy_c
        ) / 2 + Delta
        self.gross_properties.e_i22 = (
            self.gross_properties.e_ixx_c + self.gross_properties.e_iyy_c
        ) / 2 - Delta

        # principal axis angle
        if (
            abs(self.gross_properties.e_ixx_c - self.gross_properties.e_i11)
            < 1e-12 * self.gross_properties.e_i11
        ):
            self.gross_properties.phi = 0
        else:
            self.gross_properties.phi = (
                np.arctan2(
                    self.gross_properties.e_ixx_c - self.gross_properties.e_i11,
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
        self.gross_properties.e_z11_plus = self.gross_properties.e_i11 / abs(y22_max)
        self.gross_properties.e_z11_minus = self.gross_properties.e_i11 / abs(y22_min)
        self.gross_properties.e_z22_plus = self.gross_properties.e_i22 / abs(x11_max)
        self.gross_properties.e_z22_minus = self.gross_properties.e_i22 / abs(x11_min)

    def calculate_gross_plastic_properties(
        self,
        max_compressive_strain: Optional[float] = None,
    ):
        """Calculates and stores gross section plastic properties.

        Calculates the plastic centroid and squash load assuming all steel is at yield
        and the concrete experiences a stress of alpha_squash * f'c. Providing
        max_compressive_strain limits the compressive strain in the steel.

        Calculates tensile load assuming all steel is at yield and the concrete is
        entirely cracked.

        :param max_compressive_strain: Maximum compressive strain in the steel under
            squash load
        :type max_compressive_strain: Optional[float]
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
                * conc_geom.material.alpha_squash
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
            if max_compressive_strain:
                force_c = area * steel_geom.material.stress_strain_profile.get_stress(
                    strain=max_compressive_strain
                )
            else:
                force_c = (
                    area * steel_geom.material.stress_strain_profile.yield_strength
                )

            force_t = -area * steel_geom.material.stress_strain_profile.yield_strength

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

        # store ultimate concrete strain (get smallest from all concrete geometries)
        conc_ult_strain = 0

        for idx, conc_geom in enumerate(self.concrete_geometries):
            ult_strain = (
                conc_geom.material.ultimate_stress_strain_profile.get_ultimate_strain()
            )
            if idx == 0:
                conc_ult_strain = ult_strain
            else:
                conc_ult_strain = min(conc_ult_strain, ult_strain)

        self.gross_properties.conc_ultimate_strain = conc_ult_strain

    def get_gross_properties(
        self,
    ) -> res.ConcreteProperties:
        """Returns the gross section properties of the reinforced concrete section.

        :return: Concrete properties object
        :rtype: :class:`~concreteproperties.results.ConcreteProperties`
        """

        return self.gross_properties

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

    def calculate_cracked_properties(
        self,
        theta: Optional[float] = 0,
    ) -> res.CrackedResults:
        r"""Calculates cracked section properties given a neutral axis angle `theta`.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :type theta: Optional[float]

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
        try:
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
        except ValueError:
            warnings.warn("brentq algorithm failed.")

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

        # centroids
        cracked_results.cx = cracked_results.e_qy_cr / cracked_results.e_a_cr
        cracked_results.cy = cracked_results.e_qx_cr / cracked_results.e_a_cr

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

        # principal 2nd moments of area about the centroidal xy axis
        Delta = (
            ((cracked_results.e_ixx_c_cr - cracked_results.e_iyy_c_cr) / 2) ** 2
            + cracked_results.e_ixy_c_cr**2
        ) ** 0.5
        cracked_results.e_i11_cr = (
            cracked_results.e_ixx_c_cr + cracked_results.e_iyy_c_cr
        ) / 2 + Delta
        cracked_results.e_i22_cr = (
            cracked_results.e_ixx_c_cr + cracked_results.e_iyy_c_cr
        ) / 2 - Delta

        # principal axis angle
        if (
            abs(cracked_results.e_ixx_c_cr - cracked_results.e_i11_cr)
            < 1e-12 * cracked_results.e_i11_cr
        ):
            cracked_results.phi_cr = 0
        else:
            cracked_results.phi_cr = (
                np.arctan2(
                    cracked_results.e_ixx_c_cr - cracked_results.e_i11_cr,
                    cracked_results.e_ixy_c_cr,
                )
                * 180
                / np.pi
            )

        return cracked_results

    def calculate_cracking_moment(
        self,
        theta: float,
    ) -> float:
        r"""Calculates the cracking moment given a bending angle `theta`.

        :param float theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)

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
            f_t = conc_geom.material.flexural_tensile_strength
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

        # concrete & steel
        for geom in cracked_geoms:
            ea = geom.calculate_area() * geom.material.elastic_modulus
            centroid = geom.calculate_centroid()

            # convert centroid to local coordinates
            _, c_v = principal_coordinate(
                phi=cracked_results.theta * 180 / np.pi, x=centroid[0], y=centroid[1]
            )

            # calculate first moment of area
            e_qu += ea * (c_v - na_local[1])

        cracked_results.cracked_geometries = cracked_geoms

        return e_qu

    def moment_curvature_analysis(
        self,
        theta: Optional[float] = 0,
        kappa_inc: Optional[float] = 1e-7,
        delta_m_min: Optional[float] = 0.15,
        delta_m_max: Optional[float] = 0.3,
    ) -> res.MomentCurvatureResults:
        r"""Performs a moment curvature analysis given a bending angle `theta`.

        Analysis continues until the steel reaches fracture strain or the concrete
        reaches its ultimate strain.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :type theta: Optional[float]
        :param kappa_inc: Initial curvature increment
        :type kappa_inc: Optional[float]
        :param delta_m_min: Relative change in moment at which to double step
        :type delta_m_min: Optional[float]
        :param delta_m_max: Relative change in moment at which to halve step
        :type delta_m_max: Optional[float]

        :return: Moment curvature results object
        :rtype: :class:`~concreteproperties.results.MomentCurvatureResults`
        """

        # initialise variables
        moment_curvature = res.MomentCurvatureResults(theta=theta)
        iter = 0

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(points=self.geometry.points, theta=theta)
        a = 1e-6 * d_t  # sufficiently small depth of compressive zone
        b = d_t  # neutral axis at extreme tensile fibre

        # create progress bar
        progress = utils.create_unknown_progress()

        with Live(progress, refresh_per_second=10) as live:
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
                try:
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
                except ValueError:
                    warnings.warn("brentq algorithm failed.")

                text_update = "[red]Generating M-K diagram: "
                text_update += f"M={moment_curvature._m_i:.3e}"

                progress.update(task, description=text_update)

                # save results
                if not moment_curvature._failure:
                    moment_curvature.kappa.append(kappa)
                    moment_curvature.moment.append(moment_curvature._m_i)
                    iter += 1

            progress.update(
                task,
                description="[bold green]:white_check_mark: M-K diagram generated",
            )
            live.refresh()

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
        moment_curvature: Optional[
            res.MomentCurvatureResults
        ] = res.MomentCurvatureResults(theta=0),
    ) -> res.MomentCurvatureResults:
        """Given a neutral axis depth `d_n` and curvature `kappa`, calculates the
        resultant axial force and bending moment.

        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float kappa: Curvature
        :param moment_curvature: Moment curvature results object
        :type moment_curvature:
            Optional[:class:`~concreteproperties.results.MomentCurvatureResults`]

        :return: Moment curvature results object
        :rtype: :class:`~concreteproperties.results.MomentCurvatureResults`
        """

        # reset failure
        moment_curvature._failure = False

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

        # create splits in concrete geometries at points in stress-strain profiles
        concrete_split_geoms = utils.split_section_at_strains(
            concrete_geometries=self.concrete_geometries,
            theta=moment_curvature.theta,
            point_na=point_na,
            ultimate=False,
            kappa=kappa,
        )

        # initialise results
        n = 0
        m_u = 0

        # calculate concrete actions
        for conc_geom in concrete_split_geoms:
            sec = AnalysisSection(geometry=conc_geom)
            n_sec, m_u_sec, max_strain = sec.service_stress_analysis(
                point_na=point_na,
                d_n=d_n,
                theta=moment_curvature.theta,
                kappa=kappa,
                na_local=na_local[1],
            )

            n += n_sec
            m_u += m_u_sec

            # check for concrete failure
            if (
                max_strain
                > conc_geom.material.stress_strain_profile.get_ultimate_strain()
            ):
                moment_curvature._failure = True
                moment_curvature.failure_geometry = conc_geom

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
                moment_curvature.failure_geometry = steel_geom

            # calculate stress and force
            stress = steel_geom.material.stress_strain_profile.get_stress(strain=strain)
            force = stress * area
            n += force

            # convert centroid to local coordinates
            _, c_v = principal_coordinate(
                phi=moment_curvature.theta * 180 / np.pi, x=centroid[0], y=centroid[1]
            )

            # calculate moment
            m_u += force * (c_v - na_local[1])

        moment_curvature._n_i = n
        moment_curvature._m_i = m_u

        return moment_curvature

    def ultimate_bending_capacity(
        self,
        theta: Optional[float] = 0,
        n: Optional[float] = 0,
    ) -> res.UltimateBendingResults:
        r"""Given a neutral axis angle `theta` and an axial force `n`, calculates the
        ultimate bending capacity.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :type theta: Optional[float]
        :param n: Net axial force
        :type n: Optional[float]

        :return: Ultimate bending results object
        :rtype: :class:`~concreteproperties.results.UltimateBendingResults`
        """

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(points=self.geometry.points, theta=theta)
        a = 1e-6 * d_t  # sufficiently small depth of compressive zone
        b = d_t  # neutral axis at extreme tensile fibre

        # initialise ultimate bending results
        ultimate_results = res.UltimateBendingResults(theta=theta)

        # find neutral axis that gives convergence of the axial force
        try:
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
        except ValueError:
            warnings.warn("brentq algorithm failed.")

        return ultimate_results

    def ultimate_normal_force_convergence(
        self,
        d_n: float,
        n: float,
        ultimate_results: res.UltimateBendingResults,
    ) -> float:
        """Given a neutral axis depth `d_n` and neutral axis angle `theta`, calculates
        the difference between the target net axial force `n` and the axial force.

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
        ultimate_results: Optional[res.UltimateBendingResults] = None,
    ) -> res.UltimateBendingResults:
        """Given a neutral axis depth `d_n` and neutral axis angle `theta`, calculates
        the resultant bending moments `m_x`, `m_y`, `m_u` and the net axial force `n`.

        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param ultimate_results: Ultimate bending results object
        :type ultimate_results:
            Optional[:class:`~concreteproperties.results.UltimateBendingResults`]

        :return: Ultimate bending results object
        :rtype: :class:`~concreteproperties.results.UltimateBendingResults`
        """

        if ultimate_results is None:
            ultimate_results = res.UltimateBendingResults(theta=0)

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

        # create splits in concrete geometries at points in stress-strain profiles
        concrete_split_geoms = utils.split_section_at_strains(
            concrete_geometries=self.concrete_geometries,
            theta=ultimate_results.theta,
            point_na=point_na,
            ultimate=True,
            ultimate_strain=self.gross_properties.conc_ultimate_strain,
            d_n=d_n,
        )

        # initialise results
        n = 0
        m_u = 0
        k_u = []

        # calculate concrete actions
        for conc_geom in concrete_split_geoms:
            sec = AnalysisSection(geometry=conc_geom)
            n_sec, m_u_sec = sec.ultimate_stress_analysis(
                point_na=point_na,
                d_n=d_n,
                theta=ultimate_results.theta,
                ultimate_strain=self.gross_properties.conc_ultimate_strain,
                pc_local=pc_local[1],
            )

            n += n_sec
            m_u += m_u_sec

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
            m_u += force * (c_v - pc_local[1])

            # calculate k_u
            _, ef_v = principal_coordinate(
                phi=ultimate_results.theta * 180 / np.pi,
                x=extreme_fibre[0],
                y=extreme_fibre[1],
            )
            d = ef_v - c_v
            k_u.append(d_n / d)

        # convert m_u to m_x & m_y
        (m_y, m_x) = global_coordinate(
            phi=ultimate_results.theta * 180 / np.pi, x11=0, y22=m_u
        )

        # save results
        ultimate_results.d_n = d_n
        ultimate_results.k_u = min(k_u)
        ultimate_results.n = n
        ultimate_results.m_x = m_x
        ultimate_results.m_y = m_y
        ultimate_results.m_u = m_u

        return ultimate_results

    def moment_interaction_diagram(
        self,
        theta: Optional[float] = 0,
        m_neg: Optional[bool] = False,
        n_points: Optional[int] = 24,
    ) -> res.MomentInteractionResults:
        r"""Generates a moment interaction diagram given a neutral axis angle `theta`
        and `n_points` calculation points between the decompression case and the pure
        bending case.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :type theta: Optional[float]
        :param m_neg: If set to True, also calculates the moment interaction for
            :math:`\theta = \theta + \pi`, i.e. sagging and hogging
        :type m_neg: Optional[bool]
        :param n_points: Number of calculation points between the decompression point
            and the pure bending point
        :type n_points: Optional[int]

        :return: Moment interaction results object
        :rtype: :class:`~concreteproperties.results.MomentInteractionResults`
        """

        # initialise results
        mi_results = res.MomentInteractionResults()

        # add squash load
        mi_results.results.append(
            res.UltimateBendingResults(
                theta=theta,
                d_n=None,
                k_u=None,
                n=self.gross_properties.squash_load,
                m_x=0,
                m_y=0,
                m_u=0,
            )
        )

        # compute extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(points=self.geometry.points, theta=theta)

        # compute neutral axis depth for pure bending case
        ult_res_pure = self.ultimate_bending_capacity(theta=theta, n=0)

        # generate list of neutral axes
        d_n_list = np.linspace(start=d_t, stop=ult_res_pure.d_n, num=n_points)

        # create progress bar
        progress = utils.create_known_progress()

        with Live(progress, refresh_per_second=10) as live:
            progress_length = n_points

            if m_neg:
                progress_length *= 2

            task = progress.add_task(
                description="[red]Generating M-N diagram",
                total=progress_length,
            )

            for d_n in d_n_list:
                ult_res = self.calculate_ultimate_section_actions(
                    d_n=d_n, ultimate_results=res.UltimateBendingResults(theta=theta)
                )
                mi_results.results.append(ult_res)
                progress.update(task, advance=1)

            if not m_neg:
                progress.update(
                    task,
                    description="[bold green]:white_check_mark: M-N diagram generated",
                )
                live.refresh()

            # add tensile load
            mi_results.results.append(
                res.UltimateBendingResults(
                    theta=theta,
                    d_n=None,
                    k_u=0,
                    n=self.gross_properties.tensile_load,
                    m_x=0,
                    m_y=0,
                    m_u=0,
                )
            )

            # if not calculating negative bending
            if not m_neg:
                return mi_results

            # negative bending
            theta += np.pi

            if theta > np.pi:
                theta -= 2 * np.pi

            # add squash load
            mi_results.results_neg.append(
                res.UltimateBendingResults(
                    theta=theta,
                    d_n=None,
                    k_u=None,
                    n=self.gross_properties.squash_load,
                    m_x=0,
                    m_y=0,
                    m_u=0,
                )
            )

            # compute extreme tensile fibre
            _, d_t = utils.calculate_extreme_fibre(
                points=self.geometry.points, theta=theta
            )

            # compute neutral axis depth for pure bending case
            ult_res_pure = self.ultimate_bending_capacity(theta=theta, n=0)

            # generate list of neutral axes
            d_n_list = np.linspace(start=d_t, stop=ult_res_pure.d_n, num=n_points)

            for d_n in d_n_list:
                ult_res = self.calculate_ultimate_section_actions(
                    d_n=d_n,
                    ultimate_results=res.UltimateBendingResults(theta=theta),
                )
                # bending moment is negative
                ult_res.m_u *= -1
                mi_results.results_neg.append(ult_res)
                progress.update(task, advance=1)

            progress.update(
                task,
                description="[bold green]:white_check_mark: M-N diagram generated",
            )
            live.refresh()

            # add tensile load
            mi_results.results_neg.append(
                res.UltimateBendingResults(
                    theta=theta,
                    d_n=None,
                    k_u=0,
                    n=self.gross_properties.tensile_load,
                    m_x=0,
                    m_y=0,
                    m_u=0,
                )
            )

        return mi_results

    def biaxial_bending_diagram(
        self,
        n: Optional[float] = 0,
        n_points: Optional[int] = 48,
    ) -> res.BiaxialBendingResults:
        """Generates a biaxial bending diagram given a net axial force `n` and
        `n_points` calculation points.

        :param n: Net axial force
        :type n: Optional[float]
        :param n_points: Number of calculation points between the decompression
        :type n_points: Optional[int]

        :return: Biaxial bending results
        :rtype: :class:`~concreteproperties.results.BiaxialBendingResults`
        """

        # initialise results
        bb_results = res.BiaxialBendingResults(n=n)

        # calculate d_theta
        d_theta = 2 * np.pi / n_points

        # generate list of thetas
        theta_list = np.linspace(start=-np.pi, stop=np.pi - d_theta, num=n_points)

        # create progress bar
        progress = utils.create_known_progress()

        with Live(progress, refresh_per_second=10) as live:
            task = progress.add_task(
                description="[red]Generating biaxial bending diagram",
                total=n_points,
            )

            # loop through thetas
            for theta in theta_list:
                ultimate_results = self.ultimate_bending_capacity(theta=theta, n=n)
                bb_results.results.append(ultimate_results)
                progress.update(task, advance=1)

            # add first result to end of list top
            bb_results.results.append(bb_results.results[0])

            progress.update(
                task,
                description="[bold green]:white_check_mark: Biaxial bending diagram generated",
            )
            live.refresh()

        return bb_results

    def calculate_uncracked_stress(
        self,
        n: Optional[float] = 0,
        m_x: Optional[float] = 0,
        m_y: Optional[float] = 0,
    ) -> res.StressResult:
        """Calculates stresses within the reinforced concrete section assuming an
        uncracked section.

        Uses gross area section properties to determine concrete and steel stresses
        given an axial force `n`, and bending moments `m_x` and `m_y`.

        :param n: Axial force
        :type n: Optional[float]
        :param m_x: Bending moment about the x-axis
        :type m_x: Optional[float]
        :param m_y: Bending moment about the y-axis
        :type m_y: Optional[float]

        :return: Stress results object
        :rtype: :class:`~concreteproperties.results.StressResult`
        """

        # initialise stress results
        analysis_sections = []
        conc_sigs = []
        conc_forces = []
        steel_sigs = []
        steel_strains = []
        steel_forces = []

        # get uncracked section properties
        e_a = self.gross_properties.e_a
        cx = self.gross_properties.cx
        cy = self.gross_properties.cy
        e_ixx = self.gross_properties.e_ixx_c
        e_iyy = self.gross_properties.e_iyy_c
        e_ixy = self.gross_properties.e_ixy_c

        # calculate neutral axis rotation
        grad = (e_ixy * m_x - e_ixx * m_y) / (e_iyy * m_x - e_ixy * m_y)
        theta = np.arctan2(grad, 1)

        if np.isclose(theta, 0):
            theta = 0

        # point on neutral axis is centroid
        point_na = (cx, cy)

        # split concrete geometries above and below neutral axis
        split_conc_geoms = []

        for conc_geom in self.concrete_geometries:
            top_geoms, bot_geoms = utils.split_section(
                geometry=conc_geom,
                point=point_na,
                theta=theta,
            )

            split_conc_geoms.extend(top_geoms)
            split_conc_geoms.extend(bot_geoms)

        # loop through all concrete geometries and calculate stress
        for conc_geom in split_conc_geoms:
            analysis_section = AnalysisSection(geometry=conc_geom)

            # calculate stress, force and point of action
            sig, n_conc, d_x, d_y = analysis_section.get_elastic_stress(
                n=n,
                m_x=m_x,
                m_y=m_y,
                e_a=e_a,
                cx=cx,
                cy=cy,
                e_ixx=e_ixx,
                e_iyy=e_iyy,
                e_ixy=e_ixy,
            )
            conc_sigs.append(sig)
            conc_forces.append((n_conc, d_x, d_y))

            # save analysis section
            analysis_sections.append(analysis_section)

        # loop through all steel geometries and calculate stress
        for steel_geom in self.steel_geometries:
            # initialise stress and position of bar
            sig = 0
            centroid = steel_geom.calculate_centroid()
            x = centroid[0] - cx
            y = centroid[1] - cy

            # axial stress
            sig += n * steel_geom.material.elastic_modulus / e_a

            # bending moment stress
            sig += steel_geom.material.elastic_modulus * (
                -(e_ixy * m_x) / (e_ixx * e_iyy - e_ixy**2) * x
                + (e_iyy * m_x) / (e_ixx * e_iyy - e_ixy**2) * y
            )
            sig += steel_geom.material.elastic_modulus * (
                +(e_ixx * m_y) / (e_ixx * e_iyy - e_ixy**2) * x
                - (e_ixy * m_y) / (e_ixx * e_iyy - e_ixy**2) * y
            )
            strain = sig / steel_geom.material.elastic_modulus

            # net force and point of action
            n_steel = sig * steel_geom.calculate_area()

            steel_sigs.append(sig)
            steel_strains.append(strain)
            steel_forces.append((n_steel, x, y))

        return res.StressResult(
            concrete_section=self,
            concrete_analysis_sections=analysis_sections,
            concrete_stresses=conc_sigs,
            concrete_forces=conc_forces,
            steel_geometries=self.steel_geometries,
            steel_stresses=steel_sigs,
            steel_strains=steel_strains,
            steel_forces=steel_forces,
        )

    def calculate_cracked_stress(
        self,
        cracked_results: res.CrackedResults,
        n: Optional[float] = 0,
        m: Optional[float] = 0,
    ) -> res.StressResult:
        """Calculates stresses within the reinforced concrete section assuming a cracked
        section.

        Uses cracked area section properties to determine concrete and steel stresses
        given an axial force `n` and bending moment `m` about the bending axis stored
        in `cracked_results`.

        :param cracked_results: Cracked results objects
        :type cracked_results: :class:`~concreteproperties.results.CrackedResults`
        :param n: Axial force
        :type n: Optional[float]
        :param m: Bending moment
        :type m: Optional[float]

        :return: Stress results object
        :rtype: :class:`~concreteproperties.results.StressResult`
        """

        # initialise stress results
        analysis_sections = []
        conc_sigs = []
        conc_forces = []
        steel_sigs = []
        steel_strains = []
        steel_forces = []

        # get cracked section properties
        e_a = cracked_results.e_a_cr
        cx = cracked_results.cx
        cy = cracked_results.cy
        e_ixx = cracked_results.e_ixx_c_cr
        e_iyy = cracked_results.e_iyy_c_cr
        e_ixy = cracked_results.e_ixy_c_cr

        # correct small e_ixy sign error
        if abs(e_ixy / cracked_results.e_i11_cr) < 1e-12:
            e_ixy = 0

        # get bending angle
        theta = cracked_results.theta

        # handle cardinal points (avoid divide by zeros)
        tan_theta = np.tan(theta)
        with np.errstate(divide="ignore"):
            c = (e_ixx - e_ixy * tan_theta) / (e_ixy - e_iyy * tan_theta)

        # calculate bending moment about each axis (figure out signs)
        if theta <= 0:
            if c < 0:
                sign = -1
            elif c > 0:
                sign = 1
        else:
            if c < 0:
                sign = 1
            else:
                sign = -1

        m_x = sign * np.sqrt(m * m / (1 + 1 / (c * c)))
        m_y = m_x / c

        # depth of neutral axis at extreme tensile fibre
        extreme_fibre, d_t = utils.calculate_extreme_fibre(
            points=self.geometry.points, theta=theta
        )

        # find point on neutral axis by shifting by d_n
        point_na = utils.point_on_neutral_axis(
            extreme_fibre=extreme_fibre, d_n=cracked_results.d_nc, theta=theta
        )

        # get principal coordinates of neutral axis
        na_local = principal_coordinate(
            phi=theta * 180 / np.pi, x=point_na[0], y=point_na[1]
        )

        # loop through all concrete geometries and calculate stress
        for geom in cracked_results.cracked_geometries:
            if isinstance(geom.material, Concrete):
                analysis_section = AnalysisSection(geometry=geom)

                # calculate stress, force and point of action
                sig, n_conc, d_x, d_y = analysis_section.get_elastic_stress(
                    n=n,
                    m_x=m_x,
                    m_y=m_y,
                    e_a=e_a,
                    cx=cx,
                    cy=cy,
                    e_ixx=e_ixx,
                    e_iyy=e_iyy,
                    e_ixy=e_ixy,
                )
                conc_sigs.append(sig)
                conc_forces.append((n_conc, d_x, d_y))

                # save analysis section
                analysis_sections.append(analysis_section)

        # loop through all steel geometries and calculate stress
        for steel_geom in self.steel_geometries:
            # initialise stress and position of bar
            sig = 0
            centroid = steel_geom.calculate_centroid()
            x = centroid[0] - cx
            y = centroid[1] - cy

            # axial stress
            sig += n * steel_geom.material.elastic_modulus / e_a

            # bending moment stress
            sig += steel_geom.material.elastic_modulus * (
                -(e_ixy * m_x) / (e_ixx * e_iyy - e_ixy**2) * x
                + (e_iyy * m_x) / (e_ixx * e_iyy - e_ixy**2) * y
            )
            sig += steel_geom.material.elastic_modulus * (
                +(e_ixx * m_y) / (e_ixx * e_iyy - e_ixy**2) * x
                - (e_ixy * m_y) / (e_ixx * e_iyy - e_ixy**2) * y
            )
            strain = sig / steel_geom.material.elastic_modulus

            # net force and point of action
            n_steel = sig * steel_geom.calculate_area()
            # _, c_v = principal_coordinate(
            #     phi=theta * 180 / np.pi, x=centroid[0], y=centroid[1]
            # )
            # d = c_v - na_local[1]

            steel_sigs.append(sig)
            steel_strains.append(strain)
            steel_forces.append((n_steel, x, y))

        return res.StressResult(
            concrete_section=self,
            concrete_analysis_sections=analysis_sections,
            concrete_stresses=conc_sigs,
            concrete_forces=conc_forces,
            steel_geometries=self.steel_geometries,
            steel_stresses=steel_sigs,
            steel_strains=steel_strains,
            steel_forces=steel_forces,
        )

    def calculate_service_stress(
        self,
        moment_curvature_results: res.MomentCurvatureResults,
        m: float,
        kappa: Optional[float] = None,
    ) -> res.StressResult:
        """Calculates service stresses within the reinforced concrete section.

        Uses linear interpolation of the moment-curvature results to determine the
        curvature of the section given the user supplied moment, and thus the stresses
        within the section. Otherwise, can provided a curvature which overrides the
        supplied moment.

        :param moment_curvature_results: Moment-curvature results objects
        :type moment_curvature_results:
            :class:`~concreteproperties.results.MomentCurvatureResults`
        :param float m: Bending moment
        :param kappa: Curvature, if provided overrides the supplied bending moment and
            plots the stress at the given curvature
        :type kappa: Optional[float]

        :return: Stress results object
        :rtype: :class:`~concreteproperties.results.StressResult`
        """

        if kappa is None:
            # get curvature
            kappa = moment_curvature_results.get_curvature(moment=m)

        # get theta
        theta = moment_curvature_results.theta

        # initialise variables
        mk = res.MomentCurvatureResults(theta=theta)

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        extreme_fibre, d_t = utils.calculate_extreme_fibre(
            points=self.geometry.points, theta=theta
        )
        a = 1e-6 * d_t  # sufficiently small depth of compressive zone
        b = d_t  # neutral axis at extreme tensile fibre

        # find neutral axis that gives convergence of the axial force
        try:
            (d_n, r) = brentq(
                f=self.service_normal_force_convergence,
                a=a,
                b=b,
                args=(kappa, mk),
                xtol=1e-3,
                rtol=1e-6,
                full_output=True,
                disp=False,
            )
        except ValueError:
            warnings.warn("brentq algorithm failed.")

        # find point on neutral axis by shifting by d_n
        point_na = utils.point_on_neutral_axis(
            extreme_fibre=extreme_fibre, d_n=d_n, theta=theta
        )

        # get principal coordinates of neutral axis
        na_local = principal_coordinate(
            phi=theta * 180 / np.pi, x=point_na[0], y=point_na[1]
        )

        # initialise stress results
        analysis_sections = []
        conc_sigs = []
        conc_forces = []
        steel_sigs = []
        steel_strains = []
        steel_forces = []

        # create splits in concrete geometries at points in stress-strain profiles
        concrete_split_geoms = utils.split_section_at_strains(
            concrete_geometries=self.concrete_geometries,
            theta=theta,
            point_na=point_na,
            ultimate=False,
            kappa=kappa,
        )

        # loop through all concrete geometries and calculate stress
        for geom in concrete_split_geoms:
            analysis_section = AnalysisSection(geometry=geom)

            # calculate stress, force and point of action
            sig, n_conc, d = analysis_section.get_service_stress(
                d_n=d_n,
                kappa=kappa,
                point_na=point_na,
                theta=theta,
                na_local=na_local[1],
            )
            conc_sigs.append(sig)
            conc_forces.append((n_conc, d, 0))

            # save analysis section
            analysis_sections.append(analysis_section)

        # loop through all steel geometries and calculate stress
        for steel_geom in self.steel_geometries:
            # get position of bar
            centroid = steel_geom.calculate_centroid()

            # get strain at centroid of steel
            strain = utils.get_service_strain(
                point=(centroid[0], centroid[1]),
                point_na=point_na,
                theta=theta,
                kappa=kappa,
            )

            # calculate stress, force and point of action
            sig = steel_geom.material.stress_strain_profile.get_stress(strain=strain)
            n_steel = sig * steel_geom.calculate_area()
            _, c_v = principal_coordinate(
                phi=theta * 180 / np.pi, x=centroid[0], y=centroid[1]
            )
            d = c_v - na_local[1]

            steel_sigs.append(sig)
            steel_strains.append(strain)
            steel_forces.append((n_steel, d, 0))

        return res.StressResult(
            concrete_section=self,
            concrete_analysis_sections=analysis_sections,
            concrete_stresses=conc_sigs,
            concrete_forces=conc_forces,
            steel_geometries=self.steel_geometries,
            steel_stresses=steel_sigs,
            steel_strains=steel_strains,
            steel_forces=steel_forces,
        )

    def calculate_ultimate_stress(
        self,
        ultimate_results: res.UltimateBendingResults,
    ) -> res.StressResult:
        """Calculates ultimate stresses within the reinforced concrete section.

        :param ultimate_results: Ultimate bending results objects
        :type ultimate_results:
            :class:`~concreteproperties.results.UltimateBendingResults`

        :return: Stress results object
        :rtype: :class:`~concreteproperties.results.StressResult`
        """

        # depth of neutral axis at extreme tensile fibre
        extreme_fibre, d_t = utils.calculate_extreme_fibre(
            points=self.geometry.points, theta=ultimate_results.theta
        )

        # find point on neutral axis by shifting by d_n
        point_na = utils.point_on_neutral_axis(
            extreme_fibre=extreme_fibre,
            d_n=ultimate_results.d_n,
            theta=ultimate_results.theta,
        )

        # get principal coordinates of neutral axis
        na_local = principal_coordinate(
            phi=ultimate_results.theta * 180 / np.pi, x=point_na[0], y=point_na[1]
        )

        # get principal coordinates of plastic centroid
        pc_local = self.get_pc_local(theta=ultimate_results.theta)

        # initialise stress results for each concrete geometry
        analysis_sections = []
        conc_sigs = []
        conc_forces = []
        steel_sigs = []
        steel_strains = []
        steel_forces = []

        # create splits in concrete geometries at points in stress-strain profiles
        concrete_split_geoms = utils.split_section_at_strains(
            concrete_geometries=self.concrete_geometries,
            theta=ultimate_results.theta,
            point_na=point_na,
            ultimate=True,
            ultimate_strain=self.gross_properties.conc_ultimate_strain,
            d_n=ultimate_results.d_n,
        )

        # loop through all concrete geometries and calculate stress
        for geom in concrete_split_geoms:
            analysis_section = AnalysisSection(geometry=geom)

            # calculate stress, force and point of action
            sig, n_conc, d = analysis_section.get_ultimate_stress(
                d_n=ultimate_results.d_n,
                point_na=point_na,
                theta=ultimate_results.theta,
                ultimate_strain=self.gross_properties.conc_ultimate_strain,
                pc_local=pc_local[1],
            )
            conc_sigs.append(sig)
            conc_forces.append((n_conc, d, 0))

            # save analysis section
            analysis_sections.append(analysis_section)

        # loop through all steel geometries and calculate stress
        for steel_geom in self.steel_geometries:
            # get position of bar
            centroid = steel_geom.calculate_centroid()

            # get strain at centroid of steel
            strain = utils.get_ultimate_strain(
                point=(centroid[0], centroid[1]),
                point_na=point_na,
                d_n=ultimate_results.d_n,
                theta=ultimate_results.theta,
                ultimate_strain=self.gross_properties.conc_ultimate_strain,
            )

            # calculate stress, force and point of action
            sig = steel_geom.material.stress_strain_profile.get_stress(strain=strain)
            n_steel = sig * steel_geom.calculate_area()
            _, c_v = principal_coordinate(
                phi=ultimate_results.theta * 180 / np.pi, x=centroid[0], y=centroid[1]
            )
            d = c_v - na_local[1]

            steel_sigs.append(sig)
            steel_strains.append(strain)
            steel_forces.append((n_steel, d, 0))

        return res.StressResult(
            concrete_section=self,
            concrete_analysis_sections=analysis_sections,
            concrete_stresses=conc_sigs,
            concrete_forces=conc_forces,
            steel_geometries=self.steel_geometries,
            steel_stresses=steel_sigs,
            steel_strains=steel_strains,
            steel_forces=steel_forces,
        )

    def get_c_local(
        self,
        theta: float,
    ) -> Tuple[float]:
        r"""Returns the elastic centroid location in local coordinates.

        :param float theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)

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
        r"""Returns the plastic centroid location in local coordinates.

        :param float theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)

        :return: Plastic centroid in local coordinates `(pc_u, pc_v)`
        :rtype: Tuple[float]
        """

        return principal_coordinate(
            phi=theta * 180 / np.pi,
            x=self.gross_properties.axial_pc_x,
            y=self.gross_properties.axial_pc_y,
        )

    def plot_section(
        self,
        title: Optional[str] = "Reinforced Concrete Section",
        background: Optional[bool] = False,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots the reinforced concrete section.

        :param title: Plot title
        :type title: Optional[str]
        :param background: If set to True, uses the plot as a background plot
        :type background: Optional[bool]
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes.Axes`
        """

        with plotting_context(title=title, **kwargs) as (fig, ax):
            # create list of already plotted materials
            plotted_materials = []
            legend_labels = []

            # plot concrete geometries
            for conc_geom in self.concrete_geometries:
                if conc_geom.material not in plotted_materials:
                    patch = mpatches.Patch(
                        color=conc_geom.material.colour, label=conc_geom.material.name
                    )
                    legend_labels.append(patch)
                    plotted_materials.append(conc_geom.material)

                # TODO - when shapely implements polygon plotting, fix this up
                sec = AnalysisSection(geometry=conc_geom)

                if not background:
                    sec.plot_shape(ax=ax)

                # plot the points and facets
                for f in conc_geom.facets:
                    if background:
                        fmt = "k-"
                    else:
                        fmt = "ko-"

                    ax.plot(
                        [conc_geom.points[f[0]][0], conc_geom.points[f[1]][0]],
                        [conc_geom.points[f[0]][1], conc_geom.points[f[1]][1]],
                        fmt,
                        markersize=2,
                        linewidth=1.5,
                    )

            # plot steel geometries
            for steel_geom in self.steel_geometries:
                if steel_geom.material not in plotted_materials:
                    patch = mpatches.Patch(
                        color=steel_geom.material.colour, label=steel_geom.material.name
                    )
                    legend_labels.append(patch)
                    plotted_materials.append(steel_geom.material)

                # plot the points and facets
                coords = list(steel_geom.geom.exterior.coords)
                bar = mpatches.Polygon(
                    xy=coords, closed=False, color=steel_geom.material.colour
                )
                ax.add_patch(bar)

            if not background:
                ax.legend(
                    loc="center left", bbox_to_anchor=(1, 0.5), handles=legend_labels
                )

            ax.set_aspect("equal", anchor="C")

        return ax
