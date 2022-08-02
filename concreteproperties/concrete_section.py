from __future__ import annotations

import warnings
from math import inf, nan, isinf
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import numpy as np
import sectionproperties.pre.geometry as sp_geom
from rich.live import Live
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import concreteproperties.results as res
import concreteproperties.utils as utils
from concreteproperties.analysis_section import AnalysisSection
from concreteproperties.material import Concrete, Steel
from concreteproperties.post import plotting_context
from concreteproperties.pre import CPGeom, CPGeomConcrete

if TYPE_CHECKING:
    import matplotlib


class ConcreteSection:
    """Class for a reinforced concrete section."""

    def __init__(
        self,
        geometry: sp_geom.CompoundGeometry,
    ):
        """Inits the ConcreteSection class.

        :param geometry: *sectionproperties* CompoundGeometry object describing the
            reinforced concrete section
        """

        self.compound_geometry = geometry

        # check overlapping regions
        polygons = [sec_geom.geom for sec_geom in self.compound_geometry.geoms]
        overlapped_regions = sp_geom.check_geometry_overlaps(polygons)
        if overlapped_regions:
            warnings.warn(
                "The provided geometry contains overlapping regions, results may be incorrect."
            )

        # sort into concrete and reinforcement (meshed and lumped) geometries
        self.all_geometries: List[Union[CPGeomConcrete, CPGeom]] = []
        self.meshed_geometries: List[Union[CPGeomConcrete, CPGeom]] = []
        self.concrete_geometries: List[CPGeomConcrete] = []
        self.reinf_geometries_meshed: List[CPGeom] = []
        self.reinf_geometries_lumped: List[CPGeom] = []

        # sort geometry into appropriate list
        for geom in self.compound_geometry.geoms:
            if isinstance(geom.material, Concrete):
                cp_geom = CPGeomConcrete(geom=geom.geom, material=geom.material)
                self.concrete_geometries.append(cp_geom)
                self.meshed_geometries.append(cp_geom)
            else:
                cp_geom = CPGeom(geom=geom.geom, material=geom.material)  # type: ignore

                if cp_geom.material.meshed:
                    self.reinf_geometries_meshed.append(cp_geom)
                    self.meshed_geometries.append(cp_geom)
                elif not cp_geom.material.meshed:
                    self.reinf_geometries_lumped.append(cp_geom)

            self.all_geometries.append(cp_geom)

        # initialise gross properties results class
        self.gross_properties = res.GrossProperties()

        # calculate gross area properties
        self.calculate_gross_area_properties()

    def calculate_gross_area_properties(
        self,
    ):
        """Calculates and stores gross section area properties."""

        # loop through all geometries
        for geom in self.all_geometries:
            # area and centroid of geometry
            area = geom.calculate_area()
            centroid = geom.calculate_centroid()

            self.gross_properties.total_area += area
            self.gross_properties.e_a += area * geom.material.elastic_modulus
            self.gross_properties.mass += area * geom.material.density
            self.gross_properties.e_qx += (
                area * geom.material.elastic_modulus * centroid[1]
            )
            self.gross_properties.e_qy += (
                area * geom.material.elastic_modulus * centroid[0]
            )

        # sum concrete areas
        for conc_geom in self.concrete_geometries:
            self.gross_properties.concrete_area += conc_geom.calculate_area()

        # sum reinforcement meshed areas
        for meshed_geom in self.reinf_geometries_meshed:
            self.gross_properties.reinf_meshed_area += meshed_geom.calculate_area()

        # sum reinforcement lumped areas
        for lumped_geom in self.reinf_geometries_lumped:
            self.gross_properties.reinf_lumped_area += lumped_geom.calculate_area()

        # perimeter
        self.gross_properties.perimeter = self.compound_geometry.calculate_perimeter()

        # centroids
        self.gross_properties.cx = (
            self.gross_properties.e_qy / self.gross_properties.e_a
        )
        self.gross_properties.cy = (
            self.gross_properties.e_qx / self.gross_properties.e_a
        )

        # global second moments of area
        # meshed geometries
        for geom in self.meshed_geometries:
            sec = AnalysisSection(geometry=geom)

            for el in sec.elements:
                el_e_ixx_g, el_e_iyy_g, el_e_ixy_g = el.second_moments_of_area()
                self.gross_properties.e_ixx_g += el_e_ixx_g
                self.gross_properties.e_iyy_g += el_e_iyy_g
                self.gross_properties.e_ixy_g += el_e_ixy_g

        # lumped geometries - treat as lumped circles
        for geom in self.reinf_geometries_lumped:
            # area, diameter and centroid of geometry
            area = geom.calculate_area()
            diam = np.sqrt(4 * area / np.pi)
            centroid = geom.calculate_centroid()

            self.gross_properties.e_ixx_g += geom.material.elastic_modulus * (
                np.pi * pow(diam, 4) / 64 + area * centroid[1] * centroid[1]
            )
            self.gross_properties.e_iyy_g += geom.material.elastic_modulus * (
                np.pi * pow(diam, 4) / 64 + area * centroid[0] * centroid[0]
            )
            self.gross_properties.e_ixy_g += geom.material.elastic_modulus * (
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
            self.gross_properties.phi = np.arctan2(
                self.gross_properties.e_ixx_c - self.gross_properties.e_i11,
                self.gross_properties.e_ixy_c,
            )

        # centroidal section moduli
        x_min, x_max, y_min, y_max = self.compound_geometry.calculate_extents()
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
            geometry=self.compound_geometry,
            cx=self.gross_properties.cx,
            cy=self.gross_properties.cy,
            theta=self.gross_properties.phi,
        )

        # evaluate principal section moduli
        self.gross_properties.e_z11_plus = self.gross_properties.e_i11 / abs(y22_max)
        self.gross_properties.e_z11_minus = self.gross_properties.e_i11 / abs(y22_min)
        self.gross_properties.e_z22_plus = self.gross_properties.e_i22 / abs(x11_max)
        self.gross_properties.e_z22_minus = self.gross_properties.e_i22 / abs(x11_min)

        # store ultimate concrete strain (get smallest from all concrete geometries)
        conc_ult_strain = 0

        for idx, conc_geom in enumerate(self.concrete_geometries):
            ult_strain = (
                conc_geom.material.ultimate_stress_strain_profile.get_ultimate_compressive_strain()
            )
            if idx == 0:
                conc_ult_strain = ult_strain
            else:
                conc_ult_strain = min(conc_ult_strain, ult_strain)

        self.gross_properties.conc_ultimate_strain = conc_ult_strain

    def get_gross_properties(
        self,
    ) -> res.GrossProperties:
        """Returns the gross section properties of the reinforced concrete section.

        :return: Gross concrete properties object
        """

        return self.gross_properties

    def get_transformed_gross_properties(
        self,
        elastic_modulus: float,
    ) -> res.TransformedConcreteProperties:
        """Transforms gross section properties given a reference elastic modulus.

        :param elastic_modulus: Reference elastic modulus

        :return: Transformed concrete properties object
        """

        return res.TransformedConcreteProperties(
            concrete_properties=self.gross_properties, elastic_modulus=elastic_modulus
        )

    def calculate_cracked_properties(
        self,
        theta: float = 0,
    ) -> res.CrackedResults:
        r"""Calculates cracked section properties given a neutral axis angle ``theta``.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)

        :return: Cracked results object
        """

        cracked_results = res.CrackedResults(theta=theta)
        cracked_results.m_cr = self.calculate_cracking_moment(theta=theta)

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(
            points=self.compound_geometry.points, theta=theta
        )
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
                rtol=1e-6,  # type: ignore
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
            # if meshed
            if geom.material.meshed:
                sec = AnalysisSection(geometry=geom)

                for el in sec.elements:
                    el_e_ixx_g, el_e_iyy_g, el_e_ixy_g = el.second_moments_of_area()
                    cracked_results.e_ixx_g_cr += el_e_ixx_g
                    cracked_results.e_iyy_g_cr += el_e_iyy_g
                    cracked_results.e_ixy_g_cr += el_e_ixy_g
            # if lumped
            else:
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
            cracked_results.phi_cr = np.arctan2(
                cracked_results.e_ixx_c_cr - cracked_results.e_i11_cr,
                cracked_results.e_ixy_c_cr,
            )

        return cracked_results

    def calculate_cracking_moment(
        self,
        theta: float,
    ) -> float:
        r"""Calculates the cracking moment given a bending angle ``theta``.

        :param theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)

        :return: Cracking moment
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
        m_c = 0
        for idx, conc_geom in enumerate(self.concrete_geometries):
            # get distance from centroid to extreme tensile fibre
            d = utils.calculate_max_bending_depth(
                points=conc_geom.points,
                c_local_v=utils.global_to_local(
                    theta=theta, x=self.gross_properties.cx, y=self.gross_properties.cy
                )[1],
                theta=theta,
            )

            # if no part of the section is in tension, go to next geometry
            if d == 0:
                continue

            # cracking moment for this geometry
            f_t = conc_geom.material.flexural_tensile_strength
            m_c_geom = (f_t / conc_geom.material.elastic_modulus) * (e_iuu / d)

            # if we are the first geometry, initialise cracking moment
            if idx == 0:
                m_c = m_c_geom
            # otherwise take smaller cracking moment
            else:
                m_c = min(m_c, m_c_geom)

        return m_c

    def cracked_neutral_axis_convergence(
        self,
        d_nc: float,
        cracked_results: res.CrackedResults,
    ) -> float:
        """Given a trial cracked neutral axis depth ``d_nc``, determines the difference
        between the first moments of area above and below the trial axis.

        :param d_nc: Trial cracked neutral axis
        :param cracked_results: Cracked results object

        :return: Cracked neutral axis convergence
        """

        # calculate extreme fibre in global coordinates
        extreme_fibre, d_t = utils.calculate_extreme_fibre(
            points=self.compound_geometry.points, theta=cracked_results.theta
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
        na_local = utils.global_to_local(
            theta=cracked_results.theta, x=point_na[0], y=point_na[1]
        )

        # split concrete geometries above and below d_nc, discard below
        cracked_geoms: List[Union[CPGeomConcrete, CPGeom]] = []

        for conc_geom in self.concrete_geometries:
            top_geoms, _ = conc_geom.split_section(
                point=point_na,
                theta=cracked_results.theta,
            )

            # save compression geometries
            cracked_geoms.extend(top_geoms)

        # add reinforcement geometries to list
        cracked_geoms.extend(self.reinf_geometries_meshed)
        cracked_geoms.extend(self.reinf_geometries_lumped)

        # determine moment of area equilibrium about neutral axis
        e_qu = 0  # initialise first moment of area

        for geom in cracked_geoms:
            ea = geom.calculate_area() * geom.material.elastic_modulus
            centroid = geom.calculate_centroid()

            # convert centroid to local coordinates
            _, c_v = utils.global_to_local(
                theta=cracked_results.theta, x=centroid[0], y=centroid[1]
            )

            # calculate first moment of area
            e_qu += ea * (c_v - na_local[1])

        cracked_results.cracked_geometries = cracked_geoms

        return e_qu

    def moment_curvature_analysis(
        self,
        theta: float = 0,
        kappa_inc: float = 1e-7,
        delta_m_min: float = 0.15,
        delta_m_max: float = 0.3,
    ) -> res.MomentCurvatureResults:
        r"""Performs a moment curvature analysis given a bending angle ``theta``.

        Analysis continues until a material reaches its ultimate strain.

        :param: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :param kappa_inc: Initial curvature increment
        :param delta_m_min: Relative change in moment at which to double step
        :param delta_m_max: Relative change in moment at which to halve step

        :return: Moment curvature results object
        """

        # initialise variables
        moment_curvature = res.MomentCurvatureResults(theta=theta)
        iter = 0

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(
            points=self.compound_geometry.points, theta=theta
        )
        a = 1e-6 * d_t  # sufficiently small depth of compressive zone
        b = d_t  # neutral axis at extreme tensile fibre

        # create progress bar
        progress = utils.create_unknown_progress()

        with Live(progress, refresh_per_second=10) as live:
            task = progress.add_task(
                description="[red]Generating M-K diagram",
                total=None,
            )

            # while there hasn't been a failure
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

                kappa = 0 if iter == 0 else moment_curvature.kappa[-1] + kappa_inc

                # find neutral axis that gives convergence of the axial force
                try:
                    (d_n, r) = brentq(
                        f=self.service_normal_force_convergence,
                        a=a,
                        b=b,
                        args=(kappa, moment_curvature),
                        xtol=1e-3,
                        rtol=1e-6,  # type: ignore
                        full_output=True,
                        disp=False,
                    )
                except ValueError:
                    warnings.warn("brentq algorithm failed.")

                m_xy = np.sqrt(
                    moment_curvature._m_x_i**2 + moment_curvature._m_y_i**2
                )

                text_update = "[red]Generating M-K diagram: "
                text_update += f"M={m_xy:.3e}"

                progress.update(task, description=text_update)

                # save results
                if not moment_curvature._failure:
                    moment_curvature.kappa.append(kappa)
                    moment_curvature.n.append(moment_curvature._n_i)
                    moment_curvature.m_x.append(moment_curvature._m_x_i)
                    moment_curvature.m_y.append(moment_curvature._m_y_i)
                    moment_curvature.m_xy.append(m_xy)
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
        """Given a neutral axis depth ``d_n`` and curvature ``kappa``, returns the the
        net axial force.

        :param d_nc: Trial cracked neutral axis
        :param kappa: Curvature
        :param moment_curvature: Moment curvature results object

        :return: Net axial force
        """

        # reset failure
        moment_curvature._failure = False

        # calculate extreme fibre in global coordinates
        extreme_fibre, d_t = utils.calculate_extreme_fibre(
            points=self.compound_geometry.points, theta=moment_curvature.theta
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

        # create splits in meshed geometries at points in stress-strain profiles
        meshed_split_geoms: List[Union[CPGeom, CPGeomConcrete]] = []

        for meshed_geom in self.meshed_geometries:
            split_geoms = utils.split_geom_at_strains(
                geom=meshed_geom,
                theta=moment_curvature.theta,
                point_na=point_na,
                ultimate=False,
                kappa=kappa,
            )

            meshed_split_geoms.extend(split_geoms)

        # initialise results
        n = 0
        m_x = 0
        m_y = 0

        # calculate meshed geometry actions
        for meshed_geom in meshed_split_geoms:
            sec = AnalysisSection(geometry=meshed_geom)

            n_sec, m_x_sec, m_y_sec, min_strain, max_strain = sec.service_analysis(
                point_na=point_na,
                theta=moment_curvature.theta,
                kappa=kappa,
                centroid=(self.gross_properties.cx, self.gross_properties.cy),
            )

            n += n_sec
            m_x += m_x_sec
            m_y += m_y_sec

            # check for failure
            ult_comp_strain = (
                meshed_geom.material.stress_strain_profile.get_ultimate_compressive_strain()
            )
            ult_tens_strain = (
                meshed_geom.material.stress_strain_profile.get_ultimate_tensile_strain()
            )

            # don't worry about tension failure in concrete
            if max_strain > ult_comp_strain or (
                min_strain < ult_tens_strain
                and not isinstance(meshed_geom, CPGeomConcrete)
            ):
                moment_curvature._failure = True
                moment_curvature.failure_geometry = meshed_geom

        # calculate lumped geometry actions
        for lumped_geom in self.reinf_geometries_lumped:
            # calculate area and centroid
            area = lumped_geom.calculate_area()
            centroid = lumped_geom.calculate_centroid()

            # get strain at centroid of lump
            strain = utils.get_service_strain(
                point=(centroid[0], centroid[1]),
                point_na=point_na,
                theta=moment_curvature.theta,
                kappa=kappa,
            )

            # check for failure
            ult_comp_strain = (
                lumped_geom.material.stress_strain_profile.get_ultimate_compressive_strain()
            )
            ult_tens_strain = (
                lumped_geom.material.stress_strain_profile.get_ultimate_tensile_strain()
            )

            if strain > ult_comp_strain or strain < ult_tens_strain:
                moment_curvature._failure = True
                moment_curvature.failure_geometry = lumped_geom

            # calculate stress and force
            stress = lumped_geom.material.stress_strain_profile.get_stress(
                strain=strain
            )
            force = stress * area
            n += force

            # calculate moment
            m_x += force * (centroid[1] - self.gross_properties.cy)
            m_y += force * (centroid[0] - self.gross_properties.cx)

        moment_curvature._n_i = n
        moment_curvature._m_x_i = m_x
        moment_curvature._m_y_i = m_y

        # calculate convergence
        return n

    def ultimate_bending_capacity(
        self,
        theta: float = 0,
        n: float = 0,
    ) -> res.UltimateBendingResults:
        r"""Given a neutral axis angle ``theta`` and an axial force ``n``, calculates
        the ultimate bending capacity.

        Note that ``k_u`` is calculated only for lumped (non-meshed) geometries.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :param n: Net axial force

        :return: Ultimate bending results object
        """

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(
            points=self.compound_geometry.points, theta=theta
        )
        a = 1e-6 * d_t  # sufficiently small depth of compressive zone
        b = 6 * d_t  # neutral axis at sufficiently large tensile fibre

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
                rtol=1e-6,  # type: ignore
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
        """Given a neutral axis depth ``d_n`` and neutral axis angle ``theta``,
        calculates the difference between the target net axial force ``n`` and the
        calculated axial force.

        :param d_n: Depth of the neutral axis from the extreme compression fibre
        :param n: Net axial force
        :param ultimate_results: Ultimate bending results object

        :return: Axial force convergence
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
        """Given a neutral axis depth ``d_n`` and neutral axis angle ``theta``,
        calculates the resultant bending moments ``m_x``, ``m_y``, ``m_xy`` and the net
        axial force ``n``.

        :param d_n: Depth of the neutral axis from the extreme compression fibre
        :param ultimate_results: Ultimate bending results object

        :return: Ultimate bending results object
        """

        if ultimate_results is None:
            ultimate_results = res.UltimateBendingResults(theta=0)

        # calculate extreme fibre in global coordinates
        extreme_fibre, _ = utils.calculate_extreme_fibre(
            points=self.compound_geometry.points, theta=ultimate_results.theta
        )

        # extreme fibre in local coordinates
        _, ef_v = utils.global_to_local(
            theta=ultimate_results.theta,
            x=extreme_fibre[0],
            y=extreme_fibre[1],
        )

        # validate d_n input
        if d_n <= 0:
            raise ValueError("d_n must be positive.")

        # find point on neutral axis by shifting by d_n
        if isinf(d_n):
            point_na = (0, 0)
        else:
            point_na = utils.point_on_neutral_axis(
                extreme_fibre=extreme_fibre, d_n=d_n, theta=ultimate_results.theta
            )

        # create splits in meshed geometries at points in stress-strain profiles
        meshed_split_geoms: List[Union[CPGeom, CPGeomConcrete]] = []

        if isinf(d_n):
            meshed_split_geoms = self.meshed_geometries
        else:
            for meshed_geom in self.meshed_geometries:
                split_geoms = utils.split_geom_at_strains(
                    geom=meshed_geom,
                    theta=ultimate_results.theta,
                    point_na=point_na,
                    ultimate=True,
                    ultimate_strain=self.gross_properties.conc_ultimate_strain,
                    d_n=d_n,
                )

                meshed_split_geoms.extend(split_geoms)

        # initialise results
        n = 0
        m_x = 0
        m_y = 0
        k_u = []

        # calculate meshed geometry actions
        for meshed_geom in meshed_split_geoms:
            sec = AnalysisSection(geometry=meshed_geom)
            n_sec, m_x_sec, m_y_sec = sec.ultimate_analysis(
                point_na=point_na,
                d_n=d_n,
                theta=ultimate_results.theta,
                ultimate_strain=self.gross_properties.conc_ultimate_strain,
                centroid=(self.gross_properties.cx, self.gross_properties.cy),
            )

            n += n_sec
            m_x += m_x_sec
            m_y += m_y_sec

        # calculate lumped actions
        for lumped_geom in self.reinf_geometries_lumped:
            # calculate area and centroid
            area = lumped_geom.calculate_area()
            centroid = lumped_geom.calculate_centroid()

            # get strain at centroid of lump
            if isinf(d_n):
                strain = self.gross_properties.conc_ultimate_strain
            else:
                strain = utils.get_ultimate_strain(
                    point=(centroid[0], centroid[1]),
                    point_na=point_na,
                    d_n=d_n,
                    theta=ultimate_results.theta,
                    ultimate_strain=self.gross_properties.conc_ultimate_strain,
                )

            # calculate stress and force
            stress = lumped_geom.material.stress_strain_profile.get_stress(
                strain=strain
            )
            force = stress * area
            n += force

            # convert centroid to local coordinates
            _, c_v = utils.global_to_local(
                theta=ultimate_results.theta, x=centroid[0], y=centroid[1]
            )

            # calculate moment
            m_x += force * (centroid[1] - self.gross_properties.cy)
            m_y += force * (centroid[0] - self.gross_properties.cx)

            # calculate k_u
            d = ef_v - c_v
            k_u.append(d_n / d)

        # calculate resultant moment
        m_xy = np.sqrt(m_x * m_x + m_y * m_y)

        # save results
        ultimate_results.d_n = d_n
        ultimate_results.k_u = min(k_u)
        ultimate_results.n = n
        ultimate_results.m_x = m_x
        ultimate_results.m_y = m_y
        ultimate_results.m_xy = m_xy

        return ultimate_results

    def moment_interaction_diagram(
        self,
        theta: float = 0,
        control_points: List[Tuple[str, float]] = [
            ("kappa0", 0.0),
            ("D", 1.0),
            ("fy", 1.0),
            ("N", 0.0),
            ("d_n", 1e-6),
        ],
        labels: List[Union[str, None]] = [None],
        n_points: Union[int, List[int]] = [4, 12, 12, 4],
        max_comp: Optional[float] = None,
        max_comp_labels: List[Union[str, None]] = [None, None],
    ) -> res.MomentInteractionResults:
        r"""Generates a moment interaction diagram given a neutral axis angle `theta`
        and `n_points` calculation points between the decompression case and the pure
        bending case.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :param control_points: List of control points over which to generate the
            interaction diagram. Each entry in ``control_points`` is a ``Tuple`` with
            the first item the type of control point and the second item defining the
            location of the control point. Acceptable types of control points are
            ``"D"`` (ratio of neutral axis depth to section depth), ``"d_n"`` (neutral
            axis depth), ``"fy"`` (yield ratio of the most extreme tensile bar), ``"N"``
            (axial force) and ``"kappa"`` (zero curvature compression - must be at start
            of list, second value in tuple is not used). Control points must be defined
            in an order which results in a decreasing neutral axis depth (decreasing
            axial force). The default control points define an interaction diagram from
            the decompression point to the pure bending point.
        :param labels: List of labels to apply to the ``control_points`` for plotting
            purposes, length must be the same as the length of ``control_points``. If a
            single value is provided, will apply this label to all control points.
        :param n_points: Number of neutral axis depths to compute between each control
            point. Length must be one less than the length of ``control_points``. If an
            integer is provided this will be used between all control points.
        :param max_comp: If provided, limits the maximum compressive force in the moment
            interaction diagram to ``max_comp``
        :param max_comp_labels: Labels to apply to the ``max_comp`` intersection points,
            first value is at zero moment, second value is at the intersection with the
            interaction diagram

        :raises ValueError: If ``control_points``, ``labels`` or ``n_points`` is invalid

        :return: Moment interaction results object
        """

        # if an integer is provided for n_points, generate a list
        if isinstance(n_points, int):
            n_points = [n_points] * (len(control_points) - 1)

        # if there are no labels provided, generate a list
        if len(labels) == 1:
            labels = labels * len(control_points)

        # validate n_points length
        if len(n_points) != len(control_points) - 1:
            raise ValueError(
                "Length of n_points must be one less than the length of control_points."
            )

        # validate n_points entries are all longer than 2
        for n_pt in n_points:
            if n_pt < 3:
                raise ValueError("n_points entries must be greater than 2.")

        # validate labels length
        if len(labels) != len(control_points):
            raise ValueError("Length of labels must equal length of control_points.")

        # initialise results
        mi_results = res.MomentInteractionResults()

        # compute extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(
            points=self.compound_geometry.points, theta=theta
        )

        # function to decode d_n from control_point
        def decode_d_n(cp):
            # multiple of section depth
            if cp[0] == "D":
                # check D
                if cp[1] <= 0:
                    raise ValueError(
                        f"Provided section depth (D) {cp[1]:.3f} must be greater than 0."
                    )
                return cp[1] * d_t
            # neutral axis depth
            elif cp[0] == "d_n":
                # check d_n
                if cp[1] <= 0:
                    raise ValueError(
                        f"Provided d_n {cp[1]:.3f} must be greater than zero."
                    )
                return cp[1]
            # extreme tensile steel yield ratio
            elif cp[0] == "fy":
                # get extreme tensile bar
                d_ext, eps_sy = self.extreme_bar(theta=theta)
                # get compressive strain at extreme fibre
                eps_cu = self.gross_properties.conc_ultimate_strain
                return d_ext * (eps_cu) / (cp[1] * eps_sy + eps_cu)
            # provided axial force
            elif cp[0] == "N":
                ult_res = self.ultimate_bending_capacity(theta=theta, n=cp[1])
                return ult_res.d_n
            # zero curvature
            elif cp[0] == "kappa0":
                return 2 * d_t  # sufficient depth to capture rectangular block
            # control point type not valid
            else:
                raise ValueError(
                    "First value of control_point tuple must be D, d_n, fy, N or kappa0."
                )

        # see if a kappa0 was used
        has_kappa0 = False

        for cp in control_points:
            if cp[0] == "kappa0":
                has_kappa0 = True
                break

        # generate list of neutral axis depths to analyse and list of labels to save
        d_n_list = []
        label_list = []
        start_d_n = 0
        end_d_n = 0

        for idx, n_pt in enumerate(n_points):
            # get netural axis depths from control_points
            start_d_n = decode_d_n(control_points[idx])
            end_d_n = decode_d_n(control_points[idx + 1])

            # generate list of neutral axis depths for this interval
            d_n_list.extend(
                np.linspace(
                    start=start_d_n, stop=end_d_n, num=n_pt - 1, endpoint=False
                ).tolist()
            )

            # add labels
            label_list.append(labels[idx])
            label_list.extend([None] * (n_pt - 2))

        # add final d_n and label
        d_n_list.append(end_d_n)
        label_list.append(labels[-1])

        # check d_n_list is ordered
        if not all(d_n_list[i] >= d_n_list[i + 1] for i in range(len(d_n_list) - 1)):
            msg = "control_points must create an ordered list of neutral axes from "
            msg += "tensile fibre to compressive fibre."
            raise ValueError(msg)

        # create progress bar
        progress = utils.create_known_progress()

        with Live(progress, refresh_per_second=10) as live:
            # add progress bar task
            task = progress.add_task(
                description="[red]Generating M-N diagram",
                total=sum(n_points) - len(n_points) + 1,
            )

            # loop through all neutral axes
            for idx, d_n in enumerate(d_n_list):
                # calculate ultimate results
                if idx == 0 and has_kappa0:
                    ult_res = self.calculate_ultimate_section_actions(
                        d_n=inf,
                        ultimate_results=res.UltimateBendingResults(theta=theta),
                    )
                else:
                    ult_res = self.calculate_ultimate_section_actions(
                        d_n=d_n,
                        ultimate_results=res.UltimateBendingResults(theta=theta),
                    )
                # add label
                ult_res.label = label_list[idx]
                # add ultimate result to moment interactions results and update progress
                mi_results.results.append(ult_res)
                progress.update(task, advance=1)

            # display finished progress bar
            progress.update(
                task,
                description="[bold green]:white_check_mark: M-N diagram generated",
            )
            live.refresh()

        # cut diagram at max_comp
        if max_comp:
            # find intersection of max comp with interaction diagram
            # and determine which points need to be removed from diagram
            x = []
            y_mx = []
            y_my = []
            y_mxy = []
            idx_to_keep = 0

            for idx, mi_res in enumerate(mi_results.results):
                # create coordinates for interpolation
                x.append(mi_res.n)
                y_mx.append(mi_res.m_x)
                y_my.append(mi_res.m_y)
                y_mxy.append(mi_res.m_xy)

                # determine which index is the first to keep
                if idx_to_keep == 0 and mi_res.n < max_comp:
                    idx_to_keep = idx

            # create interpolation function and determine moment which corresponds to
            # an axial force of max_comp
            f_mx = interp1d(x=x, y=y_mx)
            f_my = interp1d(x=x, y=y_my)
            f_mxy = interp1d(x=x, y=y_mxy)
            m_max_comp_mx = f_mx(max_comp)
            m_max_comp_my = f_my(max_comp)
            m_max_comp_mxy = f_mxy(max_comp)

            # remove points in diagram
            del mi_results.results[:idx_to_keep]

            # add first two points to diagram
            # (m_max_comp, max_comp)
            mi_results.results.insert(
                0,
                res.UltimateBendingResults(
                    theta=theta,
                    d_n=nan,
                    k_u=nan,
                    n=max_comp,
                    m_x=m_max_comp_mx,
                    m_y=m_max_comp_my,
                    m_xy=m_max_comp_mxy,
                    label=max_comp_labels[1],
                ),
            )
            # (0, max_comp)
            mi_results.results.insert(
                0,
                res.UltimateBendingResults(
                    theta=theta,
                    d_n=inf,
                    k_u=0,
                    n=max_comp,
                    m_x=0,
                    m_y=0,
                    m_xy=0,
                    label=max_comp_labels[0],
                ),
            )

        return mi_results

    def biaxial_bending_diagram(
        self,
        n: float = 0,
        n_points: int = 48,
    ) -> res.BiaxialBendingResults:
        """Generates a biaxial bending diagram given a net axial force ``n`` and
        ``n_points`` calculation points.

        :param n: Net axial force
        :param n_points: Number of calculation points between the decompression

        :return: Biaxial bending results
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
        n: float = 0,
        m_x: float = 0,
        m_y: float = 0,
    ) -> res.StressResult:
        """Calculates stresses within the reinforced concrete section assuming an
        uncracked section.

        Uses gross area section properties to determine concrete and steel stresses
        given an axial force `n`, and bending moments `m_x` and `m_y`.

        :param n: Axial force
        :param m_x: Bending moment about the x-axis
        :param m_y: Bending moment about the y-axis

        :return: Stress results object
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
        n: float = 0,
        m: float = 0,
    ) -> res.StressResult:
        """Calculates stresses within the reinforced concrete section assuming a cracked
        section.

        Uses cracked area section properties to determine concrete and steel stresses
        given an axial force `n` and bending moment `m` about the bending axis stored
        in `cracked_results`.

        :param cracked_results: Cracked results objects
        :param n: Axial force
        :param m: Bending moment

        :return: Stress results object
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
        sign = 0
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
        na_local = utils.global_to_local(theta=theta, x=point_na[0], y=point_na[1])

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
        :param m: Bending moment
        :param kappa: Curvature, if provided overrides the supplied bending moment and
            plots the stress at the given curvature

        :return: Stress results object
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
                rtol=1e-6,  # type: ignore
                full_output=True,
                disp=False,
            )
        except ValueError:
            warnings.warn("brentq algorithm failed.")
            d_n = 0

        # initialise stress results
        analysis_sections = []
        conc_sigs = []
        conc_forces = []
        steel_sigs = []
        steel_strains = []
        steel_forces = []

        # find point on neutral axis by shifting by d_n
        point_na = utils.point_on_neutral_axis(
            extreme_fibre=extreme_fibre, d_n=d_n, theta=theta
        )

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
            sig, n_conc, d_x, d_y = analysis_section.get_service_stress(
                d_n=d_n,
                kappa=kappa,
                point_na=point_na,
                theta=theta,
                centroid=(self.gross_properties.cx, self.gross_properties.cy),
            )
            conc_sigs.append(sig)
            conc_forces.append((n_conc, d_x, d_y))

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

            steel_sigs.append(sig)
            steel_strains.append(strain)
            steel_forces.append(
                (
                    n_steel,
                    centroid[0] - -self.gross_properties.cx,
                    centroid[1] - -self.gross_properties.cy,
                )
            )

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

        :return: Stress results object
        """

        # depth of neutral axis at extreme tensile fibre
        extreme_fibre, _ = utils.calculate_extreme_fibre(
            points=self.geometry.points, theta=ultimate_results.theta
        )

        # find point on neutral axis by shifting by d_n
        if isinf(ultimate_results.d_n):
            point_na = (0, 0)
        else:
            point_na = utils.point_on_neutral_axis(
                extreme_fibre=extreme_fibre,
                d_n=ultimate_results.d_n,
                theta=ultimate_results.theta,
            )

        # initialise stress results for each concrete geometry
        analysis_sections = []
        conc_sigs = []
        conc_forces = []
        steel_sigs = []
        steel_strains = []
        steel_forces = []

        # create splits in concrete geometries at points in stress-strain profiles
        if isinf(ultimate_results.d_n):
            concrete_split_geoms = self.concrete_geometries
        else:
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
            sig, n_conc, d_x, d_y = analysis_section.get_ultimate_stress(
                d_n=ultimate_results.d_n,
                point_na=point_na,
                theta=ultimate_results.theta,
                ultimate_strain=self.gross_properties.conc_ultimate_strain,
                centroid=(self.gross_properties.cx, self.gross_properties.cy),
            )
            conc_sigs.append(sig)
            conc_forces.append((n_conc, d_x, d_y))

            # save analysis section
            analysis_sections.append(analysis_section)

        # loop through all steel geometries and calculate stress
        for steel_geom in self.steel_geometries:
            # get position of bar
            centroid = steel_geom.calculate_centroid()

            # get strain at centroid of steel
            if isinf(ultimate_results.d_n):
                strain = self.gross_properties.conc_ultimate_strain
            else:
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

            steel_sigs.append(sig)
            steel_strains.append(strain)
            steel_forces.append(
                (
                    n_steel,
                    centroid[0] - self.gross_properties.cx,
                    centroid[1] - self.gross_properties.cy,
                )
            )

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

    def extreme_bar(
        self,
        theta: float,
    ) -> Tuple[float, float]:
        r"""Given neutral axis angle ``theta``, determines the depth of the furthest
        lumped reinforcement from the extreme compressive fibre and also returns its
        yield strain.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal
            axis (:math:`-\pi \leq \theta \leq \pi`)

        :return: Depth of furthest bar and its yield strain
        """

        # initialise variables
        d_ext = 0

        # calculate extreme fibre in local coordinates
        extreme_fibre, _ = utils.calculate_extreme_fibre(
            points=self.compound_geometry.points, theta=theta
        )
        _, ef_v = utils.global_to_local(
            theta=theta, x=extreme_fibre[0], y=extreme_fibre[1]
        )

        # get depth to extreme lumped reinforcement
        extreme_geom = self.reinf_geometries_lumped[0]

        for lumped_geom in self.reinf_geometries_lumped:
            centroid = lumped_geom.calculate_centroid()

            # convert centroid to local coordinates
            _, c_v = utils.global_to_local(theta=theta, x=centroid[0], y=centroid[1])

            # calculate d
            d = ef_v - c_v

            if d > d_ext:
                d_ext = d
                extreme_geom = lumped_geom

        # calculate yield strain
        yield_strain = (
            extreme_geom.material.stress_strain_profile.get_yield_strength()
            / extreme_geom.material.stress_strain_profile.get_elastic_modulus()
        )

        return d_ext, yield_strain

    def plot_section(
        self,
        title: str = "Reinforced Concrete Section",
        background: bool = False,
        **kwargs,
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots the reinforced concrete section.

        :param title: Plot title
        :param background: If set to True, uses the plot as a background plot
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        """

        with plotting_context(title=title, **kwargs) as (fig, ax):
            # create list of already plotted materials
            plotted_materials = []
            legend_labels = []

            # plot meshed geometries
            for meshed_geom in self.meshed_geometries:
                if meshed_geom.material not in plotted_materials:
                    patch = mpatches.Patch(
                        color=meshed_geom.material.colour,
                        label=meshed_geom.material.name,
                    )
                    legend_labels.append(patch)
                    plotted_materials.append(meshed_geom.material)

                # TODO - when shapely implements polygon plotting, fix this up
                sec = AnalysisSection(geometry=meshed_geom)

                if not background:
                    sec.plot_shape(ax=ax)

                # plot the points and facets
                for f in meshed_geom.facets:
                    if background:
                        fmt = "k-"
                    else:
                        fmt = "ko-"

                    ax.plot(  # type: ignore
                        [meshed_geom.points[f[0]][0], meshed_geom.points[f[1]][0]],
                        [meshed_geom.points[f[0]][1], meshed_geom.points[f[1]][1]],
                        fmt,
                        markersize=2,
                        linewidth=1.5,
                    )

            # plot lumped geometries
            for lumped_geom in self.reinf_geometries_lumped:
                if lumped_geom.material not in plotted_materials:
                    patch = mpatches.Patch(
                        color=lumped_geom.material.colour,
                        label=lumped_geom.material.name,
                    )
                    legend_labels.append(patch)
                    plotted_materials.append(lumped_geom.material)

                # plot the points and facets
                coords = list(lumped_geom.geom.exterior.coords)
                bar = mpatches.Polygon(
                    xy=coords, closed=False, color=lumped_geom.material.colour
                )
                ax.add_patch(bar)  # type: ignore

            if not background:
                ax.legend(  # type: ignore
                    loc="center left", bbox_to_anchor=(1, 0.5), handles=legend_labels
                )

            ax.set_aspect("equal", anchor="C")  # type: ignore

        return ax
