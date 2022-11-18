from __future__ import annotations

import warnings
from math import inf, isinf, nan
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
from concreteproperties.material import Concrete
from concreteproperties.post import plotting_context
from concreteproperties.pre import CPGeom, CPGeomConcrete

if TYPE_CHECKING:
    import matplotlib


class ConcreteSection:
    """Class for a reinforced concrete section."""

    def __init__(
        self,
        geometry: sp_geom.CompoundGeometry,
        moment_centroid: Optional[Tuple[float, float]] = None,
        geometric_centroid_override: bool = False,
    ):
        """Inits the ConcreteSection class.

        :param geometry: *sectionproperties* CompoundGeometry object describing the
            reinforced concrete section
        :param moment_centroid: If specified, all moments for service and ultimate
            analyses are calculated about this point. If not specified, all moments are
            calculated about the gross cross-section centroid, i.e. no material
            properties applied.
        :param geometric_centroid_override: If set to True, sets ``moment_centroid`` to
            the geometric centroid i.e. material properties applied (useful for
            composite section analysis)
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

        # set moment centroid
        if moment_centroid:
            self.moment_centroid = moment_centroid
        else:
            self.moment_centroid = (
                self.gross_properties.cx_gross,
                self.gross_properties.cy_gross,
            )

        # if moment centroid overriden
        if geometric_centroid_override:
            self.moment_centroid = self.gross_properties.cx, self.gross_properties.cy

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
            self.gross_properties.qx_gross += area * centroid[1]
            self.gross_properties.qy_gross += area * centroid[0]

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
        self.gross_properties.cx_gross = (
            self.gross_properties.qy_gross / self.gross_properties.total_area
        )
        self.gross_properties.cy_gross = (
            self.gross_properties.qx_gross / self.gross_properties.total_area
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
    ) -> res.TransformedGrossProperties:
        """Transforms gross section properties given a reference elastic modulus.

        :param elastic_modulus: Reference elastic modulus

        :return: Transformed concrete properties object
        """

        return res.TransformedGrossProperties(
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
            msg = "Analysis failed. Please raise an issue at "
            msg += "https://github.com/robbievanleeuwen/concrete-properties/issues"
            raise utils.AnalysisError(msg)

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
        kappa_mult: float = 2,
        kappa_inc_max: float = 5e-6,
        delta_m_min: float = 0.15,
        delta_m_max: float = 0.3,
        progress_bar: bool = True,
    ) -> res.MomentCurvatureResults:
        r"""Performs a moment curvature analysis given a bending angle ``theta``.

        Analysis continues until a material reaches its ultimate strain.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :param kappa_inc: Initial curvature increment
        :param kappa_mult: Multiplier to apply to the curvature increment ``kappa_inc``
            when ``delta_m_max`` is satisfied. When ``delta_m_min`` is satisfied, the
            inverse of this multipler is applied to ``kappa_inc``.
        :param kappa_inc_max: Maximum curvature increment
        :param delta_m_min: Relative change in moment at which to reduce the curvature
            increment
        :param delta_m_max: Relative change in moment at which to increase the curvature
            increment
        :param progress_bar: If set to True, displays the progress bar

        :return: Moment curvature results object
        """

        # initialise variables
        moment_curvature = res.MomentCurvatureResults(theta=theta)

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(
            points=self.compound_geometry.points, theta=theta
        )
        a = 1e-6 * d_t  # sufficiently small depth of compressive zone
        b = d_t  # neutral axis at extreme tensile fibre

        # function that performs moment curvature analysis
        def mcurve(kappa_inc=kappa_inc, progress=None):
            iter = 0
            kappa = 0

            while not moment_curvature._failure:
                # calculate adaptive step size for curvature
                if iter > 2:
                    moment_diff = (
                        abs(moment_curvature.kappa[-1] - moment_curvature.kappa[-2])
                        / moment_curvature.kappa[-1]
                    )
                    if moment_diff <= delta_m_min:
                        kappa_inc *= kappa_mult
                    elif moment_diff >= delta_m_max:
                        kappa_inc *= 1 / kappa_mult

                    # enforce maximum curvature increment
                    if kappa_inc > kappa_inc_max:
                        kappa_inc = kappa_inc_max

                # update curvature
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
                    if not moment_curvature._failure:
                        msg = "Analysis failed. Please raise an issue at "
                        msg += "https://github.com/robbievanleeuwen/concrete-properties/issues"
                        raise utils.AnalysisError(msg)

                m_xy = np.sqrt(
                    moment_curvature._m_x_i**2 + moment_curvature._m_y_i**2
                )

                if progress:
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
                    moment_curvature.convergence.append(
                        moment_curvature._failure_convergence
                    )
                    iter += 1

            # find kappa corresponding to failure strain:
            # curvature before and after failure
            kappa_a = moment_curvature.kappa[-1]
            kappa_b = kappa

            # this method (given a kappa) outputs the failure convergence
            # (normalised to zero)
            def failure_kappa(kappa_fail):
                # given kappa find equilibrium
                brentq(
                    f=self.service_normal_force_convergence,
                    a=a,
                    b=b,
                    args=(kappa_fail, moment_curvature),
                    xtol=1e-3,
                    rtol=1e-6,  # type: ignore
                    full_output=True,
                    disp=False,
                )

                return moment_curvature._failure_convergence - 1

            if progress:
                progress.update(task, description="[red]Finding failure curvature...")

            # find curvature corresponding to failure
            brentq(
                f=failure_kappa,
                a=kappa_a,
                b=kappa_b,
                full_output=True,
                disp=False,
            )

            # save final results
            m_xy = np.sqrt(moment_curvature._m_x_i**2 + moment_curvature._m_y_i**2)
            moment_curvature.kappa.append(moment_curvature._kappa)
            moment_curvature.n.append(moment_curvature._n_i)
            moment_curvature.m_x.append(moment_curvature._m_x_i)
            moment_curvature.m_y.append(moment_curvature._m_y_i)
            moment_curvature.m_xy.append(m_xy)
            moment_curvature.convergence.append(moment_curvature._failure_convergence)

        # create progress bar
        if progress_bar:
            # create progress bar
            progress = utils.create_unknown_progress()

            with Live(progress, refresh_per_second=10) as live:
                task = progress.add_task(
                    description="[red]Generating M-K diagram",
                    total=None,
                )

                mcurve(progress=progress)

                progress.update(
                    task,
                    description="[bold green]:white_check_mark: M-K diagram generated",
                )
                live.refresh()
        else:
            mcurve()

        return moment_curvature

    def service_normal_force_convergence(
        self,
        d_n: float,
        kappa: float,
        moment_curvature: res.MomentCurvatureResults,
    ) -> float:
        """Given a neutral axis depth ``d_n`` and curvature ``kappa``, returns the the
        net axial force.

        :param d_n: Trial neutral axis
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
        failure_convergence = 0

        # calculate meshed geometry actions
        for meshed_geom in meshed_split_geoms:
            sec = AnalysisSection(geometry=meshed_geom)

            n_sec, m_x_sec, m_y_sec, min_strain, max_strain = sec.service_analysis(
                point_na=point_na,
                theta=moment_curvature.theta,
                kappa=kappa,
                centroid=self.moment_centroid,
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

            # update failure convergence
            # compression failure
            failure_convergence = max(max_strain / ult_comp_strain, failure_convergence)
            # tensile failure (ignore concrete)
            if not isinstance(meshed_geom, CPGeomConcrete):
                failure_convergence = max(
                    min_strain / ult_tens_strain, failure_convergence
                )

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

            # update failure convergence
            # compression failure
            failure_convergence = max(strain / ult_comp_strain, failure_convergence)
            # tensile failure
            failure_convergence = max(strain / ult_tens_strain, failure_convergence)

            # calculate stress and force
            stress = lumped_geom.material.stress_strain_profile.get_stress(
                strain=strain
            )
            force = stress * area
            n += force

            # calculate moment
            m_x += force * (centroid[1] - self.moment_centroid[1])
            m_y += force * (centroid[0] - self.moment_centroid[0])

        moment_curvature._kappa = kappa
        moment_curvature._n_i = n
        moment_curvature._m_x_i = m_x
        moment_curvature._m_y_i = m_y
        moment_curvature._failure_convergence = failure_convergence

        # return normal force convergence
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
            msg = "Analysis failed. The solver could not find a neutral axis that "
            msg += "satisfies equilibrium. This may be due to an axial force that "
            msg += "exceeds the tensile or compressive capacity of the cross-section."
            raise utils.AnalysisError(msg)

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
                centroid=self.moment_centroid,
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
            m_x += force * (centroid[1] - self.moment_centroid[1])
            m_y += force * (centroid[0] - self.moment_centroid[0])

            # calculate k_u
            d = ef_v - c_v
            k_u.append(d_n / d)

        # calculate resultant moment
        m_xy = np.sqrt(m_x * m_x + m_y * m_y)

        # save results
        ultimate_results.d_n = d_n
        ultimate_results.n = n
        ultimate_results.m_x = m_x
        ultimate_results.m_y = m_y
        ultimate_results.m_xy = m_xy

        if k_u:
            ultimate_results.k_u = min(k_u)

        return ultimate_results

    def moment_interaction_diagram(
        self,
        theta: float = 0,
        limits: List[Tuple[str, float]] = [
            ("D", 1.0),
            ("d_n", 1e-6),
        ],
        control_points: List[Tuple[str, float]] = [
            ("kappa0", 0.0),
            ("fy", 1.0),
            ("N", 0.0),
        ],
        labels: Optional[List[str]] = None,
        n_points: int = 24,
        n_spacing: Optional[int] = None,
        max_comp: Optional[float] = None,
        max_comp_labels: Optional[List[str]] = None,
        progress_bar: bool = True,
    ) -> res.MomentInteractionResults:
        r"""Generates a moment interaction diagram given a neutral axis angle ``theta``.

        ``limits`` and ``control_points`` accept a list of tuples that define points on
        the moment interaction diagram. The first item in the tuple defines the type of
        control points, while the second item defines the location of the control point.
        Types of control points are detailed below:

        .. admonition:: Control points

          - ``"D"`` - ratio of neutral axis depth to section depth
          - ``"d_n"`` - neutral axis depth
          - ``"fy"`` - yield ratio of the most extreme tensile bar
          - ``"N"`` - axial force
          - ``"kappa0"`` - zero curvature compression (N.B second item in tuple is not
            used)

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :param limits: List of control points that define the start and end of the
            interaction diagram. List length must equal two. The default limits range
            from concrete decompression strain to zero curvature tension.
        :param control_points: List of additional control points to add to the moment
            interatction diagram. The default control points include the pure
            compression point (``kappa0``), the balanced point (``fy=1``) and the pure
            bending point (``N=0``). Control points may lie outside the limits of the
            moment interaction diagram as long as equilibrium can be found.
        :param labels: List of labels to apply to the ``limits`` and ``control_points``
            for plotting purposes. The first two values in ``labels`` apply labels to
            the ``limits``, the remaining values apply labels to the ``control_points``.
            If a single value is provided, this value will be applied to both ``limits``
            and all ``control_points``. The length of ``labels`` must equal ``1`` or
            ``2 + len(control_points)``.
        :param n_points: Number of points to compute including and between the
            ``limits`` of the moment interaction diagram. Generates equally spaced
            neutral axis depths between the ``limits``.
        :param n_spacing: If provided, overrides ``n_points`` and generates the moment
            interaction diagram using ``n_spacing`` equally spaced axial loads. Note
            that using ``n_spacing`` negatively affects performance, as the neutral axis
            depth must first be located for each point on the moment interaction
            diagram.
        :param max_comp: If provided, limits the maximum compressive force in the moment
            interaction diagram to ``max_comp``
        :param max_comp_labels: Labels to apply to the ``max_comp`` intersection points,
            first value is at zero moment, second value is at the intersection with the
            interaction diagram
        :param progress_bar: If set to True, displays the progress bar

        :return: Moment interaction results object
        """

        # compute extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(
            points=self.compound_geometry.points, theta=theta
        )

        # validate limits length
        if len(limits) != 2:
            raise ValueError("Length of limits must equal 2.")

        # get neutral axis depths for limits
        limits_dn = []

        for cp in limits:
            limits_dn.append(self.decode_d_n(theta=theta, cp=cp, d_t=d_t))

        # get neutral axis depths for additional control points
        add_cp_dn = []

        for cp in control_points:
            add_cp_dn.append(self.decode_d_n(theta=theta, cp=cp, d_t=d_t))

        # validate labels length
        if labels and len(labels) != 1 and len(labels) != 2 + len(control_points):
            raise ValueError(
                "Length of labels must be 1 or 2 + number of control points"
            )

        # if one label is provided, generate a list
        if labels and len(labels) == 1:
            labels = labels * (len(control_points) + 2)

        # initialise results
        mi_results = res.MomentInteractionResults()

        # generate list of neutral axis depths/axial forces to analyse
        # if we are spacing by axial force
        if n_spacing:
            # get axial force of the limits
            start_res = self.calculate_ultimate_section_actions(
                d_n=limits_dn[0],
                ultimate_results=res.UltimateBendingResults(theta=theta),
            )
            end_res = self.calculate_ultimate_section_actions(
                d_n=limits_dn[1],
                ultimate_results=res.UltimateBendingResults(theta=theta),
            )

            # generate list of axial forces
            analysis_list = np.linspace(
                start=start_res.n, stop=end_res.n, num=n_spacing, dtype=float
            ).tolist()
        else:
            # check for infinity in limits - this will not work with linspace
            # for sake of distributing neutral axes let kappa0 ~= 2 * D
            if limits_dn[0] == inf:
                start = 2 * d_t
            else:
                start = limits_dn[0]

            if limits_dn[1] == inf:
                stop = 2 * d_t
            else:
                stop = limits_dn[1]

            # generate list of neutral axes
            analysis_list = np.linspace(
                start=start, stop=stop, num=n_points, dtype=float
            ).tolist()

        # function that performs moment interaction analysis
        def micurve(progress=None):
            # loop through all analysis points
            for idx, analysis_point in enumerate(analysis_list):
                # calculate ultimate results:
                # if we have axial forces
                if n_spacing:
                    # limits should be calculated based on neutral axis values
                    if idx == 0:
                        ult_res = self.calculate_ultimate_section_actions(
                            d_n=limits_dn[0],
                            ultimate_results=res.UltimateBendingResults(theta=theta),
                        )
                    elif idx == len(analysis_list) - 1:
                        ult_res = self.calculate_ultimate_section_actions(
                            d_n=limits_dn[1],
                            ultimate_results=res.UltimateBendingResults(theta=theta),
                        )
                    else:
                        ult_res = self.ultimate_bending_capacity(
                            theta=theta, n=analysis_point
                        )
                # if we have neutral axes
                else:
                    ult_res = self.calculate_ultimate_section_actions(
                        d_n=analysis_point,
                        ultimate_results=res.UltimateBendingResults(theta=theta),
                    )

                # add labels for limits
                if labels:
                    if idx == 0:
                        ult_res.label = labels[0]
                    elif idx == len(analysis_list) - 1:
                        ult_res.label = labels[1]

                # add ultimate result to moment interactions results
                mi_results.results.append(ult_res)

                # update progress
                if progress:
                    progress.update(task, advance=1)

            # add control points
            for idx, d_n in enumerate(add_cp_dn):
                ult_res = self.calculate_ultimate_section_actions(
                    d_n=d_n,
                    ultimate_results=res.UltimateBendingResults(theta=theta),
                )

                # add label
                if labels:
                    ult_res.label = labels[idx + 2]

                # add ultimate result to moment interactions results
                mi_results.results.append(ult_res)

                # update progress
                if progress:
                    progress.update(task, advance=1)

            # sort results
            mi_results.sort_results()

        if progress_bar:
            # create progress bar
            progress = utils.create_known_progress()

            with Live(progress, refresh_per_second=10) as live:
                # add progress bar task
                task = progress.add_task(
                    description="[red]Generating M-N diagram",
                    total=len(analysis_list) + len(control_points),
                )

                micurve(progress=progress)

                # display finished progress bar
                progress.update(
                    task,
                    description="[bold green]:white_check_mark: M-N diagram generated",
                )
                live.refresh()
        else:
            micurve()

        # cut diagram at max_comp
        if max_comp:
            # check input - if value greater than maximum compression
            if max_comp > mi_results.results[0].n:
                msg = f"max_comp={max_comp} is greater than the maximum axial capacity "
                msg += f"{mi_results.results[0].n}."
                raise ValueError(msg)

            # get max_comp point
            ult_res = self.ultimate_bending_capacity(theta=theta, n=max_comp)

            # determine which results to delete
            idx_to_keep = 0

            for idx, mi_res in enumerate(mi_results.results):
                # determine which index is the first to keep
                if idx_to_keep == 0 and mi_res.n < max_comp:
                    idx_to_keep = idx
                    break

            # remove points in diagram
            del mi_results.results[:idx_to_keep]

            # get labels
            if max_comp_labels:
                pt1_label = max_comp_labels[0]
                pt2_label = max_comp_labels[1]
            else:
                pt1_label = None
                pt2_label = None

            # add first two points to diagram
            # (m_max_comp, max_comp)
            # apply label
            ult_res.label = pt2_label
            mi_results.results.insert(0, ult_res)

            # (0, max_comp)
            mi_results.results.insert(
                0,  # insertion index
                res.UltimateBendingResults(
                    theta=theta,
                    d_n=inf,
                    k_u=0,
                    n=max_comp,
                    m_x=0,
                    m_y=0,
                    m_xy=0,
                    label=pt1_label,
                ),
            )

        return mi_results

    def biaxial_bending_diagram(
        self,
        n: float = 0,
        n_points: int = 48,
        progress_bar: bool = True,
    ) -> res.BiaxialBendingResults:
        """Generates a biaxial bending diagram given a net axial force ``n`` and
        ``n_points`` calculation points.

        :param n: Net axial force
        :param n_points: Number of calculation points
        :param progress_bar: If set to True, displays the progress bar

        :return: Biaxial bending results
        """

        # initialise results
        bb_results = res.BiaxialBendingResults(n=n)

        # calculate d_theta
        d_theta = 2 * np.pi / n_points

        # generate list of thetas
        theta_list = np.linspace(start=-np.pi, stop=np.pi - d_theta, num=n_points)

        # function that performs biaxial bending analysis
        def bbcurve(progress=None):
            # loop through thetas
            for theta in theta_list:
                ultimate_results = self.ultimate_bending_capacity(theta=theta, n=n)
                bb_results.results.append(ultimate_results)

                if progress:
                    progress.update(task, advance=1)

            # add first result to end of list top
            bb_results.results.append(bb_results.results[0])

        if progress_bar:
            # create progress bar
            progress = utils.create_known_progress()

            with Live(progress, refresh_per_second=10) as live:
                task = progress.add_task(
                    description="[red]Generating biaxial bending diagram",
                    total=n_points,
                )

                bbcurve(progress=progress)

                progress.update(
                    task,
                    description="[bold green]:white_check_mark: Biaxial bending diagram generated",
                )
                live.refresh()
        else:
            bbcurve()

        return bb_results

    def calculate_uncracked_stress(
        self,
        n: float = 0,
        m_x: float = 0,
        m_y: float = 0,
    ) -> res.StressResult:
        """Calculates stresses within the reinforced concrete section assuming an
        uncracked section.

        Uses gross area section properties to determine concrete and reinforcement
        stresses given an axial force ``n``, and bending moments ``m_x`` and ``m_y``.

        :param n: Axial force
        :param m_x: Bending moment about the x-axis
        :param m_y: Bending moment about the y-axis

        :return: Stress results object
        """

        # initialise stress results
        conc_sections = []
        conc_sigs = []
        conc_forces = []
        meshed_reinf_sections = []
        meshed_reinf_sigs = []
        meshed_reinf_forces = []
        lumped_reinf_geoms = []
        lumped_reinf_sigs = []
        lumped_reinf_strains = []
        lumped_reinf_forces = []

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

        # split meshed geometries above and below neutral axis
        split_meshed_geoms = []

        for meshed_geom in self.meshed_geometries:
            top_geoms, bot_geoms = meshed_geom.split_section(
                point=point_na,
                theta=theta,
            )

            split_meshed_geoms.extend(top_geoms)
            split_meshed_geoms.extend(bot_geoms)

        # loop through all meshed geometries and calculate stress
        for meshed_geom in split_meshed_geoms:
            analysis_section = AnalysisSection(geometry=meshed_geom)

            # calculate stress, force and point of action
            sig, n_sec, d_x, d_y = analysis_section.get_elastic_stress(
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

            # save results
            if isinstance(meshed_geom, CPGeomConcrete):
                conc_sigs.append(sig)
                conc_forces.append((n_sec, d_x, d_y))
                conc_sections.append(analysis_section)
            else:
                meshed_reinf_sigs.append(sig)
                meshed_reinf_forces.append((n_sec, d_x, d_y))
                meshed_reinf_sections.append(analysis_section)

        # loop through all lumped geometries and calculate stress
        for lumped_geom in self.reinf_geometries_lumped:
            # initialise stress and position
            sig = 0
            centroid = lumped_geom.calculate_centroid()
            x = centroid[0] - cx
            y = centroid[1] - cy

            # axial stress
            sig += n * lumped_geom.material.elastic_modulus / e_a

            # bending moment stress
            sig += lumped_geom.material.elastic_modulus * (
                -(e_ixy * m_x) / (e_ixx * e_iyy - e_ixy**2) * x
                + (e_iyy * m_x) / (e_ixx * e_iyy - e_ixy**2) * y
            )
            sig += lumped_geom.material.elastic_modulus * (
                +(e_ixx * m_y) / (e_ixx * e_iyy - e_ixy**2) * x
                - (e_ixy * m_y) / (e_ixx * e_iyy - e_ixy**2) * y
            )
            strain = sig / lumped_geom.material.elastic_modulus

            # net force and point of action
            n_lumped = sig * lumped_geom.calculate_area()

            lumped_reinf_sigs.append(sig)
            lumped_reinf_strains.append(strain)
            lumped_reinf_forces.append((n_lumped, x, y))
            lumped_reinf_geoms.append(lumped_geom)

        return res.StressResult(
            concrete_section=self,
            concrete_analysis_sections=conc_sections,
            concrete_stresses=conc_sigs,
            concrete_forces=conc_forces,
            meshed_reinforcement_sections=meshed_reinf_sections,
            meshed_reinforcement_stresses=meshed_reinf_sigs,
            meshed_reinforcement_forces=meshed_reinf_forces,
            lumped_reinforcement_geometries=lumped_reinf_geoms,
            lumped_reinforcement_stresses=lumped_reinf_sigs,
            lumped_reinforcement_strains=lumped_reinf_strains,
            lumped_reinforcement_forces=lumped_reinf_forces,
        )

    def calculate_cracked_stress(
        self,
        cracked_results: res.CrackedResults,
        n: float = 0,
        m: float = 0,
    ) -> res.StressResult:
        """Calculates stresses within the reinforced concrete section assuming a cracked
        section.

        Uses cracked area section properties to determine concrete and reinforcement
        stresses given an axial force ``n`` and bending moment ``m`` about the bending
        axis stored in ``cracked_results``.

        :param cracked_results: Cracked results objects
        :param n: Axial force
        :param m: Bending moment

        :return: Stress results object
        """

        # initialise stress results
        conc_sections = []
        conc_sigs = []
        conc_forces = []
        meshed_reinf_sections = []
        meshed_reinf_sigs = []
        meshed_reinf_forces = []
        lumped_reinf_geoms = []
        lumped_reinf_sigs = []
        lumped_reinf_strains = []
        lumped_reinf_forces = []

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

        # loop through all meshed geometries and calculate stress
        for geom in cracked_results.cracked_geometries:
            if geom.material.meshed:
                analysis_section = AnalysisSection(geometry=geom)

                # calculate stress, force and point of action
                sig, n_sec, d_x, d_y = analysis_section.get_elastic_stress(
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

                # save results
                if isinstance(geom, CPGeomConcrete):
                    conc_sigs.append(sig)
                    conc_forces.append((n_sec, d_x, d_y))
                    conc_sections.append(analysis_section)
                else:
                    meshed_reinf_sigs.append(sig)
                    meshed_reinf_forces.append((n_sec, d_x, d_y))
                    meshed_reinf_sections.append(analysis_section)

        # loop through all lumped geometries and calculate stress
        for lumped_geom in self.reinf_geometries_lumped:
            # initialise stress and position of bar
            sig = 0
            centroid = lumped_geom.calculate_centroid()
            x = centroid[0] - cx
            y = centroid[1] - cy

            # axial stress
            sig += n * lumped_geom.material.elastic_modulus / e_a

            # bending moment stress
            sig += lumped_geom.material.elastic_modulus * (
                -(e_ixy * m_x) / (e_ixx * e_iyy - e_ixy**2) * x
                + (e_iyy * m_x) / (e_ixx * e_iyy - e_ixy**2) * y
            )
            sig += lumped_geom.material.elastic_modulus * (
                +(e_ixx * m_y) / (e_ixx * e_iyy - e_ixy**2) * x
                - (e_ixy * m_y) / (e_ixx * e_iyy - e_ixy**2) * y
            )
            strain = sig / lumped_geom.material.elastic_modulus

            # net force and point of action
            n_lumped = sig * lumped_geom.calculate_area()

            lumped_reinf_sigs.append(sig)
            lumped_reinf_strains.append(strain)
            lumped_reinf_forces.append((n_lumped, x, y))
            lumped_reinf_geoms.append(lumped_geom)

        return res.StressResult(
            concrete_section=self,
            concrete_analysis_sections=conc_sections,
            concrete_stresses=conc_sigs,
            concrete_forces=conc_forces,
            meshed_reinforcement_sections=meshed_reinf_sections,
            meshed_reinforcement_stresses=meshed_reinf_sigs,
            meshed_reinforcement_forces=meshed_reinf_forces,
            lumped_reinforcement_geometries=lumped_reinf_geoms,
            lumped_reinforcement_stresses=lumped_reinf_sigs,
            lumped_reinforcement_strains=lumped_reinf_strains,
            lumped_reinforcement_forces=lumped_reinf_forces,
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
        within the section. Otherwise, a curvature can be provided which overrides the
        supplied moment.

        :param moment_curvature_results: Moment-curvature results objects
        :param m: Bending moment
        :param kappa: Curvature, if provided overrides the supplied bending moment and
            calculates the stress at the given curvature

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
            points=self.compound_geometry.points, theta=theta
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
            msg = "Analysis failed. Confirm that the supplied moment/curvature is "
            msg += "within the range of the moment-curvature analysis."
            raise utils.AnalysisError(msg)

        # initialise stress results
        conc_sections = []
        conc_sigs = []
        conc_forces = []
        meshed_reinf_sections = []
        meshed_reinf_sigs = []
        meshed_reinf_forces = []
        lumped_reinf_geoms = []
        lumped_reinf_sigs = []
        lumped_reinf_strains = []
        lumped_reinf_forces = []

        # find point on neutral axis by shifting by d_n
        point_na = utils.point_on_neutral_axis(
            extreme_fibre=extreme_fibre, d_n=d_n, theta=theta
        )

        # create splits in meshed geometries at points in stress-strain profiles
        meshed_split_geoms: List[Union[CPGeom, CPGeomConcrete]] = []

        for meshed_geom in self.meshed_geometries:
            split_geoms = utils.split_geom_at_strains(
                geom=meshed_geom,
                theta=theta,
                point_na=point_na,
                ultimate=False,
                kappa=kappa,
            )

            meshed_split_geoms.extend(split_geoms)

        # loop through all meshed geometries and calculate stress
        for meshed_geom in meshed_split_geoms:
            analysis_section = AnalysisSection(geometry=meshed_geom)

            # calculate stress, force and point of action
            sig, n_sec, d_x, d_y = analysis_section.get_service_stress(
                d_n=d_n,
                kappa=kappa,
                point_na=point_na,
                theta=theta,
                centroid=self.moment_centroid,
            )

            # save results
            if isinstance(meshed_geom, CPGeomConcrete):
                conc_sigs.append(sig)
                conc_forces.append((n_sec, d_x, d_y))
                conc_sections.append(analysis_section)
            else:
                meshed_reinf_sigs.append(sig)
                meshed_reinf_forces.append((n_sec, d_x, d_y))
                meshed_reinf_sections.append(analysis_section)

        # loop through all lumped geometries and calculate stress
        for lumped_geom in self.reinf_geometries_lumped:
            # get position of geometry
            centroid = lumped_geom.calculate_centroid()

            # get strain at centroid of lump
            strain = utils.get_service_strain(
                point=(centroid[0], centroid[1]),
                point_na=point_na,
                theta=theta,
                kappa=kappa,
            )

            # calculate stress, force and point of action
            sig = lumped_geom.material.stress_strain_profile.get_stress(strain=strain)
            n_lumped = sig * lumped_geom.calculate_area()

            lumped_reinf_sigs.append(sig)
            lumped_reinf_strains.append(strain)
            lumped_reinf_forces.append(
                (
                    n_lumped,
                    centroid[0] - self.moment_centroid[0],
                    centroid[1] - self.moment_centroid[1],
                )
            )
            lumped_reinf_geoms.append(lumped_geom)

        return res.StressResult(
            concrete_section=self,
            concrete_analysis_sections=conc_sections,
            concrete_stresses=conc_sigs,
            concrete_forces=conc_forces,
            meshed_reinforcement_sections=meshed_reinf_sections,
            meshed_reinforcement_stresses=meshed_reinf_sigs,
            meshed_reinforcement_forces=meshed_reinf_forces,
            lumped_reinforcement_geometries=lumped_reinf_geoms,
            lumped_reinforcement_stresses=lumped_reinf_sigs,
            lumped_reinforcement_strains=lumped_reinf_strains,
            lumped_reinforcement_forces=lumped_reinf_forces,
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
            points=self.compound_geometry.points, theta=ultimate_results.theta
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
        conc_sections = []
        conc_sigs = []
        conc_forces = []
        meshed_reinf_sections = []
        meshed_reinf_sigs = []
        meshed_reinf_forces = []
        lumped_reinf_geoms = []
        lumped_reinf_sigs = []
        lumped_reinf_strains = []
        lumped_reinf_forces = []

        # create splits in meshed geometries at points in stress-strain profiles
        meshed_split_geoms: List[Union[CPGeom, CPGeomConcrete]] = []

        if isinf(ultimate_results.d_n):
            meshed_split_geoms = self.meshed_geometries
        else:
            for meshed_geom in self.meshed_geometries:
                split_geoms = utils.split_geom_at_strains(
                    geom=meshed_geom,
                    theta=ultimate_results.theta,
                    point_na=point_na,
                    ultimate=True,
                    ultimate_strain=self.gross_properties.conc_ultimate_strain,
                    d_n=ultimate_results.d_n,
                )

                meshed_split_geoms.extend(split_geoms)

        # loop through all concrete geometries and calculate stress
        for meshed_geom in meshed_split_geoms:
            analysis_section = AnalysisSection(geometry=meshed_geom)

            # calculate stress, force and point of action
            sig, n_sec, d_x, d_y = analysis_section.get_ultimate_stress(
                d_n=ultimate_results.d_n,
                point_na=point_na,
                theta=ultimate_results.theta,
                ultimate_strain=self.gross_properties.conc_ultimate_strain,
                centroid=self.moment_centroid,
            )

            # save results
            if isinstance(meshed_geom, CPGeomConcrete):
                conc_sigs.append(sig)
                conc_forces.append((n_sec, d_x, d_y))
                conc_sections.append(analysis_section)
            else:
                meshed_reinf_sigs.append(sig)
                meshed_reinf_forces.append((n_sec, d_x, d_y))
                meshed_reinf_sections.append(analysis_section)

        # loop through all lumped geometries and calculate stress
        for lumped_geom in self.reinf_geometries_lumped:
            # get position of lump
            centroid = lumped_geom.calculate_centroid()

            # get strain at centroid of lump
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
            sig = lumped_geom.material.stress_strain_profile.get_stress(strain=strain)
            n_lumped = sig * lumped_geom.calculate_area()

            lumped_reinf_sigs.append(sig)
            lumped_reinf_strains.append(strain)
            lumped_reinf_forces.append(
                (
                    n_lumped,
                    centroid[0] - self.moment_centroid[0],
                    centroid[1] - self.moment_centroid[1],
                )
            )
            lumped_reinf_geoms.append(lumped_geom)

        return res.StressResult(
            concrete_section=self,
            concrete_analysis_sections=conc_sections,
            concrete_stresses=conc_sigs,
            concrete_forces=conc_forces,
            meshed_reinforcement_sections=meshed_reinf_sections,
            meshed_reinforcement_stresses=meshed_reinf_sigs,
            meshed_reinforcement_forces=meshed_reinf_forces,
            lumped_reinforcement_geometries=lumped_reinf_geoms,
            lumped_reinforcement_stresses=lumped_reinf_sigs,
            lumped_reinforcement_strains=lumped_reinf_strains,
            lumped_reinforcement_forces=lumped_reinf_forces,
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

    def decode_d_n(
        self,
        theta: float,
        cp: Tuple[str, float],
        d_t: float,
    ) -> float:
        r"""Decodes a neutral axis depth given a control point ``cp``.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :param cp: Control point to decode
        :param d_t: Depth to extreme tensile fibre

        :return: Decoded neutral axis depth
        """

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
                raise ValueError(f"Provided d_n {cp[1]:.3f} must be greater than zero.")

            return cp[1]

        # extreme tensile reinforcement yield ratio
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
            return inf

        # control point type not valid
        else:
            msg = "First value of control point tuple must be D, d_n, fy, N or "
            msg += "kappa0."
            raise ValueError(msg)

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

        with plotting_context(title=title, aspect=True, **kwargs) as (fig, ax):
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
                coords = list(lumped_geom.geom.exterior.coords)  # type: ignore
                bar = mpatches.Polygon(
                    xy=coords, closed=False, color=lumped_geom.material.colour
                )
                ax.add_patch(bar)  # type: ignore

            if not background:
                ax.legend(  # type: ignore
                    loc="center left", bbox_to_anchor=(1, 0.5), handles=legend_labels
                )

        return ax
