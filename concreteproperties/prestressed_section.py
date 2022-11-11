from __future__ import annotations

from typing import List, Union

import numpy as np
import sectionproperties.pre.geometry as sp_geom
from scipy.optimize import brentq

import concreteproperties.results as res
import concreteproperties.utils as utils
from concreteproperties.analysis_section import AnalysisSection
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.material import SteelStrand
from concreteproperties.pre import CPGeom, CPGeomConcrete


class PrestressedSection(ConcreteSection):
    """Class for a prestressed concrete section.

    Note that the section must be symmetric about its vertical (``y``) axis and all
    bending is assumed to be about the ``x`` axis.

    The only meshed geometries that are permitted are concrete geometries.
    """

    def __init__(
        self,
        geometry: sp_geom.CompoundGeometry,
        calculate_prestress_actions: bool = True,
    ) -> None:
        """Inits the ConcreteSection class.

        :param geometry: *sectionproperties* CompoundGeometry object describing the
            prestressed concrete section
        :param calculate_prestress_actions: If set to True, adds the prestressed axial
            load and induced moment to all actions
        """

        super().__init__(geometry=geometry)

        self.calculate_prestressed_actions = calculate_prestress_actions

        # check symmetry about y-axis
        if not np.isclose(
            self.gross_properties.e_zyy_minus, self.gross_properties.e_zyy_plus
        ):
            raise ValueError("PrestressedSection must be symmetric about y-axis.")

        # check for any meshed geometries
        if self.reinf_geometries_meshed:
            msg = "Meshed reinforcement geometries are not permitted in "
            msg += "PrestressedSection."
            raise ValueError(msg)

    def calculate_gross_area_properties(self) -> None:
        """Calculates and stores gross section area properties."""

        super().calculate_gross_area_properties()

        # sum strand areas
        for strand_geom in self.strand_geometries:
            self.gross_properties.strand_area += strand_geom.calculate_area()

        # calculate prestressed actions
        n_prestress = 0
        m_prestress = 0

        for strand in self.strand_geometries:
            if isinstance(strand.material, SteelStrand):
                # add axial force
                n_strand = strand.material.prestress_force
                n_prestress += n_strand

                # add moment
                centroid = strand.calculate_centroid()
                # TODO: fix with moment_centroid
                m_prestress += n_strand * (centroid[1] - self.gross_properties.cy)

        self.gross_properties.n_prestress = n_prestress
        self.gross_properties.m_prestress = m_prestress

    def calculate_cracked_properties(
        self,
        m_ext: float,
        n_ext: float = 0,
    ) -> res.CrackedResults:
        """Calculates cracked section properties given an axial loading and bending
        moment.

        For a cracked analysis, prestressing actions are included irrespective of the
        value of ``self.calculate_prestressed_actions``.

        :param n_ext: External axial force
        :param m_ext: External bending moment

        :return: Cracked results object
        """

        # initialise cracked results object
        cracked_results = res.CrackedResults(
            theta=0,
            n=self.gross_properties.n_prestress + n_ext,
            m=m_ext,
        )

        # calculate cracking moment
        cracked_results.m_cr = self.calculate_cracking_moment(
            n=self.gross_properties.n_prestress + n_ext,
            m_int=self.gross_properties.m_prestress,
        )

        # set neutral axis depth limits
        # depth of neutral axis at extreme tensile fibre
        _, d_t = utils.calculate_extreme_fibre(
            points=self.compound_geometry.points, theta=0
        )
        a = 1e-6 * d_t  # sufficiently small depth of compressive zone
        b = d_t  # neutral axis at extreme tensile fibre

        # find neutral axis that gives convergence of the the cracked neutral axis
        try:
            cracked_results.d_nc, r = brentq(
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

        return cracked_results

    def calculate_cracking_moment(
        self,
        n: float,
        m_int: float,
    ) -> float:
        """Calculates the cracking moment given an axial load ``n`` and internal bending
        moment ``m_int``.

        :param n: Axial load
        :param m_int: Internal bending moment

        :return: Cracking moment
        """

        # get centroidal second moments of area
        e_ixx = self.gross_properties.e_ixx_c

        # loop through all concrete geometries to find lowest cracking moment
        m_c = 0

        for idx, conc_geom in enumerate(self.concrete_geometries):
            # get distance from centroid to extreme tensile fibre
            d = utils.calculate_max_bending_depth(
                points=conc_geom.points,
                c_local_v=utils.global_to_local(
                    theta=0, x=self.gross_properties.cx, y=self.gross_properties.cy
                )[1],
                theta=0,
            )

            # if no part of the section is in tension, go to next geometry
            if d == 0:
                continue

            # calculate stress required for cracking
            f_t = conc_geom.material.flexural_tensile_strength
            f_n = n * conc_geom.material.elastic_modulus / self.gross_properties.e_a
            f_r = f_t + f_n

            # cracking moment for this geometry
            m_c_geom = (f_r / conc_geom.material.elastic_modulus) * (e_ixx / d) - m_int

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
        """Given a trial cracked neutral axis depth ``d_nc``, determines the minimum
        concrete stress. For a cracked elastic analysis this should be zero (no tension
        allowed).

        :param d_nc: Trial cracked neutral axis
        :param cracked_results: Cracked results object

        :return: Cracked neutral axis convergence
        """

        # calculate extreme fibre in global coordinates
        extreme_fibre, d_t = utils.calculate_extreme_fibre(
            points=self.compound_geometry.points, theta=0
        )

        # validate d_nc input
        if d_nc <= 0:
            raise ValueError("d_nc must be positive.")
        elif d_nc > d_t:
            raise ValueError("d_nc must lie within the section, i.e. d_nc <= d_t")

        # find point on neutral axis by shifting by d_nc
        point_na = utils.point_on_neutral_axis(
            extreme_fibre=extreme_fibre, d_n=d_nc, theta=0
        )

        # split concrete geometries above and below d_nc, discard below
        cracked_geoms: List[Union[CPGeomConcrete, CPGeom]] = []

        for conc_geom in self.concrete_geometries:
            top_geoms, _ = conc_geom.split_section(point=point_na, theta=0)

            # save compression geometries
            cracked_geoms.extend(top_geoms)

        # add reinforcement geometries to list
        cracked_geoms.extend(self.reinf_geometries_lumped)
        cracked_geoms.extend(self.strand_geometries)

        # save cracked geometries and calculate properties
        cracked_results.cracked_geometries = cracked_geoms
        self.cracked_section_properties(cracked_results=cracked_results)

        # conduct cracked stress analysis
        cr_stress_res = self.calculate_cracked_stress(cracked_results=cracked_results)

        # get minimum concrete stress is zero
        min_stress, _ = cr_stress_res.get_concrete_stress_limits()

        return min_stress

    def moment_curvature_analysis(self):
        raise NotImplementedError

    def ultimate_bending_capacity(self):
        raise NotImplementedError

    def moment_interaction_diagram(self):
        raise NotImplementedError

    def biaxial_bending_diagram(self):
        raise NotImplementedError

    def calculate_uncracked_stress(
        self,
        n: float = 0,
        m: float = 0,
    ) -> res.StressResult:
        """Calculates stresses within the prestressed concrete section assuming an
        uncracked section.

        Uses gross area section properties to determine concrete, reinforcement and
        strand stresses given an axial force ``n`` and bending moment ``m``.

        If ``self.calculate_prestressed_actions=True``, prestressed
        actions are included in the analysis. If
        ``self.calculate_prestressed_actions=False``, prestressed actions must be added
        to ``n`` and ``m``.

        :param n: Axial force
        :param m: Bending moment

        :return: Stress results object
        """

        # initialise stress results
        conc_sections = []
        conc_sigs = []
        conc_forces = []
        lumped_reinf_geoms = []
        lumped_reinf_sigs = []
        lumped_reinf_strains = []
        lumped_reinf_forces = []
        strand_geoms = []
        strand_sigs = []
        strand_strains = []
        strand_forces = []

        # get uncracked section properties
        e_a = self.gross_properties.e_a
        cx = self.gross_properties.cx
        cy = self.gross_properties.cy
        e_ixx = self.gross_properties.e_ixx_c
        e_iyy = self.gross_properties.e_iyy_c
        e_ixy = self.gross_properties.e_ixy_c

        # add prestressed actions
        if self.calculate_prestressed_actions:
            n += self.gross_properties.n_prestress
            m += self.gross_properties.m_prestress

        # calculate neutral axis rotation
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

        # loop through all concrete geometries and calculate stress
        # for conc_geom in self.concrete_geometries:
        for conc_geom in split_meshed_geoms:
            analysis_section = AnalysisSection(geometry=conc_geom)

            # calculate stress, force and point of action
            sig, n_sec, d_x, d_y = analysis_section.get_elastic_stress(
                n=n,
                m_x=m,
                m_y=0,
                e_a=e_a,
                cx=cx,
                cy=cy,
                e_ixx=e_ixx,
                e_iyy=e_iyy,
                e_ixy=e_ixy,
            )

            # save results
            conc_sigs.append(sig)
            conc_forces.append((n_sec, d_x, d_y))
            conc_sections.append(analysis_section)

        # loop through all lumped and strand geometries and calculate stress
        for lumped_geom in self.reinf_geometries_lumped + self.strand_geometries:
            # initialise stress and position
            sig = 0
            centroid = lumped_geom.calculate_centroid()
            x = centroid[0] - cx
            y = centroid[1] - cy

            # axial stress
            sig += n * lumped_geom.material.elastic_modulus / e_a

            # bending moment stress
            sig += lumped_geom.material.elastic_modulus * (
                -(e_ixy * m) / (e_ixx * e_iyy - e_ixy**2) * x
                + (e_iyy * m) / (e_ixx * e_iyy - e_ixy**2) * y
            )

            # add initial prestress
            if isinstance(lumped_geom.material, SteelStrand):
                sig += (
                    -lumped_geom.material.prestress_force / lumped_geom.calculate_area()
                )

            strain = sig / lumped_geom.material.elastic_modulus

            # net force and point of action
            n_lumped = sig * lumped_geom.calculate_area()

            if isinstance(lumped_geom.material, SteelStrand):
                strand_sigs.append(sig)
                strand_strains.append(strain)
                strand_forces.append((n_lumped, x, y))
                strand_geoms.append(lumped_geom)
            else:
                lumped_reinf_sigs.append(sig)
                lumped_reinf_strains.append(strain)
                lumped_reinf_forces.append((n_lumped, x, y))
                lumped_reinf_geoms.append(lumped_geom)

        return res.StressResult(
            concrete_section=self,
            concrete_analysis_sections=conc_sections,
            concrete_stresses=conc_sigs,
            concrete_forces=conc_forces,
            meshed_reinforcement_sections=[],
            meshed_reinforcement_stresses=[],
            meshed_reinforcement_forces=[],
            lumped_reinforcement_geometries=lumped_reinf_geoms,
            lumped_reinforcement_stresses=lumped_reinf_sigs,
            lumped_reinforcement_strains=lumped_reinf_strains,
            lumped_reinforcement_forces=lumped_reinf_forces,
            strand_geometries=strand_geoms,
            strand_stresses=strand_sigs,
            strand_strains=strand_strains,
            strand_forces=strand_forces,
        )

    def calculate_cracked_stress(
        self,
        cracked_results: res.CrackedResults,
    ) -> res.StressResult:
        """Calculates stresses within the prestressed concrete section assuming a
        cracked section.

        Uses cracked area section properties to determine concrete, reinforcement and
        strand stresses given the actions provided during the cracked analysis.

        :param cracked_results: Cracked results objects

        :return: Stress results object
        """

        # initialise stress results
        conc_sections = []
        conc_sigs = []
        conc_forces = []
        lumped_reinf_geoms = []
        lumped_reinf_sigs = []
        lumped_reinf_strains = []
        lumped_reinf_forces = []
        strand_geoms = []
        strand_sigs = []
        strand_strains = []
        strand_forces = []

        # get cracked section properties
        e_a = cracked_results.e_a_cr
        cx = cracked_results.cx
        cy = cracked_results.cy
        e_ixx = cracked_results.e_ixx_c_cr
        e_iyy = cracked_results.e_iyy_c_cr
        e_ixy = cracked_results.e_ixy_c_cr

        # determine net moment
        # (recalculate moment due to prestressing force about cracked centroid)
        m_net = 0

        for strand in self.strand_geometries:
            if isinstance(strand.material, SteelStrand):
                n_strand = strand.material.prestress_force
                centroid = strand.calculate_centroid()
                m_net += n_strand * (centroid[1] - cy)

        m_net += cracked_results.m

        # loop through all meshed geometries and calculate stress
        for geom in cracked_results.cracked_geometries:
            if geom.material.meshed:
                analysis_section = AnalysisSection(geometry=geom)

                # calculate stress, force and point of action
                sig, n_sec, d_x, d_y = analysis_section.get_elastic_stress(
                    n=cracked_results.n,
                    m_x=m_net,
                    m_y=0,
                    e_a=e_a,
                    cx=cx,
                    cy=cy,
                    e_ixx=e_ixx,
                    e_iyy=e_iyy,
                    e_ixy=e_ixy,
                )

                # save results
                conc_sigs.append(sig)
                conc_forces.append((n_sec, d_x, d_y))
                conc_sections.append(analysis_section)

        # loop through all lumped and strand geometries and calculate stress
        for lumped_geom in self.reinf_geometries_lumped + self.strand_geometries:
            # initialise stress and position of bar
            sig = 0
            centroid = lumped_geom.calculate_centroid()
            x = centroid[0] - cx
            y = centroid[1] - cy

            # axial stress
            sig += cracked_results.n * lumped_geom.material.elastic_modulus / e_a

            # bending moment stress
            sig += lumped_geom.material.elastic_modulus * (
                -(e_ixy * m_net) / (e_ixx * e_iyy - e_ixy**2) * x
                + (e_iyy * m_net) / (e_ixx * e_iyy - e_ixy**2) * y
            )

            # add initial prestress
            if isinstance(lumped_geom.material, SteelStrand):
                sig += (
                    -lumped_geom.material.prestress_force / lumped_geom.calculate_area()
                )

            strain = sig / lumped_geom.material.elastic_modulus

            # net force and point of action
            n_lumped = sig * lumped_geom.calculate_area()

            if isinstance(lumped_geom.material, SteelStrand):
                strand_sigs.append(sig)
                strand_strains.append(strain)
                strand_forces.append((n_lumped, x, y))
                strand_geoms.append(lumped_geom)
            else:
                lumped_reinf_sigs.append(sig)
                lumped_reinf_strains.append(strain)
                lumped_reinf_forces.append((n_lumped, x, y))
                lumped_reinf_geoms.append(lumped_geom)

        return res.StressResult(
            concrete_section=self,
            concrete_analysis_sections=conc_sections,
            concrete_stresses=conc_sigs,
            concrete_forces=conc_forces,
            meshed_reinforcement_sections=[],
            meshed_reinforcement_stresses=[],
            meshed_reinforcement_forces=[],
            lumped_reinforcement_geometries=lumped_reinf_geoms,
            lumped_reinforcement_stresses=lumped_reinf_sigs,
            lumped_reinforcement_strains=lumped_reinf_strains,
            lumped_reinforcement_forces=lumped_reinf_forces,
            strand_geometries=strand_geoms,
            strand_stresses=strand_sigs,
            strand_strains=strand_strains,
            strand_forces=strand_forces,
        )

    def calculate_service_stress(self):
        raise NotImplementedError

    def calculate_ultimate_stress(self):
        raise NotImplementedError
