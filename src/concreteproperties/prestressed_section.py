"""Class for a prestressed concrete section."""

from __future__ import annotations

from math import isinf

import numpy as np
import sectionproperties.pre.geometry as sp_geom
from scipy.optimize import brentq, root_scalar

import concreteproperties.results as res
import concreteproperties.utils as utils
from concreteproperties.analysis_section import AnalysisSection
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.material import SteelStrand
from concreteproperties.pre import CPGeom, CPGeomConcrete


class PrestressedSection(ConcreteSection):
    """Class for a prestressed concrete section.

    .. note::

        Prestressed concrete sections analysed in ``concreteproperties`` must be
        symmetric about their vertical (``y``) axis, with all flexure assumed to be
        about the ``x`` axis.

    .. warning::

        The only meshed geometries that are permitted are concrete geometries.
    """

    def __init__(
        self,
        geometry: sp_geom.CompoundGeometry,
        moment_centroid: tuple[float, float] | None = None,
        geometric_centroid_override: bool = True,
    ) -> None:
        """Inits the ConcreteSection class.

        Args:
            geometry: ``sectionproperties`` ``CompoundGeometry`` object describing the
                prestressed concrete section
            moment_centroid: If specified, all moments for service and ultimate
                analyses are calculated about this point. If not specified, all moments
                are calculated about the gross cross-section centroid, i.e. no material
                properties applied.
            geometric_centroid_override: If set to True, sets ``moment_centroid`` to
                the geometric centroid i.e. material properties applied

        Raises:
            ValueError: If the section is not symmetric about the y-axis
            ValueError: If meshed reinforcement geometries are provided
        """
        super().__init__(
            geometry=geometry,
            moment_centroid=moment_centroid,
            geometric_centroid_override=geometric_centroid_override,
        )

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

        # sum strand areas
        for strand_geom in self.strand_geometries:
            self.gross_properties.strand_area += strand_geom.calculate_area()

        # calculate prestressed actions
        n_prestress = 0
        m_prestress = 0

        for strand in self.strand_geometries:
            if isinstance(strand.material, SteelStrand):
                # add axial force
                n_strand = (
                    strand.material.get_prestress_stress() * strand.calculate_area()
                )
                n_prestress += n_strand

                # add moment
                centroid = strand.calculate_centroid()
                m_prestress += n_strand * (centroid[1] - self.moment_centroid[1])

        self.gross_properties.n_prestress = n_prestress
        self.gross_properties.m_prestress = m_prestress

    def calculate_cracked_properties(
        self,
        m_ext: float,
        n_ext: float = 0,
    ) -> res.CrackedResults:
        """Calculate cracked section properties.

        Calculates cracked section properties given an axial loading and bending moment.

        Args:
            m_ext: External bending moment
            n_ext: External axial force

        Raises:
            AnalysisError: If the provided loads do not result in tension within the
                concrete

        Returns:
            Cracked results object
        """
        # check there is tension in the section
        uncr_stress = self.calculate_uncracked_stress(n=n_ext, m=m_ext)

        if uncr_stress.get_concrete_stress_limits()[0] > 0:
            msg = "Analysis failed, section contains no tension. Please provide a "
            msg += "combination of m_ext and n_ext that results in a tensile stress "
            msg += "within the section when combined with the prestressing actions."
            raise utils.AnalysisError(msg)

        # initialise cracked results object
        cracked_results = res.CrackedResults(
            theta=0,
            n=self.gross_properties.n_prestress + n_ext,
            m=m_ext,
        )

        # calculate cracking moment
        m_cr_pos = self.calculate_cracking_moment(
            n=self.gross_properties.n_prestress + n_ext,
            m_int=self.gross_properties.m_prestress,
            positive=True,
        )

        m_cr_neg = self.calculate_cracking_moment(
            n=self.gross_properties.n_prestress + n_ext,
            m_int=self.gross_properties.m_prestress,
            positive=False,
        )

        cracked_results.m_cr = (m_cr_pos, -m_cr_neg)

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
                rtol=1e-6,
                full_output=True,
                disp=False,
            )
        except ValueError as exc:
            msg = "Analysis failed. Please raise an issue at "
            msg += "https://github.com/robbievanleeuwen/concrete-properties/issues"
            raise utils.AnalysisError(msg) from exc

        return cracked_results

    def calculate_cracking_moment(
        self,
        n: float,
        m_int: float,
        positive: bool,
    ) -> float:
        """Calculates the cracking moment.

        Calculates the cracking moment given an axial load ``n`` and internal bending
        moment ``m_int``.

        Args:
            n: Axial load
            m_int: Internal bending moment
            positive: If set to True, determines the cracking moment for positive
                bending, otherwise determines the cracking moment for negative bending

        Returns:
            Cracking moment
        """
        # determine theta
        theta = 0 if positive else np.pi

        # get centroidal second moments of area
        e_ixx = self.gross_properties.e_ixx_c

        # loop through all concrete geometries to find lowest cracking moment
        m_c = 0
        valid_geom_count = 0

        for conc_geom in self.concrete_geometries:
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

            # calculate stress required for cracking
            f_t = conc_geom.material.flexural_tensile_strength
            f_n = n * conc_geom.material.elastic_modulus / self.gross_properties.e_a
            f_r = f_t + f_n

            # cracking moment for this geometry
            m_int_sign = -1 if positive else 1
            m_c_geom = (f_r / conc_geom.material.elastic_modulus) * (
                e_ixx / d
            ) + m_int_sign * m_int

            # if we are the first geometry, initialise cracking moment
            if valid_geom_count == 0:
                m_c = m_c_geom
            # otherwise take smaller cracking moment
            else:
                m_c = min(m_c, m_c_geom)

            valid_geom_count += 1

        return m_c

    def cracked_neutral_axis_convergence(
        self,
        d_nc: float,
        cracked_results: res.CrackedResults,
    ) -> float:
        """Calculates cracked netural axis convergence.

        Given a trial cracked neutral axis depth ``d_nc``, determines the minimum
        concrete stress. For a cracked elastic analysis this should be zero (no tension
        allowed).

        Args:
            d_nc: Trial cracked neutral axis
            cracked_results: Cracked results object

        Returns:
            Cracked neutral axis convergence
        """
        # guess hogging or sagging
        m_net_guess = cracked_results.m + self.gross_properties.m_prestress

        if m_net_guess > 0:
            theta = 0
        else:
            theta = np.pi

        def calc_min_stress():
            # calculate extreme fibre in global coordinates
            extreme_fibre, d_t = utils.calculate_extreme_fibre(
                points=self.compound_geometry.points, theta=theta
            )

            # find point on neutral axis by shifting by d_nc
            point_na = utils.point_on_neutral_axis(
                extreme_fibre=extreme_fibre, d_n=d_nc, theta=theta
            )

            # split concrete geometries above and below d_nc, discard below
            cracked_geoms: list[CPGeomConcrete | CPGeom] = []

            for conc_geom in self.concrete_geometries:
                top_geoms, _ = conc_geom.split_section(point=point_na, theta=theta)

                # save compression geometries
                cracked_geoms.extend(top_geoms)

            # add reinforcement geometries to list
            cracked_geoms.extend(self.reinf_geometries_lumped)
            cracked_geoms.extend(self.strand_geometries)

            # save cracked geometries and calculate properties
            cracked_results.cracked_geometries = cracked_geoms
            self.cracked_section_properties(cracked_results=cracked_results)

            # conduct cracked stress analysis
            return self.calculate_cracked_stress(cracked_results=cracked_results)

        cr_stress_res = calc_min_stress()

        # check _m_net exists
        if cr_stress_res._m_net:
            m_net = cr_stress_res._m_net
            # if we guess the bending direction wrong
            if m_net > 0 and m_net_guess < 0 or m_net < 0 and m_net_guess > 0:
                # change bending direction
                theta -= np.pi

                cr_stress_res = calc_min_stress()

        # get minimum concrete stress is zero
        min_stress, _ = cr_stress_res.get_concrete_stress_limits()

        return min_stress

    def moment_curvature_analysis(
        self,
        positive: bool = True,
        n: float = 0,
        kappa_inc: float = 1e-7,
        kappa_mult: float = 2,
        kappa_inc_max: float = 5e-6,
        delta_m_min: float = 0.15,
        delta_m_max: float = 0.3,
        progress_bar: bool = True,
    ) -> res.MomentCurvatureResults:
        """Performs a moment curvature analysis given an applied axial force ``n``.

        Analysis continues until a material reaches its ultimate strain.

        Args:
            positive: If set to True, performs the moment curvature analysis for
                positive bending, otherwise performs the moment curvature analysis for
                negative bending
            n: Axial force
            kappa_inc: Initial curvature increment
            kappa_mult: Multiplier to apply to the curvature increment ``kappa_inc``
                when ``delta_m_max`` is satisfied. When ``delta_m_min`` is satisfied,
                the inverse of this multipler is applied to ``kappa_inc``.
            kappa_inc_max: Maximum curvature increment
            delta_m_min: Relative change in moment at which to reduce the curvature
                increment
            delta_m_max: Relative change in moment at which to increase the curvature
                increment
            progress_bar: If set to True, displays the progress bar

        Returns:
            Moment curvature results object
        """
        # determine theta
        theta = 0 if positive else np.pi

        # determine initial curvature that gives zero moment
        def find_intial_curvature(kappa0):
            # initialise moment curvature result
            mk_res = res.MomentCurvatureResults(theta=theta, n_target=n)

            # find neutral axis that gives convergence of axial force
            brentq(
                f=self.service_normal_force_convergence,
                a=-0.1,
                b=0.1,
                args=(kappa0, mk_res),
            )

            # calculate moment convergence
            return mk_res._m_x_i

        # find initial curvature
        kappa0 = root_scalar(f=find_intial_curvature, x0=0, x1=-1e-6)

        return super().moment_curvature_analysis(
            theta=theta,
            n=n,
            kappa0=kappa0.root,
            kappa_inc=kappa_inc,
            kappa_mult=kappa_mult,
            kappa_inc_max=kappa_inc_max,
            delta_m_min=delta_m_min,
            delta_m_max=delta_m_max,
            progress_bar=progress_bar,
        )

    def ultimate_bending_capacity(
        self,
        positive: bool = True,
        n: float = 0,
    ) -> res.UltimateBendingResults:
        """Given axial force ``n``, calculates the ultimate bending capacity.

        Note that ``k_u`` is calculated only for lumped (non-meshed) geometries.

        Args:
            positive: If set to True, calculates the positive bending capacity,
                otherwise calculates the negative bending capacity.
            n: Net axial force

        Returns:
            Ultimate bending results object
        """
        # determine theta
        theta = 0 if positive else np.pi

        return super().ultimate_bending_capacity(theta=theta, n=n)

    def moment_interaction_diagram(self):
        """Generates a moment interaction diagram.

        Raises:
            NotImplementedError: This feature has not yet been implemented.
        """
        raise NotImplementedError

    def biaxial_bending_diagram(self):
        """Generates a biaxial bending diagram.

        Raises:
            NotImplementedError: This feature has not yet been implemented.
        """
        raise NotImplementedError

    def calculate_uncracked_stress(
        self,
        n: float = 0,
        m: float = 0,
    ) -> res.StressResult:
        """Calculates uncracked streses.

        Calculates stresses within the prestressed concrete section assuming an
        uncracked section. Uses gross area section properties to determine concrete,
        reinforcement and strand stresses given an axial force ``n`` and bending moment
        ``m``.

        Args:
            n: Axial force
            m: Bending moment

        Returns:
            Stress results object
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
                sig += -lumped_geom.material.get_prestress_stress()

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
        """Calculates cracked streses.

        Calculates stresses within the prestressed concrete section assuming a
        cracked section. Uses cracked area section properties to determine concrete,
        reinforcement and strand stresses given the actions provided during the cracked
        analysis.

        Args:
            cracked_results: Cracked results objects

        Returns:
            Stress results object
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
                n_strand = (
                    strand.material.get_prestress_stress() * strand.calculate_area()
                )
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
                sig += -lumped_geom.material.get_prestress_stress()

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
            _m_net=m_net,
        )

    def calculate_service_stress(
        self,
        moment_curvature_results: res.MomentCurvatureResults,
        m: float,
        kappa: float | None = None,
    ) -> res.StressResult:
        """Calculates service stresses within the prestressed concrete section.

        Uses linear interpolation of the moment-curvature results to determine the
        curvature of the section given the user supplied moment, and thus the stresses
        within the section. Otherwise, a curvature can be provided which overrides the
        supplied moment.

        Args:
            moment_curvature_results: Moment-curvature results objects
            m: Bending moment
            kappa: Curvature, if provided overrides the supplied bending moment and
                calculates the stress at the given curvature

        Raises:
            AnalysisError: If the analysis fails

        Returns:
            Stress results object
        """
        if kappa is None:
            # get curvature
            kappa = moment_curvature_results.get_curvature(moment=m)

        # initialise variables
        mk = res.MomentCurvatureResults(
            theta=0, n_target=moment_curvature_results.n_target
        )

        # find neutral axis that gives convergence of the axial force
        try:
            eps0, r = brentq(
                f=self.service_normal_force_convergence,
                a=-0.1,
                b=0.1,
                args=(kappa, mk),
                full_output=True,
                disp=False,
            )
        except ValueError as exc:
            msg = "Analysis failed. Confirm that the supplied moment/curvature is "
            msg += "within the range of the moment-curvature analysis."
            raise utils.AnalysisError(msg) from exc

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

        # get global coordinates of extreme compressive fibre
        ecf, _ = utils.calculate_extreme_fibre(
            points=self.compound_geometry.points, theta=0
        )

        # create splits in meshed geometries at points in stress-strain profiles
        meshed_split_geoms: list[CPGeom | CPGeomConcrete] = []

        for meshed_geom in self.meshed_geometries:
            split_geoms = utils.split_geom_at_strains_service(
                geom=meshed_geom,
                theta=0,
                ecf=ecf,
                eps0=eps0,
                kappa=kappa,
            )

            meshed_split_geoms.extend(split_geoms)

        # loop through all meshed geometries and calculate stress
        for meshed_geom in meshed_split_geoms:
            analysis_section = AnalysisSection(geometry=meshed_geom)

            # calculate stress, force and point of action
            sig, n_sec, d_x, d_y = analysis_section.get_service_stress(
                kappa=kappa,
                ecf=ecf,
                eps0=eps0,
                theta=0,
                centroid=self.moment_centroid,
            )

            # save results
            conc_sigs.append(sig)
            conc_forces.append((n_sec, d_x, d_y))
            conc_sections.append(analysis_section)

        # loop through all lumped and strand geometries and calculate stress
        for lumped_geom in self.reinf_geometries_lumped + self.strand_geometries:
            # calculate area and centroid
            area = lumped_geom.calculate_area()
            centroid = lumped_geom.calculate_centroid()

            # get strain at centroid of lump
            strain = utils.get_service_strain(
                point=(centroid[0], centroid[1]),
                ecf=ecf,
                eps0=eps0,
                theta=0,
                kappa=kappa,
            )

            # add initial prestress strain
            if isinstance(lumped_geom.material, SteelStrand):
                eps_pe = -lumped_geom.material.get_prestress_strain()
                strain += eps_pe

            # calculate stress, force and point of action
            sig = lumped_geom.material.stress_strain_profile.get_stress(strain=strain)
            n_lumped = sig * area

            if isinstance(lumped_geom.material, SteelStrand):
                strand_sigs.append(sig)
                strand_strains.append(strain)
                strand_forces.append(
                    (
                        n_lumped,
                        centroid[0] - self.moment_centroid[0],
                        centroid[1] - self.moment_centroid[1],
                    )
                )
                strand_geoms.append(lumped_geom)
            else:
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

    def calculate_ultimate_stress(
        self,
        ultimate_results: res.UltimateBendingResults,
    ) -> res.StressResult:
        """Calculates ultimate stresses within the prestressed concrete section.

        Args:
            ultimate_results: Ultimate bending results objects

        Returns:
            Stress results object
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

        # create splits in meshed geometries at points in stress-strain profiles
        meshed_split_geoms: list[CPGeom | CPGeomConcrete] = []

        if isinf(ultimate_results.d_n):
            meshed_split_geoms = self.meshed_geometries
        else:
            for meshed_geom in self.meshed_geometries:
                split_geoms = utils.split_geom_at_strains_ultimate(
                    geom=meshed_geom,
                    theta=ultimate_results.theta,
                    point_na=point_na,
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

        # loop through all lumped and strand geometries and calculate stress
        for lumped_geom in self.reinf_geometries_lumped + self.strand_geometries:
            # calculate area and centroid
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

            # add initial prestress strain
            if isinstance(lumped_geom.material, SteelStrand):
                eps_pe = -lumped_geom.material.get_prestress_strain()
                strain += eps_pe

            # calculate stress, force and point of action
            sig = lumped_geom.material.stress_strain_profile.get_stress(strain=strain)
            n_lumped = sig * lumped_geom.calculate_area()

            if isinstance(lumped_geom.material, SteelStrand):
                strand_sigs.append(sig)
                strand_strains.append(strain)
                strand_forces.append(
                    (
                        n_lumped,
                        centroid[0] - self.moment_centroid[0],
                        centroid[1] - self.moment_centroid[1],
                    )
                )
                strand_geoms.append(lumped_geom)
            else:
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
