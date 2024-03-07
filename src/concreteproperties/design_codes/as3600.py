"""AS3600 class for designing to the Australian Standard AS 3600:2018."""

from __future__ import annotations

from copy import deepcopy
from math import inf
from typing import TYPE_CHECKING

import numpy as np
from rich.live import Live
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import concreteproperties.results as res
import concreteproperties.stress_strain_profile as ssp
from concreteproperties.design_codes.design_code import DesignCode
from concreteproperties.material import Concrete, SteelBar
from concreteproperties.utils import AnalysisError, create_known_progress


if TYPE_CHECKING:
    from concreteproperties.concrete_section import ConcreteSection


class AS3600(DesignCode):
    """Design code class for Australian standard AS 3600:2018.

    .. note::

        Note that this design code only supports
        :class:`~concreteproperties.material.Concrete` and
        :class:`~concreteproperties.material.SteelBar` material objects. Meshed
        :class:`~concreteproperties.material.Steel` material objects are **not**
        supported, as this falls under the composite structures design code.
    """

    def __init__(self) -> None:
        """Inits the AS3600 class."""
        super().__init__()

    def assign_concrete_section(
        self,
        concrete_section: ConcreteSection,
    ) -> None:
        """Assigns a concrete section to the design code.

        Args:
            concrete_section: Concrete section object to analyse

        Raises:
            ValueError: If there is meshed reinforcement within the concrete_section
        """
        self.concrete_section = concrete_section

        # check to make sure there are no meshed reinforcement regions
        if self.concrete_section.reinf_geometries_meshed:
            raise ValueError(
                "Meshed reinforcement is not supported in this design code."
            )

        # determine reinforcement class
        self.reinforcement_class = "N"

        for steel_geom in self.concrete_section.reinf_geometries_lumped:
            if (
                abs(
                    steel_geom.material.stress_strain_profile.get_ultimate_tensile_strain()
                )
                < 0.05
            ):
                self.reinforcement_class = "L"

        # calculate squash and tensile load
        squash, tensile = self.squash_tensile_load()
        self.squash_load = squash
        self.tensile_load = tensile

    def create_concrete_material(
        self,
        compressive_strength: float,
        colour: str = "lightgrey",
    ) -> Concrete:
        r"""Returns a concrete material object to AS 3600.

        .. admonition:: Material assumptions

          - *Density*: 2400 kg/m\ :sup:`3`

          - *Elastic modulus*: Interpolated from Table 3.1.2

          - *Service stress-strain profile*: Linear with no tension, compressive
            strength at :math:`0.9f'_c`

          - *Ultimate stress-strain profile*: Rectangular stress block, parameters from
            Cl. 8.1.3

          - *Alpha squash*: From Cl. 10.6.2.2

          - *Flexural tensile strength*: From Cl. 3.1.1.3

        Args:
            compressive_strength: Characteristic compressive strength of concrete at 28
                days in megapascals (MPa)
            colour: Colour of the concrete for rendering

        Raises:
            ValueError: If ``compressive_strength`` is not between 20 MPa and 100 MPa.

        Returns:
            Concrete material object
        """
        if compressive_strength < 20 or compressive_strength > 100:
            raise ValueError("compressive_strength must be between 20 MPa and 100 MPa.")

        # create concrete name
        name = f"{compressive_strength:.0f} MPa Concrete (AS 3600:2018)"

        # calculate elastic modulus
        fc_list = [20, 25, 32, 40, 50, 65, 80, 100]
        ec_list = [24000, 26700, 30100, 32800, 34800, 37400, 39600, 42200]
        f_ec = interp1d(fc_list, ec_list)
        elastic_modulus = f_ec(compressive_strength)

        # calculate stress block parameters
        alpha = 0.85 - 0.0015 * compressive_strength
        alpha = max(alpha, 0.67)
        gamma = 0.97 - 0.0025 * compressive_strength
        gamma = max(gamma, 0.67)

        # calculate flexural_tensile_strength
        flexural_tensile_strength = 0.6 * np.sqrt(compressive_strength)

        return Concrete(
            name=name,
            density=2.4e-6,
            stress_strain_profile=ssp.ConcreteLinearNoTension(
                elastic_modulus=elastic_modulus,
                ultimate_strain=0.003,
                compressive_strength=0.9 * compressive_strength,
            ),
            ultimate_stress_strain_profile=ssp.RectangularStressBlock(
                compressive_strength=compressive_strength,
                alpha=alpha,
                gamma=gamma,
                ultimate_strain=0.003,
            ),
            flexural_tensile_strength=flexural_tensile_strength,
            colour=colour,
        )

    def create_steel_material(
        self,
        yield_strength: float = 500,
        ductility_class: str = "N",
        colour: str = "grey",
    ) -> SteelBar:
        r"""Returns a steel bar material object.

        .. admonition:: Material assumptions

          - *Density*: 7850 kg/m\ :sup:`3`

          - *Elastic modulus*: 200000 MPa

          - *Stress-strain profile*: Elastic-plastic, fracture strain from Table 3.2.1


        Args:
            yield_strength: Steel yield strength
            ductility_class: Steel ductility class ("N" or "L")
            colour: Colour of the steel for rendering

        Raises:
            ValueError: If ``ductility_class`` is not "N" or "L"

        Returns:
            Steel material object
        """
        if ductility_class == "N":
            fracture_strain = 0.05
        elif ductility_class == "L":
            fracture_strain = 0.015
        else:
            raise ValueError("ductility_class must be N or L.")

        return SteelBar(
            name=f"{yield_strength:.0f} MPa Steel (AS 3600:2018)",
            density=7.85e-6,
            stress_strain_profile=ssp.SteelElasticPlastic(
                yield_strength=yield_strength,
                elastic_modulus=200e3,
                fracture_strain=fracture_strain,
            ),
            colour=colour,
        )

    def squash_tensile_load(self) -> tuple[float, float]:
        """Calculates the squash and tensile load of the reinforced concrete section.

        Returns:
            Squash and tensile load
        """
        # initialise the squash load, tensile load and squash moment variables
        squash_load = 0
        tensile_load = 0

        # loop through all concrete geometries
        for conc_geom in self.concrete_section.concrete_geometries:
            # calculate area
            area = conc_geom.calculate_area()

            # calculate alpha_squash
            comp_strength = (
                conc_geom.material.stress_strain_profile.get_compressive_strength()
            )

            if comp_strength:
                alpha_squash = 1 - 0.003 * comp_strength
                alpha_squash = min(alpha_squash, 0.85)
                alpha_squash = max(alpha_squash, 0.72)
            else:
                alpha_squash = 1

            # calculate compressive force
            ult_profile = conc_geom.material.ultimate_stress_strain_profile
            force_c = area * alpha_squash * ult_profile.get_compressive_strength()

            # add to totals
            squash_load += force_c

        # loop through all steel geometries
        for steel_geom in self.concrete_section.reinf_geometries_lumped:
            # calculate area
            area = steel_geom.calculate_area()

            # calculate compressive and tensile force
            force_c = area * steel_geom.material.stress_strain_profile.get_stress(
                strain=0.025
            )

            force_t = (
                -area * steel_geom.material.stress_strain_profile.get_yield_strength()
            )

            # add to totals
            squash_load += force_c
            tensile_load += force_t

        return squash_load, tensile_load

    def capacity_reduction_factor(
        self,
        n_u: float,
        n_ub: float,
        n_uot: float,
        k_uo: float,
        phi_0: float,
    ) -> float:
        """Returns the AS 3600 capacity reduction factor (Table 2.2.2).

        ``n_ub`` and ``phi_0`` only required for compression, ``n_uot`` only required
        for tension.

        Args:
            n_u: Axial force in member
            n_ub: Axial force at balanced point
            n_uot: Axial force at ultimate tension load
            k_uo: Neutral axis parameter at pure bending
            phi_0: Capacity reduction factor for dominant compression

        Returns:
            Capacity reduction factor
        """
        # pure bending phi
        if self.reinforcement_class == "N":
            phi = 1.24 - 13 * k_uo / 12
            phi = min(phi, 0.85)
            phi = max(phi, 0.65)
        else:
            phi = 0.65

        # compression
        if n_u > 0:
            if n_u >= n_ub:
                return phi_0
            else:
                return phi_0 + (phi - phi_0) * (1 - n_u / n_ub)
        # tension
        else:
            if self.reinforcement_class == "N":
                return phi + (0.85 - phi) * (n_u / n_uot)
            else:
                return 0.65

    def get_k_uo(
        self,
        theta: float,
    ) -> float:
        r"""Returns k_uo for the reinforced concrete cross-section given ``theta``.

        Args:
            theta: Angle (in radians) the neutral axis makes with the horizontal axis
                (:math:`-\pi \leq \theta \leq \pi`)

        Returns:
            Bending parameter ``k_uo``
        """
        pure_res = self.concrete_section.ultimate_bending_capacity(theta=theta)

        return pure_res.k_u

    def get_n_ub(
        self,
        theta: float,
    ) -> float:
        r"""Returns n_ub for the reinforced concrete cross-section given ``theta``.

        Args:
            theta: Angle (in radians) the neutral axis makes with the horizontal axis
                (:math:`-\pi \leq \theta \leq \pi`)

        Returns:
            Balanced axial force ``n_ub``
        """
        # get depth to extreme tensile bar and its yield strain
        d_0, eps_sy = self.concrete_section.extreme_bar(theta=theta)

        # get compressive strain at extreme fibre
        eps_cu = self.concrete_section.gross_properties.conc_ultimate_strain

        # calculate d_n at balanced load
        d_nb = d_0 * (eps_cu) / (eps_sy + eps_cu)

        # calculate axial force at balanced load
        balanced_res = self.concrete_section.calculate_ultimate_section_actions(
            d_n=d_nb, ultimate_results=res.UltimateBendingResults(theta=theta)
        )

        return balanced_res.n

    def ultimate_bending_capacity(
        self,
        theta: float = 0,
        n_design: float = 0,
        phi_0: float = 0.6,
    ) -> tuple[res.UltimateBendingResults, res.UltimateBendingResults, float]:
        r"""Calculates the ultimate bending capacity with capacity factors to AS 3600.

        Args:
            theta: Angle (in radians) the neutral axis makes with the horizontal axis
                (:math:`-\pi \leq \theta \leq \pi`)
            n_design: Design axial force, N*
            phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)

        Raises:
            AnalysisError: If the design load is greater than the squash load
            AnalysisError: If the design load is greater than the tensile load

        Returns:
            Factored and unfactored ultimate bending results objects, and capacity
            reduction factor (``factored_results``, ``unfactored_results``, ``phi``)
        """
        # get parameters to determine phi
        n_uot = self.tensile_load
        k_uo = self.get_k_uo(theta=theta)
        n_ub = self.get_n_ub(theta=theta)

        # non-linear calculation of phi
        def non_linear_phi(phi_guess):
            phi = self.capacity_reduction_factor(
                n_u=n_design / phi_guess,
                n_ub=n_ub,
                n_uot=n_uot,
                k_uo=k_uo,
                phi_0=phi_0,
            )

            return phi - phi_guess

        phi, _ = brentq(
            f=non_linear_phi,
            a=phi_0,
            b=0.85,
            xtol=1e-3,
            rtol=1e-6,
            full_output=True,
            disp=False,
        )

        # generate basic moment interaction diagram
        f_mi_res, _, _ = self.moment_interaction_diagram(
            theta=theta,
            control_points=[
                ("N", 0.0),
            ],
            n_points=2,
            phi_0=phi_0,
            progress_bar=False,
        )

        # get significant axial loads
        n_squash = f_mi_res.results[0].n
        n_decomp = f_mi_res.results[1].n
        n_tensile = f_mi_res.results[-1].n

        # DETERMINE where we are on interaction diagram
        # if we are above the squash load or tensile load
        if n_design > n_squash:
            raise AnalysisError(
                f"N = {n_design} is greater than the squash load, phiNc = {n_squash}."
            )
        elif n_design < n_tensile:
            raise AnalysisError(
                f"N = {n_design} is greater than the tensile load, phiNt = {n_tensile}"
            )
        # compression linear interpolation
        elif n_design > n_decomp:
            factor = (n_design - n_decomp) / (n_squash - n_decomp)
            squash = f_mi_res.results[0]
            decomp = f_mi_res.results[1]
            ult_res = res.UltimateBendingResults(
                theta=theta,
                d_n=inf,
                k_u=0,
                n=n_design / phi,
                m_x=(decomp.m_x + factor * (squash.m_x - decomp.m_x)) / phi,
                m_y=(decomp.m_y + factor * (squash.m_y - decomp.m_y)) / phi,
                m_xy=(decomp.m_xy + factor * (squash.m_xy - decomp.m_xy)) / phi,
            )
        # regular calculation
        elif n_design >= 0:
            ult_res = self.concrete_section.ultimate_bending_capacity(
                theta=theta, n=n_design / phi
            )
        # tensile linear interpolation
        else:
            factor = n_design / n_tensile
            pure = f_mi_res.results[-2]
            ult_res = res.UltimateBendingResults(
                theta=theta,
                d_n=inf,
                k_u=0,
                n=n_design / phi,
                m_x=(1 - factor) * pure.m_x / phi,
                m_y=(1 - factor) * pure.m_y / phi,
                m_xy=(1 - factor) * pure.m_xy / phi,
            )

        # factor ultimate results
        f_ult_res = deepcopy(ult_res)
        f_ult_res.n *= phi
        f_ult_res.m_x *= phi
        f_ult_res.m_y *= phi
        f_ult_res.m_xy *= phi

        return f_ult_res, ult_res, phi

    def moment_interaction_diagram(
        self,
        theta: float = 0,
        limits: list[tuple[str, float]] | None = None,
        control_points: list[tuple[str, float]] | None = None,
        labels: list[str] | None = None,
        n_points: int = 24,
        n_spacing: int | None = None,
        phi_0: float = 0.6,
        progress_bar: bool = True,
    ) -> tuple[res.MomentInteractionResults, res.MomentInteractionResults, list[float]]:
        r"""Generates a moment interaction diagram with capacity factors to AS 3600.

        See
        :meth:`concreteproperties.concrete_section.ConcreteSection.moment_interaction_diagram`
        for allowable control points.

        .. note::

            When providing ``"N"`` to ``limits`` or ``control_points``, ``"N"`` is taken
            to be the unfactored net (nominal) axial load :math:`N^{*} / \phi`.

        Args:
            theta: Angle (in radians) the neutral axis makes with the horizontal axis
                (:math:`-\pi \leq \theta \leq \pi`)
            limits: List of control points that define the start and end of the
                interaction diagram. List length must equal two. The default limits
                range from concrete decompression strain to the pure bending point,
                ``[("D", 1.0), ("N", 0.0)]``.
            control_points: List of additional control points to add to the moment
                interaction diagram. The default control points include the balanced
                point, ``fy = 1``, i.e. ``[("fy", 1.0)]``. Control points may lie
                outside the limits of the moment interaction diagram as long as
                equilibrium can be found.
            labels: List of labels to apply to the ``limits`` and ``control_points`` for
                plotting purposes. The first two values in ``labels`` apply labels to
                the ``limits``, the remaining values apply labels to the
                ``control_points``. If a single value is provided, this value will be
                applied to both ``limits`` and all ``control_points``. The length of
                ``labels`` must equal ``1`` or ``2 + len(control_points)``.
            n_points: Number of points to compute including and between the ``limits``
                of the moment interaction diagram. Generates equally spaced neutral axes
                between the ``limits``.
            n_spacing: If provided, overrides ``n_points`` and generates the moment
                interaction diagram using ``n_spacing`` equally spaced axial loads. Note
                that using ``n_spacing`` negatively affects performance, as the neutral
                axis depth must first be located for each point on the moment
                interaction diagram.
            phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)
            progress_bar: If set to True, displays the progress bar

        Returns:
            Factored and unfactored moment interaction results objects, and list of
            capacity reduction factors (``factored_results``, ``unfactored_results``,
            ``phis``)
        """
        if limits is None:
            limits = [("D", 1.0), ("N", 0.0)]

        if control_points is None:
            control_points = [("fy", 1.0)]

        mi_res = self.concrete_section.moment_interaction_diagram(
            theta=theta,
            limits=limits,
            control_points=control_points,
            labels=labels,
            n_points=n_points,
            n_spacing=n_spacing,
            progress_bar=progress_bar,
        )

        # get theta
        theta = mi_res.results[0].theta

        # add squash load
        mi_res.results.insert(
            0,
            res.UltimateBendingResults(
                theta=theta,
                d_n=inf,
                k_u=0,
                n=self.squash_load,
                m_x=0,
                m_y=0,
                m_xy=0,
            ),
        )

        # add tensile load
        mi_res.results.append(
            res.UltimateBendingResults(
                theta=theta,
                d_n=0,
                k_u=0,
                n=self.tensile_load,
                m_x=0,
                m_y=0,
                m_xy=0,
            )
        )

        # make a copy of the results to factor
        f_mi_res = deepcopy(mi_res)

        # list to store phis
        phis = []

        # get required constants for phi
        n_uot = self.tensile_load
        k_uo = self.get_k_uo(theta=theta)
        n_ub = self.get_n_ub(theta=theta)

        # factor results
        for ult_res in f_mi_res.results:
            phi = self.capacity_reduction_factor(
                n_u=ult_res.n, n_ub=n_ub, n_uot=n_uot, k_uo=k_uo, phi_0=phi_0
            )
            ult_res.n *= phi
            ult_res.m_x *= phi
            ult_res.m_y *= phi
            ult_res.m_xy *= phi
            phis.append(phi)

        return f_mi_res, mi_res, phis

    def biaxial_bending_diagram(
        self,
        n_design: float = 0,
        n_points: int = 48,
        phi_0: float = 0.6,
        progress_bar: bool = True,
    ) -> tuple[res.BiaxialBendingResults, list[float]]:
        """Generates a biaxial bending with capacity factors to AS 3600.

        Args:
            n_design: Design axial force, N*
            n_points: Number of calculation points
            phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)
            progress_bar: If set to True, displays the progress bar

        Returns:
            Factored biaxial bending results object and list of capacity reduction
            factors (``factored_results``, ``phis``)
        """
        # initialise results
        f_bb_res = res.BiaxialBendingResults(n=n_design)
        phis = []

        # calculate d_theta
        d_theta = 2 * np.pi / n_points

        # generate list of thetas
        theta_list = np.linspace(start=-np.pi, stop=np.pi - d_theta, num=n_points)

        # function that performs biaxial bending analysis
        def bbcurve(progress=None):
            # loop through thetas
            for theta in theta_list:
                # factored capacity
                f_ult_res, _, phi = self.ultimate_bending_capacity(
                    theta=theta, n_design=n_design, phi_0=phi_0
                )
                f_bb_res.results.append(f_ult_res)
                phis.append(phi)

                if progress:
                    progress.update(task, advance=1)

        if progress_bar:
            # create progress bar
            progress = create_known_progress()

            with Live(progress, refresh_per_second=10) as live:
                task = progress.add_task(
                    description="[red]Generating biaxial bending diagram",
                    total=n_points,
                )

                bbcurve(progress=progress)

                msg = "[bold green]:white_check_mark: Biaxial bending diagram generated"
                progress.update(
                    task,
                    description=msg,
                )
                live.refresh()
        else:
            bbcurve()

        # add first result to end of list top
        f_bb_res.results.append(f_bb_res.results[0])
        phis.append(phis[0])

        return f_bb_res, phis
