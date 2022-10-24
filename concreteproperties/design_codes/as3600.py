from __future__ import annotations

from copy import deepcopy
from math import inf
from typing import TYPE_CHECKING, List, Tuple, Union

import concreteproperties.results as res
import concreteproperties.stress_strain_profile as ssp
import concreteproperties.utils as utils
import numpy as np
from concreteproperties.design_codes.design_code import DesignCode
from concreteproperties.material import Concrete, SteelBar
from rich.live import Live
from scipy.interpolate import interp1d
from scipy.optimize import brentq

if TYPE_CHECKING:
    from concreteproperties.concrete_section import ConcreteSection


class AS3600(DesignCode):
    """Design code class for Australian standard AS 3600:2018.

    Note that this design code only supports :class:`~concreteproperties.pre.Concrete`
    and :class:`~concreteproperties.pre.SteelBar` material objects. Meshed
    :class:`~concreteproperties.pre.Steel` material objects are **not** supported
    as this falls under the composite structures design code.
    """

    def __init__(
        self,
    ):
        """Inits the AS3600 class."""

        super().__init__()

    def assign_concrete_section(
        self,
        concrete_section: ConcreteSection,
    ):
        """Assigns a concrete section to the design code.

        :param concrete_section: Concrete section object to analyse
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
        r"""Returns a concrete material object to AS 3600:2018.

        | **Material assumptions:**
        | - *Density*: 2400 kg/m\ :sup:`3`
        | - *Elastic modulus*: Interpolated from Table 3.1.2
        | - *Service stress-strain profile*: Linear with no tension, compressive strength
          at 0.9 * f'c
        | - *Ultimate stress-strain profile*: Rectangular stress block, parameters from
          Cl. 8.1.3
        | - *Alpha squash*: From Cl. 10.6.2.2
        | - *Flexural tensile strength*: From Cl. 3.1.1.3

        :param compressive_strength: Characteristic compressive strength of
            concrete at 28 days in megapascals (MPa)
        :param colour: Colour of the concrete for rendering

        :raises ValueError: If ``compressive_strength`` is not between 20 MPa and
            100 MPa.

        :return: Concrete material object
        """

        if compressive_strength < 20 or compressive_strength > 100:
            raise ValueError("compressive_strength must be between 20 MPa and 100 MPa.")

        # create concrete name
        name = f"{compressive_strength:.0f} MPa Concrete (AS 3600:2018)"

        # calculate elastic modulus
        fc_list = [20, 25, 32, 40, 50, 65, 80, 100]
        Ec_list = [24000, 26700, 30100, 32800, 34800, 37400, 39600, 42200]
        f_Ec = interp1d(fc_list, Ec_list)
        elastic_modulus = f_Ec(compressive_strength)

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

        | **Material assumptions:**
        | - *Density*: 7850 kg/m\ :sup:`3`
        | - *Elastic modulus*: 200,000 MPa
        | - *Stress-strain profile:* Elastic-plastic, fracture strain from Table 3.2.1

        :param yield_strength: Steel yield strength
        :param ductility_class: Steel ductility class ("N" or "L")
        :param colour: Colour of the steel for rendering

        :raises ValueError: If ``ductility_class`` is not "N" or "L"

        :return: Steel material object
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

    def squash_tensile_load(
        self,
    ) -> Tuple[float, float]:
        """Calculates the squash and tensile load of the reinforced concrete section.

        :return: Squash and tensile load
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
            force_c = (
                area
                * alpha_squash
                * conc_geom.material.ultimate_stress_strain_profile.get_compressive_strength()
            )

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
        """Returns the AS 3600:2018 capacity reduction factor (Table 2.2.2).

        ``n_ub`` and ``phi_0`` only required for compression, ``n_uot`` only required
        for tension.

        :param n_u: Axial force in member
        :param n_ub: Axial force at balanced point
        :param n_uot: Axial force at ultimate tension load
        :param k_uo: Neutral axis parameter at pure bending
        :param phi_0: Capacity reduction factor for dominant compression

        :return: Capacity reduction factor
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

        :param theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)

        :return: Bending parameter k_uo
        """

        pure_res = self.concrete_section.ultimate_bending_capacity(theta=theta)

        return pure_res.k_u

    def get_n_ub(
        self,
        theta: float,
    ) -> float:
        r"""Returns n_ub for the reinforced concrete cross-section given ``theta``.

        :param theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)

        :return: Balanced axial force n_ub
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
        n: float = 0,
        phi_0: float = 0.6,
    ) -> Tuple[res.UltimateBendingResults, res.UltimateBendingResults, float]:
        r"""Calculates the ultimate bending capacity with capacity factors to
        AS 3600:2018.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :param n: Net axial force
        :param phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)

        :return: Factored and unfactored ultimate bending results objects, and capacity
            reduction factor *(factored_results, unfactored_results, phi)*
        """

        # get parameters to determine phi
        n_uot = self.tensile_load
        k_uo = self.get_k_uo(theta=theta)
        n_ub = self.get_n_ub(theta=theta)

        # non-linear calculation of phi
        def non_linear_phi(phi_guess):
            phi = self.capacity_reduction_factor(
                n_u=n / phi_guess,
                n_ub=n_ub,
                n_uot=n_uot,
                k_uo=k_uo,
                phi_0=phi_0,
            )

            return phi - phi_guess

        (phi, r) = brentq(
            f=non_linear_phi,
            a=phi_0,
            b=0.85,
            xtol=1e-3,
            rtol=1e-6,  # type: ignore
            full_output=True,
            disp=False,
        )

        # calculate ultimate bending capacity
        ult_res = self.concrete_section.ultimate_bending_capacity(
            theta=theta, n=n / phi
        )

        # factor ultimate results
        factored_ult_res = deepcopy(ult_res)
        factored_ult_res.n *= phi
        factored_ult_res.m_x *= phi
        factored_ult_res.m_y *= phi
        factored_ult_res.m_xy *= phi

        return factored_ult_res, ult_res, phi

    def moment_interaction_diagram(
        self,
        theta: float = 0,
        control_points: List[Tuple[str, float]] = [
                ("D", 1.0),
                ("fy", 1.0),
                ("N", 0.0),
            ],
        labels: List[Union[str, None]] = [None],
        n_points: Union[int, List[int]] = [12, 12],
        phi_0: float = 0.6,
    ) -> Tuple[res.MomentInteractionResults, res.MomentInteractionResults, List[float]]:
        r"""Generates a moment interaction diagram with capacity factors to AS 3600:2018.

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
        :param phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)

        :return: Factored and unfactored moment interaction results objects, and list of
            capacity reduction factors *(factored_results, unfactored_results, phis)*
        """

        mi_res = self.concrete_section.moment_interaction_diagram(
            theta=theta,
            control_points=control_points,
            labels=labels,
            n_points=n_points,
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
        factored_mi_res = deepcopy(mi_res)

        # list to store phis
        phis = []

        # get required constants for phi
        n_uot = self.tensile_load
        k_uo = self.get_k_uo(theta=theta)
        n_ub = self.get_n_ub(theta=theta)

        # factor results
        for ult_res in factored_mi_res.results:
            phi = self.capacity_reduction_factor(
                n_u=ult_res.n, n_ub=n_ub, n_uot=n_uot, k_uo=k_uo, phi_0=phi_0
            )
            ult_res.n *= phi
            ult_res.m_x *= phi
            ult_res.m_y *= phi
            ult_res.m_xy *= phi
            phis.append(phi)

        return factored_mi_res, mi_res, phis

    def biaxial_bending_diagram(
        self,
        n: float = 0,
        n_points: int = 48,
        phi_0: float = 0.6,
    ) -> Tuple[res.BiaxialBendingResults, List[float]]:
        """Generates a biaxial bending with capacity factors to AS 3600:2018.

        :param n: Net axial force
        :param n_points: Number of calculation points between the decompression
        :param phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)

        :return: Factored biaxial bending results object and list of capacity reduction
            factors *(factored_results, phis)*
        """

        pass

        # initialise results
        f_bb_res = res.BiaxialBendingResults(n=n)
        phis = []

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
                # factored capacity
                f_ult_res, _, phi = self.ultimate_bending_capacity(
                    theta=theta, n=n, phi_0=phi_0
                )
                f_bb_res.results.append(f_ult_res)
                phis.append(phi)

                progress.update(task, advance=1)

            # add first result to end of list top
            f_bb_res.results.append(f_bb_res.results[0])
            phis.append(phis[0])

            progress.update(
                task,
                description="[bold green]:white_check_mark: Biaxial bending diagram generated",
            )
            live.refresh()

        return f_bb_res, phis
