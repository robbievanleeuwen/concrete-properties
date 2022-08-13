from __future__ import annotations

from copy import deepcopy
from math import inf
from multiprocessing.sharedctypes import Value
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from rich.live import Live
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import concreteproperties.results as res
import concreteproperties.stress_strain_profile as ssp
import concreteproperties.utils as utils
from concreteproperties.material import Concrete, SteelBar

if TYPE_CHECKING:
    from concreteproperties.concrete_section import ConcreteSection


class DesignCode:
    """Abstract class for a design code object."""

    def __init__(
        self,
    ):
        """Inits the DesignCode class."""

        pass

    def assign_concrete_section(
        self,
        concrete_section: ConcreteSection,
    ):
        """Assigns a concrete section to the design code.

        :param concrete_section: Concrete section object to analyse
        """

        self.concrete_section = concrete_section

    def create_concrete_material(
        self,
        compressive_strength: float,
        colour: str = "lightgrey",
    ) -> Concrete:
        """Returns a concrete material object.

        List assumptions of material properties here...

        :param compressive_strength: Concrete compressive strength
        :param colour: Colour of the concrete for rendering

        :return: Concrete material object
        """

        raise NotImplementedError

    def create_steel_material(
        self,
        yield_strength: float,
        colour: str = "grey",
    ) -> SteelBar:
        """Returns a steel bar material object.

        List assumptions of material properties here...

        :param yield_strength: Steel yield strength
        :param colour: Colour of the steel for rendering

        :return: Steel material object
        """

        raise NotImplementedError

    def get_gross_properties(
        self,
        **kwargs,
    ) -> res.GrossProperties:
        """Returns the gross section properties of the reinforced concrete section.

        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.get_gross_properties`

        :return: Concrete properties object
        """

        return self.concrete_section.get_gross_properties(**kwargs)

    def get_transformed_gross_properties(
        self,
        **kwargs,
    ) -> res.TransformedGrossProperties:
        """Transforms gross section properties.

        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.get_transformed_gross_properties`

        :return: Transformed concrete properties object
        """

        return self.concrete_section.get_transformed_gross_properties(**kwargs)

    def calculate_cracked_properties(
        self,
        **kwargs,
    ) -> res.CrackedResults:
        """Calculates cracked section properties.

        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_cracked_properties`

        :return: Cracked results object
        """

        return self.concrete_section.calculate_cracked_properties(**kwargs)

    def moment_curvature_analysis(
        self,
        **kwargs,
    ) -> res.MomentCurvatureResults:
        """Performs a moment curvature analysis. No reduction factors are applied to the
        moments.

        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.moment_curvature_analysis`

        :return: Moment curvature results object
        """

        return self.concrete_section.moment_curvature_analysis(**kwargs)

    def ultimate_bending_capacity(
        self,
        **kwargs,
    ) -> res.UltimateBendingResults:
        """Calculates the ultimate bending capacity.

        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.ultimate_bending_capacity`

        :return: Ultimate bending results object
        """

        return self.concrete_section.ultimate_bending_capacity(**kwargs)

    def moment_interaction_diagram(
        self,
        **kwargs,
    ) -> res.MomentInteractionResults:
        """Generates a moment interaction diagram.

        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.moment_interaction_diagram`

        :return: Moment interaction results object
        """

        return self.concrete_section.moment_interaction_diagram(**kwargs)

    def biaxial_bending_diagram(
        self,
        **kwargs,
    ) -> res.BiaxialBendingResults:
        """Generates a biaxial bending diagram.

        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.biaxial_bending_diagram`

        :return: Biaxial bending results
        """

        return self.concrete_section.biaxial_bending_diagram(**kwargs)

    def calculate_uncracked_stress(
        self,
        **kwargs,
    ) -> res.StressResult:
        """Calculates stresses within the reinforced concrete section assuming an
        uncracked section.

        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_uncracked_stress`

        :return: Stress results object
        """

        return self.concrete_section.calculate_uncracked_stress(**kwargs)

    def calculate_cracked_stress(
        self,
        **kwargs,
    ) -> res.StressResult:
        """Calculates stresses within the reinforced concrete section assuming a cracked
        section.

        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_cracked_stress`

        :return: Stress results object
        """

        return self.concrete_section.calculate_cracked_stress(**kwargs)

    def calculate_service_stress(
        self,
        **kwargs,
    ) -> res.StressResult:
        """Calculates service stresses within the reinforced concrete section.

        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_service_stress`

        :return: Stress results object
        """

        return self.concrete_section.calculate_service_stress(**kwargs)

    def calculate_ultimate_stress(
        self,
        **kwargs,
    ) -> res.StressResult:
        """Calculates ultimate stresses within the reinforced concrete section.

        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_ultimate_stress`

        :return: Stress results object
        """

        return self.concrete_section.calculate_ultimate_stress(**kwargs)


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

        # max compression strain for squash load = 0.0025!

        # calculate alpha_squash
        alpha_squash = 1 - 0.003 * compressive_strength
        alpha_squash = min(alpha_squash, 0.85)
        alpha_squash = max(alpha_squash, 0.72)

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
            alpha_squash=alpha_squash,
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
            # calculate area and centroid
            area = conc_geom.calculate_area()

            # calculate compressive force
            force_c = (
                area
                * conc_geom.material.alpha_squash
                * conc_geom.material.ultimate_stress_strain_profile.get_compressive_strength()
            )

            # add to totals
            squash_load += force_c

        # loop through all steel geometries
        for steel_geom in self.concrete_section.reinf_geometries_lumped:
            # calculate area and centroid
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
        phi_0: float = 0.6,
    ) -> Tuple[res.MomentInteractionResults, res.MomentInteractionResults, List[float]]:
        """Generates a moment interaction diagram with capacity factors to AS 3600:2018.

        :param phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)

        :return: Factored and unfactored moment interaction results objects, and list of
            capacity reduction factors *(factored_results, unfactored_results, phis)*
        """

        mi_res = self.concrete_section.moment_interaction_diagram(
            control_points=[
                ("D", 1.0),
                ("fy", 1.0),
                ("N", 0.0),
            ],
            n_points=[12, 12],
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


class NZS3101(DesignCode):
    # TODO - add options for returning overstrength capacity/interaction curves
    # TODO - For design checks have three scenarios
    #           1) Normal design phi and material strengths
    #           2) Normal design material strengths, overstrength phi = 1.0
    #           3) Overstrength material strengths, overstrength phi = 1.0
    # TODO - add density as input as E_conc reliant on it
    """Design code class for New Zealand standard NZS3101:2006.

    Note that this design code only supports :class:`~concreteproperties.pre.Concrete`
    and :class:`~concreteproperties.pre.SteelBar` material objects. Meshed
    :class:`~concreteproperties.pre.Steel` material objects are **not** supported
    as this falls under the composite structures design code.
    """

    def __init__(
        self,
    ):
        """Inits the NZS3101 class."""
        self.analysis_code = 'NZS3101:2006'

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
        # TODO - update to NZ code, reinforcement class not required for NZ design as all bars required to be class E

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

    def E_conc(self, compressive_strength, density=None):
        '''calculate Youngs Modulus (:math: `E_c`) for concrete in accordance with NZS3101:2006 CL 5.2.3(b)
        If :math: `f_c` is only provided, then the default density is assumed

            :math: `E_c=\\displaystyle{4700\\sqrt{f\\textquotesingle_c} \\frac{\\rho}{2300}}`

        :param compressive_strength: 28 day compressive concrete strength (MPa)
        :type compressive_strength: float
        :param density: concrete density \\rho in accordance with NZS3101:2006 CL 5.2.2, defaults to None. If None is specified then default for normal weight concrete is adopted in accordance with specified analysis code derivation
        :type density: float, optional
        :return: :math: `E_c`, Youngs Modulus (MPa)
        :rtype: float
        '''

        # low and high limit on density in NZS3101:2006 CL 5.2.2 for E_c equation to be valid
        low_limit = 1800
        high_limit = 2800

        if density is None:
            density = 2300

        self.check_limits(density, low_limit, high_limit)

        E_c = (4700 * (compressive_strength**0.5)) * (density / 2300) ** 1.5

        return E_c

    def check_limits(self, test_value, low_limit, high_limit):
        if test_value < low_limit or test_value > high_limit:
            raise Exception(
                f'The specified concrete density of {test_value}kg/m^3 is not within the bounds of {low_limit}kg/m^3 & {high_limit}kg/m^3 for the {self.analysis_code} code'
            )

    def alpha_1(self, compressive_strength):
        '''scaling factor relating the nominal 28 day concrete compressive strength to
        the effective concrete compressive strength used for design purposes within the
        concrete stress block. For an equivalent rectangular compressive stress block it
        relates the 28 day concrete compressive strength to the average concrete
        compressive design strength.

            :math:`\\alpha_1=\\displaystyle{\\frac{f_{DESIGN}}{f\\rq_c}}`

        :param compressive_strength: 28 day compressive design strength (MPa)
        :type compressive_strength: float
        :return: :math:`\\alpha_1` factor
        :rtype: float
        '''
        if compressive_strength <= 55:
            alpha_1 = 0.85
        if compressive_strength > 55:
            alpha_1 = min(0.85 - 0.004 * (compressive_strength - 55), 0.75)

        return alpha_1

    def beta_1(self, compressive_strength):
        '''scaling factor relating the depth of an equivalent rectangular compressive
        stress block :math:`a` to the depth of the neutral axis :math:`c`.
        A function of the concrete strength

            :math:`\\beta_1=\\displaystyle{\\frac{a}{c}}`

        :param compressive_strength: 28 day compressive design strength (MPa)
        :type compressive_strength: float
        :return: :math:`\\beta_1` factor
        :rtype: float
        '''

        if compressive_strength <= 30:
            beta_1 = 0.85
        if compressive_strength >= 55:
            beta_1 = 0.65
        if compressive_strength > 30 and compressive_strength < 55:
            beta_1 = 0.85 - 0.008 * (compressive_strength - 30)

        return beta_1

    def lamda(self, density=2300):
        '''modification factor reflecting the reduced mechanical properties of lightweight concrete relative to normal weight concrete of the same compression strength

            :math:`\\lamda=0.4+\\displaystyle{\\frac{0.6\\rho}{2200}} \\leq 1.0`

        :param density: Saturated surface dry density of concrete material, defaults to 2300 kg/m\ :sup:`3`
        :type density: int, optional
        :return: :math:`\\lamda` factor
        :rtype: float
        '''
        return min(0.4 + 0.6 * density / 2200, 1)

    # TODO - add PPHR classification, density (as E_conc dependant on this)
    def create_concrete_material(
        self,
        compressive_strength: float,
        ultimate_strain: [float] = 0.003,
        density: Optional[float] = 2300,
        pphr_class: Optional[str] = 'NDPR',
        colour: Optional[str] = 'lightgrey',
    ) -> Concrete:
        # TODO update assumptions, and code clauses
        """Returns a concrete material object to NZS3101:2006.


        | **Material assumptions:**
        | - *Density*: Defaults to 2300 kg/m\ :sup:`3` unless supplied by user
        | - *Elastic modulus*: Calculated from NZS3101:2006 Eq. 5-1
        | - *Service stress-strain profile*: Linear with no tension
        | - *Ultimate stress-strain profile*: Rectangular stress block, parameters from NZS3101:2006 CL 7.4.2.7
        | - *Alpha squash*: From Cl. 10.6.2.2
        | - *Modulus of rupture*: Calculated from NZS3101:2006 Eq. 5-4

        :param compressive_strength: 28 day compressive design strength (MPa)
        :type compressive_strength: float
        :param ultimate_strain: Maximum concrete compressive strain at crushing of the concrete for design, defaults to 0.003
        :type ultimate_strain: float
        :param density: Saturated surface dry density of concrete material, defaults to 2300 kg/m\ :sup:`3`
        :type density: Optional[float], optional
        :param pphr_class: Potential Plastic Hinge Region (PPHR) classification, NDPR/LDPR/DPR, defaults to 'NDPR'
        :type pphr_class: Optional[str], optional
        :param colour: Colour of the concrete for rendering
        :type colour: Optional[str]

        #TODO need to update this, depends on type of PPHR
        :raises ValueError: If compressive_strength is not between 20 MPa and 100 MPa. # need to update

        :return: Concrete material object
        """

        # Check NZS3101:2006 CL 5.2.1 compressive strength limits (dependant on PPHR class)
        self.check_f_c_limits(compressive_strength, pphr_class)

        # create concrete name
        name = f'{compressive_strength:.0f} MPa Concrete ({self.analysis_code})'

        # calculate elastic modulus
        elastic_modulus = self.E_conc(compressive_strength, density)

        # calculate rectangular stress block parameters
        alpha_1 = self.alpha_1(compressive_strength)
        beta_1 = self.beta_1(compressive_strength)

        # max compression strain for squash load = 0.0025!
        # TODO is this relevant, assuming this is simply the maximum compression strength which is dependant on the PPHR classification
        # calculate alpha_squash
        alpha_squash = 1 - 0.003 * compressive_strength
        alpha_squash = min(alpha_squash, 0.85)
        alpha_squash = max(alpha_squash, 0.72)

        # calculate modulus of rupture in accordance with NZS3101:2006 CL 5.2.5
        lamda = self.lamda(density)
        modulus_of_rupture = 0.6 * lamda * np.sqrt(compressive_strength)

        return Concrete(
            name=name,
            density=density,
            stress_strain_profile=ssp.ConcreteLinearNoTension(
                elastic_modulus=elastic_modulus,
                ultimate_strain=ultimate_strain,
                compressive_strength=compressive_strength,
            ),
            ultimate_stress_strain_profile=ssp.RectangularStressBlock(
                compressive_strength=compressive_strength,
                alpha=alpha_1,
                gamma=beta_1,
                ultimate_strain=ultimate_strain,
            ),
            alpha_squash=alpha_squash,
            flexural_tensile_strength=modulus_of_rupture,
            colour=colour,
        )

    def check_f_c_limits(self, compressive_strength, pphr_class):
        # TODO - complete docstring
        '''_summary_

        :param compressive_strength: _description_
        :type compressive_strength: _type_
        :param pphr_class: _description_
        :type pphr_class: _type_
        :raises Exception: _description_
        :raises ValueError: _description_
        '''
        f_c_lower = 20
        if pphr_class.upper() in ['NDPR']:
            f_c_upper = 100
        elif pphr_class.upper() in ['LDPR', 'DPR']:
            f_c_upper = 70
        else:
            raise Exception(
                f'The specified PPHR class specified ({pphr_class}) should be NDPR, LDPR or DPR for the {self.analysis_code} code'
            )
        if not f_c_lower <= compressive_strength <= f_c_upper:
            raise ValueError(
                f'Concrete compressive strength must be between {f_c_lower} MPa & {f_c_upper} MPa for a {pphr_class} PPHR for the {self.analysis_code} code'
            )

    def create_steel_material(
        self,
        yield_strength: float = 500,
        colour: Optional[str] = "grey",
    ) -> SteelBar:
        # TODO - update docstring for NZS3101 code
        r"""Returns a steel material object.

        | **Material assumptions:**
        | - *Density*: 7850 kg/m\ :sup:`3`
        | - *Elastic modulus*: 200,000 MPa
        | - *Stress-strain profile:* Elastic-plastic, fracture strain from Table 3.2.1

        :param yield_strength: Steel yield strength
        :param colour: Colour of the steel for rendering

        :return: Steel bar material object
        """
        # TODO - not required for NZS3101, all bars class E
        # TODO -
        fracture_strain = 0.10
        # if ductility_class == "N":
        #     fracture_strain = 0.05
        # elif ductility_class == "L":
        #     fracture_strain = 0.015
        # else:
        #     raise ValueError("ductility_class must be N or L.")
        # TODO - note no specific fracture strain defined in NZS3101, inferred by curvature limits potentially?? Determine how this is utilised
        # TODO - fracture strain not given in AS/NZS4671, but strain at max ultimate strength is given, hich coul dbe considered a lower bound
        return SteelBar(
            name=f"{yield_strength:.0f} MPa Steel (NZS3101:2006)",
            density=7.85e-6,
            stress_strain_profile=ssp.SteelElasticPlastic(
                yield_strength=yield_strength,
                elastic_modulus=200e3,
                fracture_strain=fracture_strain,
            ),
            colour=colour,
        )

    # TODO - update for NZS3101, max comprssive loads based on PPHR classification
    # TODO - this should be a function to return max compressive load from CL 10.3.4.2 or CL 10.4.4 depending on PPHR classification, and the max tensile capacity
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
            # calculate area and centroid
            area = conc_geom.calculate_area()

            # calculate compressive force
            force_c = (
                area
                * conc_geom.material.alpha_squash
                * conc_geom.material.ultimate_stress_strain_profile.get_compressive_strength()
            )

            # add to totals
            squash_load += force_c

        # loop through all steel geometries
        for steel_geom in self.concrete_section.reinf_geometries_lumped:
            # calculate area and centroid
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

    # TODO - phi is constant 0.85 in NZS3101:2006, so this is not required, should strength reduction factor be defined as a variable in init, what about overstrength checks where phi = 1.0?
    # def capacity_reduction_factor(
    #     self,
    #     n_u: float,
    #     n_ub: float,
    #     n_uot: float,
    #     k_uo: float,
    #     phi_0: float,
    # ) -> float:
    #     """Returns the AS 3600:2018 capacity reduction factor (Table 2.2.2).

    #     ``n_ub`` and ``phi_0`` only required for compression, ``n_uot`` only required
    #     for tension.

    #     :param n_u: Axial force in member
    #     :param n_ub: Axial force at balanced point
    #     :param n_uot: Axial force at ultimate tension load
    #     :param k_uo: Neutral axis parameter at pure bending
    #     :param phi_0: Capacity reduction factor for dominant compression

    #     :return: Capacity reduction factor
    #     """

    #     # pure bending phi
    #     if self.reinforcement_class == "N":
    #         phi = 1.24 - 13 * k_uo / 12
    #         phi = min(phi, 0.85)
    #         phi = max(phi, 0.65)
    #     else:
    #         phi = 0.65

    #     # compression
    #     if n_u > 0:
    #         if n_u >= n_ub:
    #             return phi_0
    #         else:
    #             return phi_0 + (phi - phi_0) * (1 - n_u / n_ub)
    #     # tension
    #     else:
    #         if self.reinforcement_class == "N":
    #             return phi + (0.85 - phi) * (n_u / n_uot)
    #         else:
    #             return 0.65

    # def get_k_uo(
    #     self,
    #     theta: float,
    # ) -> float:
    #     r"""Returns k_uo for the reinforced concrete cross-section given ``theta``.

    #     :param theta: Angle (in radians) the neutral axis makes with the
    #         horizontal axis (:math:`-\pi \leq \theta \leq \pi`)

    #     :return: Bending parameter k_uo
    #     """

    #     pure_res = self.concrete_section.ultimate_bending_capacity(theta=theta)

    #     return pure_res.k_u

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

    # TODO - update based on NZS3101 code, phi will differ
    def ultimate_bending_capacity(
        self,
        theta: float = 0,
        n: float = 0,
        phi_0: float = 0.6,
    ) -> Tuple[res.UltimateBendingResults, res.UltimateBendingResults, float]:
        # TODO - update docstring
        r"""Calculates the ultimate bending capacity with capacity factors to
        NZS3101:2006.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :param n: Net axial force
        :param phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)

        :return: Factored and unfactored ultimate bending results objects, and capacity
            reduction factor *(factored_results, unfactored_results, phi)*
        """
        # TODO - not required with constant phi
        # # get parameters to determine phi
        # n_uot = self.tensile_load
        # k_uo = self.get_k_uo(theta=theta)
        # n_ub = self.get_n_ub(theta=theta)

        # # non-linear calculation of phi
        # def non_linear_phi(phi_guess):
        #     phi = self.capacity_reduction_factor(
        #         n_u=n / phi_guess,
        #         n_ub=n_ub,
        #         n_uot=n_uot,
        #         k_uo=k_uo,
        #         phi_0=phi_0,
        #     )

        #     return phi - phi_guess

        # (phi, r) = brentq(
        #     f=non_linear_phi,
        #     a=phi_0,
        #     b=0.85,
        #     xtol=1e-3,
        #     rtol=1e-6,  # type: ignore
        #     full_output=True,
        #     disp=False,
        # )

        # create constant phi list
        phi = [self.phi] * len(theta)

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

    # TODO - update for phi in accordance with NZS3101
    # TODO - update docstring
    def moment_interaction_diagram(
        self,
        phi_0: float = 0.6,
    ) -> Tuple[res.MomentInteractionResults, res.MomentInteractionResults, List[float]]:
        """Generates a moment interaction diagram with capacity factors to NZS3101:2006.

        :param phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)

        :return: Factored and unfactored moment interaction results objects, and list of
            capacity reduction factors *(factored_results, unfactored_results, phis)*
        """

        mi_res = self.concrete_section.moment_interaction_diagram(
            control_points=[
                ("D", 1.0),
                ("fy", 1.0),
                ("N", 0.0),
            ],
            n_points=[12, 12],
        )

        # get theta
        theta = mi_res.results[0].theta

        # TODO - need to determine what k_u is here?
        # TODO - check how this returns the correct Mx and My values so 'load angle' = theta? This doe snto seem correct as theata does not equal load angle typically?
        # TODO - what is m_xy here? is it the resultant moment?
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

        # TODO - need to update for phi
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

    # TODO - need to update for constant phi
    # TODO - update docstring
    def biaxial_bending_diagram(
        self,
        n: float = 0,
        n_points: int = 48,
        phi_0: float = 0.6,
    ) -> Tuple[res.BiaxialBendingResults, List[float]]:
        """Generates a biaxial bending with capacity factors to NZS3101:2006.

        :param n: Net axial force
        :param n_points: Number of calculation points between the decompression
        :param phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)

        :return: Factored biaxial bending results object and list of capacity reduction
            factors *(factored_results, phis)*
        """

        pass

        # TODO - what does the f stand for here bb = biaxial bending.... f = factored?
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
                # TODO check if this returns the Mx and My values
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
