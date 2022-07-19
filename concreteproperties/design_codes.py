from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from copy import deepcopy
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from rich.live import Live

from concreteproperties.material import Concrete, Steel
import concreteproperties.stress_strain_profile as ssp
import concreteproperties.results as res
import concreteproperties.utils as utils

from sectionproperties.analysis.fea import principal_coordinate

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
        :type concrete_section:
            :class:`~concreteproperties.concrete_section.ConcreteSection`
        """

        self.concrete_section = concrete_section

    def create_concrete_material(
        self,
        compressive_strength: float,
        colour: Optional[str] = "lightgrey",
    ) -> Concrete:
        """Returns a concrete material object.

        List assumptions of material properties here...

        :param float compressive_strength: Concrete compressive strength
        :param colour: Colour of the concrete for rendering
        :type colour: Optional[str]

        :return: Concrete material object
        :rtype: :class:`~concreteproperties.material.Concrete`
        """

        raise NotImplementedError

    def create_steel_material(
        self,
        yield_strength: float,
        colour: Optional[str] = "grey",
    ) -> Steel:
        """Returns a steel material object.

        List assumptions of material properties here...

        :param float yield_strength: Steel yield strength
        :param colour: Colour of the steel for rendering
        :type colour: Optional[str]

        :return: Steel material object
        :rtype: :class:`~concreteproperties.material.Steel`
        """

        raise NotImplementedError

    def get_gross_properties(
        self,
        **kwargs,
    ) -> res.ConcreteProperties:
        """Returns the gross section properties of the reinforced concrete section.

        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.get_gross_properties`

        :return: Concrete properties object
        :rtype: :class:`~concreteproperties.results.ConcreteProperties`
        """

        return self.concrete_section.get_gross_properties(**kwargs)

    def get_transformed_gross_properties(
        self,
        **kwargs,
    ) -> res.TransformedConcreteProperties:
        """Transforms gross section properties.

        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.get_transformed_gross_properties`

        :return: Transformed concrete properties object
        :rtype:
            :class:`~concreteproperties.results.TransformedConcreteProperties`
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
        :rtype: :class:`~concreteproperties.results.CrackedResults`
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
        :rtype: :class:`~concreteproperties.results.MomentCurvatureResults`
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
        :rtype: :class:`~concreteproperties.results.UltimateBendingResults`
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
        :rtype: :class:`~concreteproperties.results.MomentInteractionResults`
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
        :rtype: :class:`~concreteproperties.results.BiaxialBendingResults`
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
        :rtype: :class:`~concreteproperties.results.StressResult`
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
        :rtype: :class:`~concreteproperties.results.StressResult`
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
        :rtype: :class:`~concreteproperties.results.StressResult`
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
        :rtype: :class:`~concreteproperties.results.StressResult`
        """

        return self.concrete_section.calculate_ultimate_stress(**kwargs)


class AS3600(DesignCode):
    """Design code class for Australian standard AS 3600:2018."""

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
        :type concrete_section:
            :class:`~concreteproperties.concrete_section.ConcreteSection`
        """

        self.concrete_section = concrete_section

        # determine reinforcement class
        self.reinforcement_class = "N"

        for steel_geom in self.concrete_section.steel_geometries:
            if steel_geom.material.stress_strain_profile.fracture_strain < 0.05:
                self.reinforcement_class = "L"

        # re-run plastic property calculation with max compressive strain
        self.concrete_section.calculate_gross_plastic_properties(
            max_compressive_strain=0.0025
        )

    def create_concrete_material(
        self,
        compressive_strength: float,
        colour: Optional[str] = "lightgrey",
    ) -> Concrete:
        """Returns a concrete material object to AS 3600:2018.

        | **Material assumptions:**
        | - *Density*: 2400 kg/m\ :sup:`3`
        | - *Elastic modulus*: Interpolated from Table 3.1.2
        | - *Service stress-strain profile*: Linear with no tension, compressive strength
          at 0.9 * f'c
        | - *Ultimate stress-strain profile*: Rectangular stress block, parameters from
          Cl. 8.1.3
        | - *Alpha squash*: From Cl. 10.6.2.2
        | - *Flexural tensile strength*: From Cl. 3.1.1.3

        :param float compressive_strength: Characteristic compressive strength of
            concrete at 28 days in megapascals (MPa)
        :param colour: Colour of the concrete for rendering
        :type colour: Optional[str]

        :raises ValueError: If compressive_strength is not between 20 MPa and 100 MPa.

        :return: Concrete material object
        :rtype: :class:`~concreteproperties.material.Concrete`
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
        yield_strength: Optional[float] = 500,
        ductility_class: Optional[str] = "N",
        colour: Optional[str] = "grey",
    ) -> Steel:
        """Returns a steel material object.

        | **Material assumptions:**
        | - *Density*: 7850 kg/m\ :sup:`3`
        | - *Elastic modulus*: 200,000 MPa
        | - *Stress-strain profile:* Elastic-plastic, fracture strain from Table 3.2.1

        :param yield_strength: Steel yield strength
        :type yield_strength: Optional[float]
        :param ductility_class: Steel ductility class ("N" or "L")
        :type ductility_class: Optional[str]
        :param colour: Colour of the steel for rendering
        :type colour: Optional[str]

        :raises ValueError: If ductility_class is not N or L

        :return: Steel material object
        :rtype: :class:`~concreteproperties.material.Steel`
        """

        if ductility_class == "N":
            fracture_strain = 0.05
        elif ductility_class == "L":
            fracture_strain = 0.015
        else:
            raise ValueError("ductility_class must be N or L.")

        return Steel(
            name=f"{yield_strength:.0f} MPa Steel (AS 3600:2018)",
            density=7.85e-6,
            stress_strain_profile=ssp.SteelElasticPlastic(
                yield_strength=yield_strength,
                elastic_modulus=200e3,
                fracture_strain=fracture_strain,
            ),
            colour=colour,
        )

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

        :param float n_u: Axial force in member
        :param float n_ub: Axial force at balanced point
        :param float n_uot: Axial force at ultimate tension load
        :param float k_uo: Neutral axis parameter at pure bending
        :param float phi_0: Capacity reduction factor for dominant compression

        :return: Capacity reduction factor
        :rtype: float
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

        :param float theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)

        :return: Bending parameter k_uo
        :rtype: float
        """

        pure_res = self.concrete_section.ultimate_bending_capacity(theta=theta)

        return pure_res.k_u

    def get_n_ub(
        self,
        theta: float,
    ) -> float:
        r"""Returns n_ub for the reinforced concrete cross-section given ``theta``.

        :param float theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)

        :return: Balanced axial force n_ub
        :rtype: float
        """

        # 1) find d_0
        d_0 = 0
        extreme_geom = None

        # calculate extreme fibre in global coordinates
        extreme_fibre, _ = utils.calculate_extreme_fibre(
            points=self.concrete_section.geometry.points, theta=theta
        )

        # get depth to extreme tensile steel
        for steel_geom in self.concrete_section.steel_geometries:
            centroid = steel_geom.calculate_centroid()

            # convert centroid to local coordinates
            _, c_v = principal_coordinate(
                phi=theta * 180 / np.pi, x=centroid[0], y=centroid[1]
            )

            # calculate d
            _, ef_v = principal_coordinate(
                phi=theta * 180 / np.pi,
                x=extreme_fibre[0],
                y=extreme_fibre[1],
            )
            d = ef_v - c_v

            if d > d_0:
                d_0 = d
                extreme_geom = steel_geom

        # 2) calculate yield strain
        yield_strain = (
            extreme_geom.material.stress_strain_profile.yield_strength
            / extreme_geom.material.stress_strain_profile.elastic_modulus
        )

        # 3) k_uo at balanced load
        k_uob = 0.003 / (0.003 + yield_strain)

        # 4) calculate d_n at balanced load
        d_nb = k_uob * d_0

        # 5) calculate axial force at balanced load
        balanced_res = self.concrete_section.calculate_ultimate_section_actions(
            d_n=d_nb, ultimate_results=res.UltimateBendingResults(theta=theta)
        )

        return balanced_res.n

    def ultimate_bending_capacity(
        self,
        theta: Optional[float] = 0,
        n: Optional[float] = 0,
        phi_0: Optional[float] = 0.6,
    ) -> Tuple[res.UltimateBendingResults, res.UltimateBendingResults, float]:
        r"""Calculates the ultimate bending capacity with capacity factors to
        AS 3600:2018.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :type theta: Optional[float]
        :param n: Net axial force
        :type n: Optional[float]
        :param phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)
        :type phi_0: Optional[float]

        :return: Factored and unfactored ultimate bending results objects, and capacity
            reduction factor *(factored_results, unfactored_results, phi)*
        :rtype: Tuple[:class:`~concreteproperties.results.UltimateBendingResults`,
            :class:`~concreteproperties.results.UltimateBendingResults`, float]
        """

        # get parameters to determine phi
        n_uot = self.concrete_section.gross_properties.tensile_load
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
            rtol=1e-6,
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
        factored_ult_res.m_u *= phi

        return factored_ult_res, ult_res, phi

    def moment_interaction_diagram(
        self,
        phi_0: Optional[float] = 0.6,
        **kwargs,
    ) -> Tuple[res.MomentInteractionResults, res.MomentInteractionResults, List[float]]:
        """Generates a moment interaction diagram with capacity factors to AS 3600:2018.

        :param phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)
        :type phi_0: Optional[float]
        :param kwargs: Keyword arguments passed to
            :meth:`~concreteproperties.concrete_section.ConcreteSection.moment_interaction_diagram`

        :return: Factored and unfactored moment interaction results objects, and list of
            capacity reduction factors *(factored_results, unfactored_results, phis)*
        :rtype: Tuple[:class:`~concreteproperties.results.MomentInteractionResults`,
            :class:`~concreteproperties.results.MomentInteractionResults`, List[float]]
        """

        mi_res = self.concrete_section.moment_interaction_diagram(**kwargs)

        # make a copy of the results to factor
        factored_mi_res = deepcopy(mi_res)

        # list to store phis
        phis = []

        # get required constants
        n_uot = self.concrete_section.gross_properties.tensile_load

        # positive bending
        k_uo = mi_res.results[-2].k_u
        n_ub = self.get_n_ub(theta=mi_res.results[0].theta)

        # negative bending
        if len(mi_res.results_neg) > 0:
            k_uo_neg = mi_res.results_neg[-2].k_u
            n_ub_neg = self.get_n_ub(theta=mi_res.results_neg[0].theta)

        # factor results for positive bending
        for ult_res in factored_mi_res.results:
            phi = self.capacity_reduction_factor(
                n_u=ult_res.n, n_ub=n_ub, n_uot=n_uot, k_uo=k_uo, phi_0=phi_0
            )
            ult_res.n *= phi
            ult_res.m_x *= phi
            ult_res.m_y *= phi
            ult_res.m_u *= phi
            phis.append(phi)

        # factor results for negative bending
        for ult_res in factored_mi_res.results_neg:
            phi = self.capacity_reduction_factor(
                n_u=ult_res.n, n_ub=n_ub_neg, n_uot=n_uot, k_uo=k_uo_neg, phi_0=phi_0
            )
            ult_res.n *= phi
            ult_res.m_x *= phi
            ult_res.m_y *= phi
            ult_res.m_u *= phi
            phis.append(phi)

        return factored_mi_res, mi_res, phis

    def biaxial_bending_diagram(
        self,
        n: Optional[float] = 0,
        n_points: Optional[int] = 48,
        phi_0: Optional[float] = 0.6,
    ) -> Tuple[res.BiaxialBendingResults, List[float]]:
        """Generates a biaxial bending with capacity factors to AS 3600:2018.

        :param n: Net axial force
        :type n: Optional[float]
        :param n_points: Number of calculation points between the decompression
        :type n_points: Optional[int]
        :param phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)
        :type phi_0: Optional[float]

        :return: Factored biaxial bending results object and list of capacity reduction
            factors *(factored_results, phis)*
        :rtype: Tuple[:class:`~concreteproperties.results.BiaxialBendingResults`,
            List[float]]
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
                f_ult_res, _, phi = self.ultimate_bending_capacity(theta=theta, n=n)
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
