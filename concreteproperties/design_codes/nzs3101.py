from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from dataclasses import dataclass, field
import concreteproperties.results as res
import concreteproperties.stress_strain_profile as ssp
import concreteproperties.utils as utils
import numpy as np
from concreteproperties.design_codes.design_code import DesignCode
from concreteproperties.material import Concrete, SteelBar
from rich.live import Live

if TYPE_CHECKING:
    from concreteproperties.concrete_section import ConcreteSection


class NZS3101(DesignCode):
    """Design code class for the New Zealand concrete design standard NZS3101:2006. Also
    implements the requirements of the NZSEE C5 assessment guidelines for probable
    strength design.

    .. note::
        Note that this design code currently only supports
        :class:`~concreteproperties.material.Concrete` and
        :class:`NZS3101.SteelBarNZ` material objects. Meshed
        :class:`~concreteproperties.material.Steel` material objects are **not**
        supported as this falls under the composite structures design code.
    """

    # TODO - Implement phi=0.75 option for singly reinforced wall design
    # TODO - Implement method for max_comp_strength for walls (currently only
    #        implemented for columns to NZS3101 Chapter 10)

    def __init__(self):
        """Inits the NZS3101 class."""
        self.analysis_code = "NZS3101:2006"
        super().__init__()

    @dataclass
    class SteelBarNZ(SteelBar):
        """Class for a steel bar material to NZS3101, treated as a lumped circular mass
        with a constant strain.

        :param name: Steel bar material name
        :param steel_grade: Designation of the grade of reinforcement bar to be
            analysed, included predefined current and historic grades are detailed in
            the :meth:`NZS3101.create_steel_material` method
        :param density: Steel bar density (mass per unit volume)
        :param phi_os: Overstrength factor depending on reinforcement grade
            (:math:`\phi_{o,f_y}`), refer to NZS3101:2006 CL 2.6.5.5 or NZSEE C5
            assessment guidelines C5.4.3
        :param stress_strain_profile: Steel bar stress-strain profile
        :param colour: Colour of the material for rendering
        """

        name: str
        steel_grade: str
        density: float
        phi_os: float
        stress_strain_profile: ssp.StressStrainProfile
        colour: str
        meshed: bool = field(default=False, init=False)

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
                "Meshed reinforcement is not supported in this design code"
            )

    def e_conc(self, compressive_strength: float, density: float = 2300) -> float:
        """Calculates Youngs Modulus (:math:`E_c`) for concrete in accordance with
        NZS3101:2006 CL 5.2.3(b).

        :math:`E_c=\displaystyle{4700\sqrt{f'_c}\\frac{\\rho}{2300}}`

        :param compressive_strength: 28 day compressive concrete strength (MPa)
        :param density: Concrete density :math:`\\rho` in accordance with NZS3101:2006
            CL 5.2.2, defaults to 2300 kg/m\ :sup:`3` for normal weight concrete
        :return: :math:`E_c`, Youngs Modulus (MPa)
        """
        # Check low and high limit on density in NZS3101:2006 CL 5.2.2 for E_c equation
        # to be valid
        low_limit = 1800
        high_limit = 2800

        # check upper and lower concrete strengths
        self.check_density_limits(density, low_limit, high_limit)

        E_c = (4700 * (compressive_strength**0.5)) * (density / 2300) ** 1.5

        return E_c

    def check_density_limits(self, density: float, low_limit: float, high_limit: float):
        """Checks that the density is within the bounds outlined within NZS3101:2006
        CL 5.2.2 for the elastic modulus expression within NZS3101:2006 CL 5.2.3(b) to
        be valid.

        :param density: Concrete density :math:`\\rho` in accordance with NZS3101:2006
            CL 5.2.2
        :param low_limit: Lower limit for density from NZS3101:2006 CL 5.2.2
        :param high_limit: Upper limit for density from NZS3101:2006 CL 5.2.2
        :raises ValueError: If density is outside of the limits within NZS3101:2006 CL
            5.2.2
        """
        if low_limit <= density <= high_limit:
            pass
        else:
            raise ValueError(
                f"The specified concrete density of {density}kg/m^3 is not within the "
                f"bounds of {low_limit}kg/m^3 & {high_limit}kg/m^3 for the "
                f"{self.analysis_code} Elastic Modulus eqn to be applicable"
            )

    def alpha_1(self, compressive_strength: float) -> float:
        """Scaling factor relating the nominal 28 day concrete compressive strength to
        the effective concrete compressive strength used for design purposes within the
        concrete stress block. For an equivalent rectangular compressive stress block it
        relates the 28 day concrete compressive strength (:math:`f'c`) to the average
        concrete compressive design strength (:math:`f_{ave}`). A function of the
        concrete compressive strength.

        :math:`\quad\\alpha_1=\displaystyle{\\frac{f_{ave}}{f'_c}}`

        Where:-

        :math:`\quad\\alpha_1=0.85-0.004(f'_c-55)\quad:0.75\leq\\alpha_1\leq0.85`

        :param compressive_strength: 28 day compressive design strength (MPa)
        :return: :math:`\\alpha_1` factor
        """
        if compressive_strength <= 55:
            alpha_1 = 0.85
        if compressive_strength > 55:
            alpha_1 = max(0.75, 0.85 - 0.004 * (compressive_strength - 55))

        return alpha_1

    def beta_1(self, compressive_strength: float) -> float:
        """Scaling factor relating the depth of an equivalent rectangular compressive
        stress block (:math:`a`) to the depth of the neutral axis (:math:`c`).
        A function of the concrete compressive strength.

        :math:`\quad\\beta_1=\displaystyle{\\frac{a}{c}}`

        Where:-

        :math:`\quad\\beta_1=0.85-0.008(f'_c-30)\quad:0.65\leq\\beta_1\leq0.85`

        :param compressive_strength: 28 day compressive design strength (MPa)
        :return: :math:`\\beta_1` factor
        """
        if compressive_strength <= 30:
            beta_1 = 0.85
        if compressive_strength > 30:
            beta_1 = max(0.65, 0.85 - 0.008 * (compressive_strength - 30))

        return beta_1

    def lamda(self, density: float) -> float:
        """Modification factor reflecting the reduced mechanical properties of
        lightweight concrete relative to normal weight concrete of the same compression
        strength.

        :math:`\quad\lambda=0.4+\displaystyle{\\frac{0.6\\rho}{2200}}\\leq1.0`

        :param density: Saturated surface dry density of concrete material
        :return: :math:`\lambda` factor
        """
        lamda = min(0.4 + 0.6 * density / 2200, 1)
        return lamda

    def concrete_capacity(
        self,
        os_design: bool = False,
        prob_design: bool = False,
        add_compressive_strength: float = 15,
    ) -> float:
        """Function to return the nominal concrete capacity or the overstrength concrete
        capacity of a concrete section.

        :param os_design: True if an overstrength capacity of a concrete section is
            required, then the material properties for concrete are scaled to reflect
            the likely maximum material strength properties
        :param prob_design: True if the probable capacity of a concrete section is
            required, then the material properties for concrete and lumped reinforcement
            are scaled to reflect the probable material strength properties
        :param add_compressive_strength: The increase in compressive strength of the
            specified 28 day compressive strength of concrete to reflect the likely
            maximum material strength, defaults to an additional 15 MPa as per
            NZS3101:2006 CL 2.6.5.5(c)
        :return: Nominal, overstrength or probable concrete yield force (N)
        """
        # initiate force variable
        force = 0

        # loop through all concrete geometries
        for conc_geom in self.concrete_section.concrete_geometries:
            # calculate concrete area & compressive strength
            concrete_area = conc_geom.calculate_area()
            compressive_strength = (
                conc_geom.material.ultimate_stress_strain_profile.get_compressive_strength()
            )

            # scale concrete compressive strength for overstrength if specified
            if prob_design:
                mult_compressive_strength = 1.5 if compressive_strength <= 40 else 1.4
                compressive_strength *= mult_compressive_strength
            elif os_design:
                compressive_strength += add_compressive_strength

            # calculate cumulative concrete force
            force += (
                self.alpha_1(compressive_strength)
                * concrete_area
                * compressive_strength
            )

        return force

    def steel_capacity(
        self, os_design: bool = False, prob_design: bool = False
    ) -> float:
        """Function to return the nominal steel reinforcement capacity or the
        overstrength steel reinforcement capacity of a concrete section.

        :param os_design: True if an overstrength capacity of a concrete section is
            required, then the material properties for lumped reinforcement are scaled
            to reflect the likely maximum material strength properties
        :param prob_design: True if the probable capacity of a concrete
            section is required, then the material properties for concrete and lumped
            reinforcement are scaled to reflect the probable material strength
            properties
        :return: Nominal, overstrength or probable steel yield force (N)
        """
        # Retrieve predefined names of probable strength based materials
        _, _, prob_properties = self.predefined_steel_materials()

        # initiate force variable
        force = 0

        # loop through all steel geometries
        for steel_geom in self.concrete_section.reinf_geometries_lumped:
            # calculate reinforcement area & yield strength & steel_grade
            steel_area = steel_geom.calculate_area()
            yield_strength = (
                steel_geom.material.stress_strain_profile.get_yield_strength()
            )
            steel_grade = steel_geom.material.steel_grade

            # establish scaling factor for overstrength if specified
            if os_design:
                mult_yield_strength = steel_geom.material.phi_os
            elif prob_design and steel_grade not in prob_properties:
                mult_yield_strength = 1.08
            else:
                mult_yield_strength = 1.0

            # calculate cumulative reinforcement force
            force += steel_area * yield_strength * mult_yield_strength

        return force

    def max_comp_strength(
        self,
        cpe_design: bool = False,
        os_design: bool = False,
        prob_design: bool = False,
    ) -> float:
        """Function to return the nominal axial load compressive strength of a concrete
        section when the load is applied with zero eccentricity.

        For non-capacity design situations, refer to NZS3101:2006 CL 10.3.4.2:-

        :math:`\quad\displaystyle{\\frac{N^*}{\phi} < 0.85N_{n,max}}`

        For capacity design situations, refer to NZS3101:2006 CL 10.4.4:-

        :math:`\quad N^*_o < 0.7N_{n,max}`

        Where:-

        :math:`\quad N_{n,max} = \\alpha_1f'_c(A_g-A_{st})+f_yA_{st}`

        :param cpe_design: True if the capacity protected element capacity of a concrete
            section is required (i.e. design capacity being checked against O/S
            actions)
        :param os_design: True if the overstrength capacity of a concrete section is
            required, then the material properties for concrete and lumped reinforcement
            are scaled to reflect the likely maximum material strength properties
        :param prob_design: True if the probable capacity of a concrete
            section is required, then the material properties for concrete and lumped
            reinforcement are scaled to reflect the probable material strength
            properties
        :return: Returns the nominal, overstrength or probable axial load compressive
            strength of a concrete section :math:`N_{n,max}`
        """
        # Calculate maximum axial compression strength
        n_n_max = self.steel_capacity(os_design, prob_design) + self.concrete_capacity(
            os_design, prob_design
        )

        if cpe_design:
            max_comp = 0.7 * n_n_max
        else:
            max_comp = 0.85 * n_n_max
        return max_comp

    def max_ten_strength(
        self, os_design: bool = False, prob_design: bool = False
    ) -> float:
        """Function to return the nominal axial load tension strength of a concrete
        section when the load is applied with zero eccentricity.

        :math:`\quad N_{t,max} = f_yA_{st}`

        :param os_design: True if an overstrength capacity of a concrete section is
            required, then the material properties for concrete and lumped reinforcement
            are scaled to reflect the likely maximum material strength properties
        :param prob_design: True if the probable capacity of a concrete
            section is required, then the material properties for concrete and lumped
            reinforcement are scaled to reflect the probable material strength
            properties
        :return: Returns the nominal, overstrength or probable axial tension strength of
            a concrete section :math:`N_{t,max}`
        """
        # Calculate maximum axial tension strength
        max_ten = self.steel_capacity(os_design, prob_design)

        return max_ten

    def check_f_y_limit(self):
        """Checks that the specified steel reinforcement strengths for all defined
        steel geometries comply with NZS3101:2006 CL 5.3.3.

        .. note:: Check does not apply to predefined steel materials based on probable
            strength properties

        :raises ValueError: If steel reinforcement yield strength is greater than 500MPa
            limit in NZS3101:2006 CL 5.3.3
        """
        # Retrieve predefined names of probable strength based materials
        _, _, prob_properties = self.predefined_steel_materials()

        # Upper bound yield strength
        f_y_upper = 500

        # loop through all steel geometries
        for steel_geom in self.concrete_section.reinf_geometries_lumped:
            # calculate defined steel grade & yield strength
            steel_grade = steel_geom.material.steel_grade
            yield_strength = (
                steel_geom.material.stress_strain_profile.get_yield_strength()
            )
            if steel_grade not in prob_properties:
                if yield_strength > f_y_upper:
                    raise ValueError(
                        f"Steel yield strength for '{steel_geom.material.name}' "
                        f"material must be less than {f_y_upper} MPa for the "
                        f"{self.analysis_code} code, {yield_strength:.0f} MPa was "
                        f"specified for this material"
                    )

    def check_f_c_limits(self, pphr_class: str):
        """Checks that a valid Potential Plastic Hinge Region (PPHR) classification has
        been specified, and that the specified compressive strengths for all defined
        concrete geometries comply with NZS3101:2006 CL 5.2.1 for the specified PPHR
        classification.

        :param pphr_class: Potential Plastic Hinge Region (PPHR) classification,
            **NDPR**/**LDPR**/**DPR**.

            - **NDPR** = Nominally Ductile Plastic Region

            - **LDPR** = Limited Ductile Plastic Region

            - **DPR** = Ductile Plastic Region

        :raises ValueError: If specified Potential Plastic Hinge Region (PPHR)
            classification is not NDPR, LDPR or DPR
        :raises ValueError: If specified compressive strength for a concrete geometry
            is not between 20 MPa and 100 MPa for NDPR PPHR's, or is not between 20 MPa
            and 70 MPa for LDPR or DPR PPHR's
        """
        # Lower bound compressive strength
        f_c_lower = 20

        # Upper bound compressive strength & check inputs within acceptable bounds
        if pphr_class.upper() in ["NDPR"]:
            f_c_upper = 100
        elif pphr_class.upper() in ["LDPR", "DPR"]:
            f_c_upper = 70
        else:
            raise ValueError(
                f"The specified PPHR class specified ({pphr_class}) should be NDPR, "
                f"LDPR or DPR for the {self.analysis_code} code, {pphr_class} was "
                f"specified"
            )

        # loop through all concrete geometries
        for conc_geom in self.concrete_section.concrete_geometries:
            # calculate compressive strength
            compressive_strength = (
                conc_geom.material.ultimate_stress_strain_profile.get_compressive_strength()
            )
            if f_c_lower <= compressive_strength <= f_c_upper:
                pass
            else:
                raise ValueError(
                    f"Concrete compressive strength for '{conc_geom.material.name}' "
                    f"material must be between {f_c_lower} MPa & {f_c_upper} MPa for a "
                    f"{pphr_class} PPHR for the {self.analysis_code} code, "
                    f"{compressive_strength:.0f} MPa was specified for this material"
                )

    def create_concrete_material(
        self,
        compressive_strength: float,
        ultimate_strain: float = 0.003,
        density: float = 2300,
        colour: str = "lightgrey",
    ) -> Concrete:
        """Returns a concrete material object to NZS3101:2006.

        .. admonition:: Material assumptions

          - *Density*: Defaults to 2300 kg/m\ :sup:`3` unless supplied as user input

          - *Elastic modulus*: Calculated from NZS3101:2006 Eq. 5-1

          - *Serviceability stress-strain profile*: Linear with no tension

          - *Ultimate stress-strain profile*: Rectangular stress block, parameters from
            NZS3101:2006 CL 7.4.2.7, maximum compressive strain of 0.003

          - *Modulus of rupture*: Calculated from NZS3101:2006 Eq. 5-4

        :param compressive_strength: 28 day compressive design strength (MPa)
        :param ultimate_strain: Maximum concrete compressive strain at crushing of the
            concrete for design
        :param density: Saturated surface dry density of concrete material
        :param colour: Colour of the concrete for rendering, defaults to 'lightgrey'
        :return: Concrete material object
        """
        # create concrete name
        name = (
            f"{compressive_strength:.0f} MPa Conc [{density:.0f} kg/m$^{{{3}}}$] "
            f"\n({self.analysis_code})"
        )

        # calculate elastic modulus
        elastic_modulus = self.e_conc(compressive_strength, density)

        # calculate rectangular stress block parameters
        alpha_1 = self.alpha_1(compressive_strength)
        beta_1 = self.beta_1(compressive_strength)

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
            flexural_tensile_strength=modulus_of_rupture,
            colour=colour,
        )

    def predefined_steel_materials(
        self,
    ) -> Tuple[Dict, List[str], List[str]]:
        """Returns a list of predefined material properties for steel grades for design
        to NZS3101:2006 & NZSEE C5 assessment guidelines.

        Refer to :meth:`NZS3101.create_steel_material` for details of predefined steel
        grades.

        :return: Returns :class:`dict` with standard predefined steel material
            properties based on current steel grade 300E & 500E material properties in
            accordance with NZS3101:2006, and based on historic steel grade material
            properties in accordance with NZSEE C5 assessment guidelines.

            Returns :class:`list` with predefined material grades that have been
            defined on characteristic strength material properties and
            :class:`list` of predefined material grades that have been defined
            based on probable strength material properties.

        .. admonition:: Dictionary keys

          +------------+--------------------------------------------------------------+
          | Dict key   | Description                                                  |
          +============+==============================================================+
          | 1          | Charateristic yield strength (:math:`f_y`)                   |
          |            | or probable yield strength (:math:`f_{yp}`)                  |
          +------------+--------------------------------------------------------------+
          | 2          | Fracture strain (:math:`\\varepsilon_{su}`)                   |
          +------------+--------------------------------------------------------------+
          | 3          | Overstrength factor (:math:`\phi_{o,f_y}` or :math:`\phi_o`) |
          |            | (note if probable strength based material                    |
          |            | property is specified then the true O/S factor to be applied |
          |            | against the characteristic yield strength is 1.08 times this |
          |            | value).                                                      |
          +------------+--------------------------------------------------------------+
          | 4          | **True** if probable strength based yield strength &         |
          |            | overstrength factor. **False** if lower characteristic       |
          |            | strength based yield strength & overstrength factor.         |
          +------------+--------------------------------------------------------------+
        """
        # Create dictionary with predefined steel grades
        properties_dict = {
            "pre-1945": {1: 280.0, 2: 0.1, 3: 1.25, 4: True},
            "33": {1: 280.0, 2: 0.1, 3: 1.25, 4: True},
            "40": {1: 324.0, 2: 0.15, 3: 1.25, 4: True},
            "275": {1: 324.0, 2: 0.15, 3: 1.25, 4: True},
            "hy60": {1: 455.0, 2: 0.12, 3: 1.5, 4: True},
            "380": {1: 455.0, 2: 0.12, 3: 1.5, 4: True},
            "430": {1: 464.0, 2: 0.12, 3: 1.25, 4: True},
            "300": {1: 324.0, 2: 0.15, 3: 1.25, 4: True},
            "500n": {1: 500.0, 2: 0.05, 3: 1.5, 4: True},
            "500": {1: 540.0, 2: 0.10, 3: 1.25, 4: True},
            "cd_mesh": {1: 600.0, 2: 0.015, 3: 1.2, 4: True},
            "duc_mesh": {1: 540.0, 2: 0.03, 3: 1.2, 4: True},
            "300e": {1: 300.0, 2: 15 / 100, 3: 1.35, 4: False},
            "500e": {1: 500.0, 2: 10 / 100, 3: 1.35, 4: False},
        }

        # list to store predefined probable strength based steel grades
        prob_properties = []
        # list to store predefined characteristic strength based steel grades
        nom_properties = []

        # Create list of probable strength based and nominal strength based steel grades
        for properties in properties_dict:
            if properties_dict[properties][4]:
                prob_properties.append(properties)
            else:
                nom_properties.append(properties)

        return properties_dict, nom_properties, prob_properties

    def create_steel_material(
        self,
        steel_grade: Optional[str] = None,
        yield_strength: Optional[float] = None,
        fracture_strain: Optional[float] = None,
        phi_os: Optional[float] = None,
        colour: str = "red",
    ) -> NZS3101.SteelBarNZ:
        """Returns a steel material object specific to the NZS3101:2006 code.

        .. admonition:: Material assumptions

          - *Density*: 7850 kg/m\ :sup:`3`

          - *Elastic modulus*: 200000 MPa

          - *Stress-strain profile*: Elastic-plastic, fracture strain
            :math:`\\varepsilon_{su}` from AS/NZS4671 Table 7.2(A) or NZSEE C5
            assessment guidelines (for historic reinforcement grades)

        :param steel_grade: Designation of the grade of reinforcement bar to be
            analysed, included predefined current and historic grades are as follows:-

        .. note::
          By using a valid steel grade designation the required input parameters are
          initiated with the required values for current reinforcement grades from the
          AS/NZS4671 standard or for historic grades from the NZSEE C5 assessment
          guidelines. Note user may overwrite any parameter of a predefined material
          by providing that parameter as input to :meth:`NZS3101.create_steel_material`.

          Note if no predefined steel grade is provided, a steel grade name of
          **'user_' + yield strength** is utilised.

        .. admonition:: NZS3101:2006 & NZSEE C5 asessment guidelines predefined steel
            materials

          **NZS3101:2006 characteristic yield strength based predefined materials**

          - **300e** - Use for design to NZS3101:2006 provisions

            - Characteristic yield strength :math:`f_y` = 300 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 15% or 0.15

            - Overstrength factor :math:`\phi_{o,f_y}` = 1.35

          - **500e** - Use for design to NZS3101:2006 provisions

            - Characteristic yield strength :math:`f_y` = 500 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 10% or 0.10

            - Overstrength factor :math:`\phi_{o,f_y}` = 1.35

          **NZSEE C5 guidelines probable yield strength based predefined materials**

          - **pre-1945** - Use for probable strength design to NZSEE C5 assessment
            guidelines

            - Probable yield strength :math:`f_{yp}` = 280 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 10% or 0.10

            - Overstrength factor :math:`\phi_{f_o}` = 1.25

          - **33** - Use for probable strength design to NZSEE C5 assessment
            guidelines

            - Probable yield strength :math:`f_{yp}` = 280 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 10% or 0.10

            - Overstrength factor :math:`\phi_{f_o}` = 1.25

          - **40** - Use for probable strength design to NZSEE C5 assessment
            guidelines

            - Probable yield strength :math:`f_{yp}` = 324 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 15% or 0.15

            - Overstrength factor :math:`\phi_{f_o}` = 1.25

          - **275** - Use for probable strength design to NZSEE C5 assessment
            guidelines

            - Probable yield strength :math:`f_{yp}` = 324 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 15% or 0.15

            - Overstrength factor :math:`\phi_{f_o}` = 1.25

          - **hy60** - Use for probable strength design to NZSEE C5 assessment
            guidelines

            - Probable yield strength :math:`f_{yp}` = 455 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 12% or 0.12

            - Overstrength factor :math:`\phi_{f_o}` = 1.5

          - **380** - Use for probable strength design to NZSEE C5 assessment
            guidelines

            - Probable yield strength :math:`f_{yp}` = 455 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 12% or 0.12

            - Overstrength factor :math:`\phi_{f_o}` = 1.5

          - **430** - Use for probable strength design to NZSEE C5 assessment
            guidelines

            - Probable yield strength :math:`f_{yp}` = 464 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 12% or 0.12

            - Overstrength factor :math:`\phi_{f_o}` = 1.25

          - **300** - Use for probable strength design to NZSEE C5 assessment
            guidelines

            - Probable yield strength :math:`f_{yp}` = 324 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 15% or 0.15

            - Overstrength factor :math:`\phi_{f_o}` = 1.25

          - **500n** - Use for probable strength design to NZSEE C5 assessment
            guidelines

            - Probable yield strength :math:`f_{yp}` = 500 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 5% or 0.05

            - Overstrength factor :math:`\phi_{f_o}` = 1.5

          - **500** - Use for probable strength design to NZSEE C5 assessment
            guidelines

            - Probable yield strength :math:`f_{yp}` = 540 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 10% or 0.10

            - Overstrength factor :math:`\phi_{f_o}` = 1.25

          - **cd_mesh** - Use for probable strength design to NZSEE C5 assessment
            guidelines

            - Probable yield strength :math:`f_{yp}` = 600 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 1.5% or 0.015

            - Overstrength factor :math:`\phi_{f_o}` = 1.2

          - **duc_mesh** - Use for probable strength design to NZSEE C5 assessment
            guidelines

            - Probable yield strength :math:`f_{yp}` = 500 MPa

            - Fracture strain :math:`\\varepsilon_{su}` = 3% or 0.03

            - Overstrength factor :math:`\phi_{f_o}` = 1.2

        :param yield_strength: Steel characteristic yield strength (MPa)

            - Note for a predefined steel grade based on probable strength
              properties this is interpreted as the probable yield strength.

            - Note for a user defined steel grade, this is **always** entered on the
              basis of a characteristic yield strength, even if undertaking a probable
              strength based analysis. The analysis will internally scale the
              characteristic yield stress by 1.08 as per NZSEE C5 assessment guidelines
              C5.4.3.

        :param fracture_strain: Lower bound tensile strain (:math:`\\varepsilon_{su}`),
            based on characteristic uniform elongation limit from AS/NZS4671 Table
            7.2(A) or NZSEE C5 assessment guidelines Table C5.4.
        :param phi_os: Overstrength factor depending on reinforcement grade
            (:math:`\phi_{o,f_y}` or :math:`\phi_o`), refer to NZS3101:2006 CL 2.6.5.5,
            or for a probable strength assessment to the NZSEE C5 assessment guidelines
            refer to NZSEE C5 Table C5.4.
        :param colour: Colour of the steel for rendering, if user does not provide a
            value, characteristic strength based materials will be rendered as red, and
            probable strength based materials will be rendered as blue.
        :raises Exception: If a predefined steel grade is not provided and the required
            material properties have not been provided. For creating a user defined
            steel material, values for **yield_strength**, **fracture_strain** &
            **phi_os** are required to define a valid user defined material.
        :return: Steel bar material object
        """
        # Populate dictionary with predefined material properties
        (
            properties_dict,
            nom_properties,
            prob_properties,
        ) = self.predefined_steel_materials()

        # Create list of all predefined steel materials
        pre_def_properties = [key.lower() for key in properties_dict.keys()]

        if steel_grade is None or steel_grade.lower() not in pre_def_properties:
            # check if all user defined parameters are provided
            if yield_strength is None or fracture_strain is None or phi_os is None:
                raise Exception(
                    f"A predefined steel grade has not been provided, to create a user "
                    f"defined steel material a yield strength, fracture strain and "
                    f"overstrength factor are required to be specified.\n   Valid "
                    f"predefined Characteristic strength based steel grades are "
                    f"{nom_properties}, refer AS/NZS4671\n   Valid predefined Probable "
                    f"strength based steel grades are {prob_properties}, refer NZSEE "
                    f"C5 assessment guidelines"
                )
        elif steel_grade.lower() in pre_def_properties:
            # initiate predefined properties unless there is a user defined property
            if yield_strength is None:
                yield_strength = properties_dict[steel_grade.lower()][1]
            if fracture_strain is None:
                fracture_strain = properties_dict[steel_grade.lower()][2]
            if phi_os is None:
                phi_os = properties_dict[steel_grade.lower()][3]

        # create steel reinforcement name
        name = f"{yield_strength:.0f} MPa Steel\n({self.analysis_code})"

        # create steel grade name if not provided
        if steel_grade is None:
            steel_grade = f"user_{yield_strength:.0f}"

        # define density
        density = 7850

        # define elastic modulus
        elastic_modulus = 200_000

        # change colour if predefined probable strength based steel grade
        if steel_grade.lower() in prob_properties and colour in ["red"]:
            colour = "steelblue"

        return NZS3101.SteelBarNZ(
            name=name,
            steel_grade=steel_grade,
            density=density,
            phi_os=phi_os,
            stress_strain_profile=ssp.SteelElasticPlastic(
                yield_strength=yield_strength,
                elastic_modulus=elastic_modulus,
                fracture_strain=fracture_strain,
            ),
            colour=colour,
        )

    def capacity_reduction_factor(
        self, analysis_type: str
    ) -> Tuple[float, bool, bool, bool]:
        """Returns the appropriate NZS3101:2006 or NZSEE C5 assessment guidelines
        capacity reduction factor dependant on the type of analysis specified. Refer to
        NZS3101:2006 CL 2.3.2.2 or NZSEE C5 assessment guidelines C5.5.1.4.

        :param analysis_type: The type of cross section analysis to undertake on the
            defined concrete section, by default a normal nominal strength design check
            is undertaken:-

            - **nom_chk** - Nominal strength design check.

              Returns the normal nominal strength section design capacity, i.e.
              undertakes the cross section analysis based on the following
              assumptions:-

              - Using a strength reduction factor of :math:`\phi` = 0.85 in
                accordance with NZS3101:2006 CL 2.3.2.2.

              - Using the lower 5% characteristic reinforcement yield strengths.

              - Using the lower 5% characteristic concrete 28 day compressive
                design strength.

            - **cpe_chk** - Capacity Protected Element (CPE) strength design check.

              Returns the capacity protected element section design capacity, i.e.
              undertakes the cross section analysis based on the following
              assumptions:-

              - Using a strength reduction factor of :math:`\phi` = 1.0 in
                accordance with NZS3101:2006 CL 2.3.2.2.

              - Using the lower 5% characteristic reinforcement yield strengths.

              - Using the lower 5% characteristic concrete 28 day compressive
                design strength.

            - **os_chk** - Overstrength (O/S) strength design check.

              Returns the O/S (overstrength) section design capacity, i.e. undertakes
              the cross section analysis based on the following assumptions:-

              - Using a strength reduction factor of :math:`\phi` = 1.0 in
                accordance with NZS3101:2006 CL 2.3.2.2.

              - Using a likely maximum reinforcement yield strength of
                :math:`\phi_{o,f_y}f_y`, typically :math:`\phi_{o,f_y}=1.35` in
                accordance with NZS3101:2006 CL 2.6.5.5(a) for grade 300E or grade
                500E reinforcement which complies with AS/NZS4671. User may define
                custom overstrength factors when defining steel reinforcement
                materials using :class:`NZS3101.SteelBarNZ`.

              - Using a likely maximum compression strength of the concrete based
                on the lower 5% characteristic concrete 28 day strength plus 15
                MPa, i.e. :math:`f'_c` + 15 in accordance with NZS3101:2006 CL
                2.6.5.5(c).

            - **prob_chk** - Probable strength design check to NZSEE C5 guidelines based
              on NZS3101:2006 analysis provisions.

              Returns the probable strength section design capacity, i.e.
              undertakes the cross section analysis based on the following assumptions:-

              - Using a strength reduction factor of :math:`\phi` = 1.0 in
                accordance with NZSEE C5 assessment guidelines C5.5.1.4.

              - Using the probable reinforcement yield strengths in accordance
                with NZSEE C5 assessment guidelines C5.4.3, typically
                :math:`f_{yp}=1.08f_y` in accordance with NZSEE C5 assessment
                guidelines C5.4.3. User may define custom probable strengths when
                defining steel reinforcement materials using
                :class:`NZS3101.SteelBarNZ`. Note if one of the predefined
                probable strength based steel grade materials are being utilised, then
                the yield strength is inclusive of the 1.08 factor noted above.

              - Using the probable compressive strength of the concrete in
                accordance with NZSEE C5 guidelines C5.4.2.2, typically for
                specified 28 day concrete compressive strengths of less than or
                equal to 40 MPa, :math:`f'_{cp}=1.5f'_c`, and for greater than
                40 MPa, :math:`f'_{cp}=1.4f'_c`.

            - **prob_os_chk** - Probable overstrength design check to NZSEE C5
              guidelines based on NZS3101:2006 analysis provisions.

              Returns the probable O/S (overstrength) strength section design
              capacity, i.e. undertakes the cross section analysis based on the
              following assumptions:-

              - Using a strength reduction factor of :math:`\phi` = 1.0 in
                accordance with NZSEE C5 assessment guidelines C5.5.1.4.

              - Using the probable overstrength reinforcement yield strengths in
                accordance with NZSEE C5 assessment guidelines C5.4.3, typically
                :math:`f_o=\phi_of_{yp}` in accordance with NZSEE C5 assessment
                guidelines C5.4.3 & C5.5.2.3. User may define custom overstrength
                factors strengths when defining steel reinforcement materials
                using :class:`NZS3101.SteelBarNZ`. Note if one of the predefined
                probable strength based steel grade materials are being utilised,
                then the overstrength factor being applied to the yield strength
                is inclusive of the 1.08 factor on the lower bound yield strength.

                :math:`\quad\phi_o=\displaystyle{\\frac{f_o}{f_{yp}}}`

                Where:-

                :math:`\quad f_{yp}=1.08f_y`

              - Using the probable compressive strength of the concrete in
                accordance with NZSEE C5 guidelines C5.4.2.2, typically for
                specified 28 day concrete compressive strengths of less than or
                equal to 40 MPa, :math:`f'_{cp}=1.5f'_c`, and for greater than
                40 MPa, :math:`f'_{cp}=1.4f'_c`.

                Note there is no enhancement to concrete strength for overstrength
                checks in accordance with the NZSEE C5 assessment guidelines.

        :raises ValueError: If analysis type is not valid
        :raises Exception: If a characteristic strength based analysis is specified, but
            a predefined probable strength based steel grade has been specified.
            Undertaking a non NZSEE C5 assessment guidelines analysis on a probable
            strength based steel grade is not consistent with an analysis to
            NZS3101:2006.
        :return: Returns the appropriate strength reduction factor :math:`\\phi` and
            variables to indicate the type of analysis being requested.
        """
        if analysis_type.lower() in ["nom_chk"]:
            phi = 0.85
            cpe_design = False
            os_design = False
            prob_design = False
        elif analysis_type.lower() in ["cpe_chk"]:
            phi = 1.0
            cpe_design = True
            os_design = False
            prob_design = False
        elif analysis_type.lower() in ["os_chk"]:
            phi = 1.0
            cpe_design = False
            os_design = True
            prob_design = False
        elif analysis_type.lower() in ["prob_chk"]:
            phi = 1.0
            cpe_design = False
            os_design = False
            prob_design = True
        elif analysis_type.lower() in ["prob_os_chk"]:
            phi = 1.0
            cpe_design = False
            os_design = True
            prob_design = True
        else:
            raise ValueError(
                f"The specified analysis type of '{analysis_type}' should be either "
                f"'nom_chk', 'cpe_chk', 'os_chk', 'prob_chk' or 'prob_os_chk'"
                f"for a {self.analysis_code} code analysis"
            )

        # check that if using a predefined probable strength based steel grade
        # that only a probable strength check is being undertaken
        _, nom_properties, prob_properties = self.predefined_steel_materials()
        for steel_geom in self.concrete_section.reinf_geometries_lumped:
            if (
                analysis_type.lower() in ["nom_chk", "cpe_chk", "os_chk"]
                and steel_geom.material.steel_grade.lower() in prob_properties
            ):
                raise Exception(
                    f"'{analysis_type}' analysis is not able to be undertaken on the "
                    f"provided concrete section as it contains predefined steel "
                    f"materials based on probable yield strengths and will give "
                    f"erroneous results for a design to {self.analysis_code} as "
                    f"material is not based on characteristic yield strengths. Define "
                    f"a user defined or predefined steel material based on "
                    f"characteristic yield properties to undertake a "
                    f"'{analysis_type}' concrete section analysis.\n   Note "
                    f"predefined steel grades based on characteristic strength based "
                    f"materials are {nom_properties}\n   Note analysis types "
                    f"consistent with a probable strength based material are "
                    f"['prob_chk', 'prob_os_chk'], undertaken in accordance with NZSEE "
                    f"C5 assessment guidelines"
                )

        return phi, cpe_design, os_design, prob_design

    def create_os_section(
        self, add_compressive_strength: float = 15
    ) -> ConcreteSection:
        """Creates a concrete section with likely maximum material strength properties
        for a cross section analysis to NZS3101:2006. Concrete and steel reinforcement
        strength properties are modified in accordance with NZS3101:2006 CL 2.6.5.5 to
        reflect likely maximum material strengths.

        :param add_compressive_strength: The increase in compressive strength of the
            specified 28 day compressive strength of concrete to reflect the likely
            maximum material strength, defaults to an additional 15 MPa as per
            NZS3101:2006 CL 2.6.5.5(c)
        :return: Returns a concrete section with material strengths modified to reflect
            likely maximum material strengths to enable an overstrength based analysis
            to be undertaken
        """
        # create copy of concrete section to modify materials to overstrength properties
        os_concrete_section = deepcopy(self.concrete_section)

        # loop through all concrete geometries and update to overstrength properties
        for conc_geom in os_concrete_section.concrete_geometries:
            # retrieve previous nominal/characteristic material properties
            prev_compressive_strength = (
                conc_geom.material.ultimate_stress_strain_profile.get_compressive_strength()
            )
            prev_ultimate_strain = (
                conc_geom.material.ultimate_stress_strain_profile.get_ultimate_compressive_strain()
            )
            prev_density = conc_geom.material.density
            prev_colour_conc = conc_geom.material.colour

            # update concrete material to new material with overstrength properties
            conc_geom.material = self.create_concrete_material(
                compressive_strength=prev_compressive_strength
                + add_compressive_strength,
                ultimate_strain=prev_ultimate_strain,
                density=prev_density,
                colour=prev_colour_conc,
            )

        # loop through all steel geometries and update to overstrength properties
        for steel_geom in os_concrete_section.reinf_geometries_lumped:
            # retrieve previous nominal/characteristic material properties
            prev_steel_grade = steel_geom.material.steel_grade
            prev_yield_strength = (
                steel_geom.material.stress_strain_profile.get_yield_strength()
            )
            prev_fracture_strain = (
                -steel_geom.material.stress_strain_profile.get_ultimate_tensile_strain()
            )
            prev_phi_os = steel_geom.material.phi_os
            prev_colour_steel = steel_geom.material.colour

            # update steel reinforcement material to new material with overstrength
            # properties
            steel_geom.material = self.create_steel_material(
                steel_grade=prev_steel_grade,
                yield_strength=prev_yield_strength * prev_phi_os,
                fracture_strain=prev_fracture_strain,
                phi_os=prev_phi_os,
                colour=prev_colour_steel,
            )

        return os_concrete_section

    def create_prob_section(self, os_design: bool = False) -> ConcreteSection:
        """Creates a concrete section with probable strength material properties
        for a cross section analysis to NZS3101:2006 & NZSEE C5 assessment guidelines.
        Concrete and steel reinforcement strength properties are modified in accordance
        with NZSEE C5 assessment guidelines C5.4.2.2 & C5.4.3.

        :param os_design: True if an overstrength probable capacity of a concrete
            section is required, then the material properties for concrete and lumped
            reinforcement are scaled to reflect the probable overstrength material
            strength properties, defaults to False which only scales the material
            properties for concrete to reflec tthe probable material strength properties
        :return: Returns a concrete section with material strengths modified to reflect
            probable material strengths or probable overstrength material strengths, to
            enable a probable strength or probable overstrength based analysis
            to be undertaken
        """
        # create copy of concrete section to modify materials to probable strength
        # properties
        prob_concrete_section = deepcopy(self.concrete_section)

        # loop through all concrete geometries & update to probable strength properties
        for conc_geom in prob_concrete_section.concrete_geometries:
            # retrieve previous nominal/characteristic material properties
            prev_compressive_strength = (
                conc_geom.material.ultimate_stress_strain_profile.get_compressive_strength()
            )
            prev_ultimate_strain = (
                conc_geom.material.ultimate_stress_strain_profile.get_ultimate_compressive_strain()
            )
            prev_density = conc_geom.material.density
            prev_colour_conc = conc_geom.material.colour

            # update concrete material to new material with probable strength properties
            mult_compressive_strength = 1.5 if prev_compressive_strength <= 40 else 1.4
            conc_geom.material = self.create_concrete_material(
                compressive_strength=prev_compressive_strength
                * mult_compressive_strength,
                ultimate_strain=prev_ultimate_strain,
                density=prev_density,
                colour=prev_colour_conc,
            )

            # update modulus of rupture to probable strength based value
            conc_geom.material.flexural_tensile_strength = 0.55 * np.sqrt(
                prev_compressive_strength * mult_compressive_strength
            )

        # populate list with predefined probable strength based steel grades
        _, _, prob_properties = self.predefined_steel_materials()

        # loop through all steel geometries and update to probable strength properties
        for steel_geom in prob_concrete_section.reinf_geometries_lumped:
            # retrieve previous nominal/characteristic material properties
            prev_steel_grade = steel_geom.material.steel_grade
            prev_yield_strength = (
                steel_geom.material.stress_strain_profile.get_yield_strength()
            )
            prev_fracture_strain = (
                -steel_geom.material.stress_strain_profile.get_ultimate_tensile_strain()
            )
            prev_phi_os = steel_geom.material.phi_os
            prev_colour_steel = steel_geom.material.colour

            # determine appropriate scaling factor for yield strength depen depending on
            # defined mmaterial and anlysis type
            if prev_steel_grade not in prob_properties and os_design:
                mult_prob_strength = prev_phi_os
            elif prev_steel_grade not in prob_properties and not os_design:
                mult_prob_strength = 1.08
            elif os_design:
                mult_prob_strength = prev_phi_os
            else:
                mult_prob_strength = 1.0

            # update steel reinforcement material to new material with probable strength
            # properties
            steel_geom.material = self.create_steel_material(
                steel_grade=prev_steel_grade,
                yield_strength=prev_yield_strength * mult_prob_strength,
                fracture_strain=prev_fracture_strain,
                phi_os=prev_phi_os,
                colour=prev_colour_steel,
            )

        return prob_concrete_section

    def ultimate_bending_capacity(
        self,
        pphr_class: str = "NDPR",
        analysis_type: str = "nom_chk",
        theta: float = 0,
        n: float = 0,
    ) -> Tuple[res.UltimateBendingResults, res.UltimateBendingResults, float]:
        """Calculates the ultimate bending capacity with capacity factors to
        NZS3101:2006 or the NZSEE C5 assessment guidelines dependant on analysis type.

        :param analysis_type: The type of cross section analysis to undertake on the
            defined concrete section, by default a normal nominal strength design check
            is undertaken, refer to :meth:`NZS3101.capacity_reduction_factor` for
            further information on analysis types.
        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \\theta \leq \pi`)
        :param n: Net axial force
        :return: Factored and unfactored ultimate bending results objects, and capacity
            reduction factor *(factored_results, unfactored_results, phi)*
        """
        # Check NZS3101:2006 CL 5.2.1 concrete compressive strength limits
        # (dependant on PPHR class)
        self.check_f_c_limits(pphr_class)

        # Check NZS3101:2006 CL 5.3.3 steel reinforcement yield strength limit
        self.check_f_y_limit()

        # determine strength reduction factor based on analysis type specified
        phi, _, os_design, _ = self.capacity_reduction_factor(analysis_type)

        # determine the section to analyse
        if analysis_type.lower() in ["nom_chk", "cpe_chk"]:
            analysis_section = self.concrete_section
        elif analysis_type.lower() in ["os_chk"]:
            analysis_section = self.create_os_section()
        else:
            analysis_section = self.create_prob_section(os_design)

        # calculate ultimate bending capacity
        ult_res = analysis_section.ultimate_bending_capacity(theta=theta, n=n / phi)

        # factor ultimate results
        f_ult_res = deepcopy(ult_res)
        f_ult_res.n *= phi
        f_ult_res.m_x *= phi
        f_ult_res.m_y *= phi
        f_ult_res.m_xy *= phi

        return f_ult_res, ult_res, phi

    def moment_interaction_diagram(
        self,
        pphr_class: str = "NDPR",
        analysis_type: str = "nom_chk",
        theta: float = 0,
    ) -> Tuple[res.MomentInteractionResults, res.MomentInteractionResults, List[float]]:
        """Generates a moment interaction diagram with capacity factors and material
        strengths to NZS3101:2006 or the NZSEE C5 assessment guidelines dependant
        on analysis type.

        :param pphr_class: Potential Plastic Hinge Region (PPHR) classification,
            **NDPR**/**LDPR**/**DPR**.

            - **NDPR** = Nominally Ductile Plastic Region

            - **LDPR** = Limited Ductile Plastic Region

            - **DPR** = Ductile Plastic Region

        :param analysis_type: The type of cross section analysis to undertake on the
            defined concrete section, by default a normal nominal strength design check
            is undertaken, refer to :meth:`NZS3101.capacity_reduction_factor` for
            further information on analysis types.
        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \\theta \leq \pi`)
        :return: Factored and unfactored moment interaction results objects, and list of
            capacity reduction factors *(factored_results, unfactored_results, phis)*
        """
        # Check NZS3101:2006 CL 5.2.1 concrete compressive strength limits
        # (dependant on PPHR class)
        self.check_f_c_limits(pphr_class)

        # Check NZS3101:2006 CL 5.3.3 steel reinforcement yield strength limit
        self.check_f_y_limit()

        # determine strength reduction factor based on analysis type specified
        phi, cpe_design, os_design, prob_design = self.capacity_reduction_factor(
            analysis_type
        )

        # determine the section to analyse
        if analysis_type.lower() in ["nom_chk", "cpe_chk"]:
            analysis_section = self.concrete_section
        elif analysis_type.lower() in ["os_chk"]:
            analysis_section = self.create_os_section()
        else:
            analysis_section = self.create_prob_section(os_design)

        # determine NZS3101:2006 maximum compression capacity
        max_comp = phi * self.max_comp_strength(
            cpe_design=cpe_design, os_design=os_design, prob_design=prob_design
        )

        # analyse the concrete section to create the M/N interaction curve
        mi_res = analysis_section.moment_interaction_diagram(
            theta=theta,
            control_points=[
                ("kappa0", 0.0),
                ("N", max_comp),
                ("fy", 0),
                ("fy", 0.5),
                ("fy", 1.0),
                ("N", 0.0),
                ("d_n", 1e-6),
            ],
            n_points=[3, 8, 10, 12, 12, 3],
            max_comp=max_comp,
        )

        # make a copy of the results to factor
        f_mi_res = deepcopy(mi_res)

        # list to store phis
        phis = []

        # factor results
        for ult_res in f_mi_res.results:
            ult_res.n *= phi
            ult_res.m_x *= phi
            ult_res.m_y *= phi
            ult_res.m_xy *= phi
            phis.append(phi)

        return f_mi_res, mi_res, phis

    def biaxial_bending_diagram(
        self,
        pphr_class: str = "NDPR",
        analysis_type: str = "nom_chk",
        n: float = 0.0,
        n_points: int = 48,
    ) -> Tuple[res.BiaxialBendingResults, List[float]]:
        """Generates a biaxial bending with capacity factors to NZS3101:2006 or the
        NZSEE C5 assessment guidelines dependant on analysis type.

        :param pphr_class: Potential Plastic Hinge Region (PPHR) classification,
            **NDPR**/**LDPR**/**DPR**.

            - **NDPR** = Nominally Ductile Plastic Region

            - **LDPR** = Limited Ductile Plastic Region

            - **DPR** = Ductile Plastic Region

        :param analysis_type: The type of cross section analysis to undertake on the
            defined concrete section, by default a normal nominal strength design check
            is undertaken, refer to :meth:`NZS3101.capacity_reduction_factor` for
            further information on analysis types.
        :param n: Net axial force
        :param n_points: Number of calculation points for neutral axis orientation
        :return: Factored biaxial bending results object and list of capacity reduction
            factors *(factored_results, phis)*.
        """

        # Check NZS3101:2006 CL 5.2.1 concrete compressive strength limits
        # (dependant on PPHR class)
        self.check_f_c_limits(pphr_class)

        # Check NZS3101:2006 CL 5.3.3 steel reinforcement yield strength limit
        self.check_f_y_limit()

        # initialise results
        f_bb_res = res.BiaxialBendingResults(n=n)

        # list to store phis
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
                    pphr_class,
                    analysis_type,
                    theta=theta,
                    n=n,
                )
                f_bb_res.results.append(f_ult_res)
                phis.append(phi)

                progress.update(task, advance=1)

            # add first result to end of list top
            f_bb_res.results.append(f_bb_res.results[0])
            phis.append(phis[0])

            progress.update(
                task,
                description=(
                    "[bold green]:white_check_mark: Biaxial bending diagram"
                    " generated"
                ),
            )
            live.refresh()

        return f_bb_res, phis
