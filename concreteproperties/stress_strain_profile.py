from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from concreteproperties.post import plotting_context

if TYPE_CHECKING:
    import matplotlib


@dataclass
class StressStrainProfile:
    """Abstract base class for a material stress-strain profile.

    Implements a piecewise linear stress-strain profile. Positive stresses & strains are
    compression.

    :param strains: List of strains (must be increasing or equal)
    :param stresses: List of stresses
    """

    strains: List[float]
    stresses: List[float]

    def __post_init__(
        self,
    ):
        # validate input - same length lists
        if len(self.strains) != len(self.stresses):
            raise ValueError("Length of strains must equal length of stresses")

        # validate input - length > 1
        if len(self.strains) < 2:
            raise ValueError("Length of strains and stresses must be greater than 1")

        # validate input - increasing values
        prev_strain = self.strains[0]

        for idx in range(len(self.strains)):
            if idx != 0:
                if self.strains[idx] < prev_strain:
                    msg = "strains must contain increasing values."
                    raise ValueError(msg)

                prev_strain = self.strains[idx]

    def get_stress(
        self,
        strain: float,
    ) -> float:
        """Returns a stress given a strain.

        :param strain: Strain at which to return a stress.

        :return: Stress
        """

        # create interpolation function
        stress_function = interp1d(
            x=self.strains,
            y=self.stresses,
            kind="linear",
            fill_value="extrapolate",  # type: ignore
        )

        return stress_function(strain)

    def get_elastic_modulus(
        self,
    ) -> float:
        """Returns the elastic modulus of the stress-strain profile.

        :return: Elastic modulus
        """

        small_strain = 1e-6

        # get stress at zero strain
        stress_0 = self.get_stress(strain=0)

        # get stress at small positive strain & compute elastic modulus
        stress_positive = self.get_stress(strain=small_strain)
        em_positive = stress_positive / small_strain

        # get stress at small negative strain & compute elastic modulus
        stress_negative = self.get_stress(strain=-small_strain)
        em_negative = stress_negative / -small_strain

        # check elastic moduli are equal, if not print warning
        if not np.isclose(em_positive, em_negative):
            warnings.warn(
                "Initial compressive and tensile elastic moduli are not equal"
            )

        if np.isclose(em_positive, 0):
            raise ValueError("Elastic modulus is zero.")

        return em_positive

    def get_compressive_strength(
        self,
    ) -> float:
        """Returns the most positive stress.

        :return: Compressive strength
        """

        return max(self.stresses)

    def get_tensile_strength(
        self,
    ) -> float:
        """Returns the most negative stress.

        :return: Tensile strength
        """

        return min(self.stresses)

    def get_yield_strength(
        self,
    ) -> float:
        """Returns the yield strength of the stress-strain profile.

        :return: Yield strength
        """

        raise NotImplementedError

    def get_ultimate_compressive_strain(
        self,
    ) -> float:
        """Returns the largest compressive strain.

        :return: Ultimate strain
        """

        return max(self.strains)

    def get_ultimate_tensile_strain(
        self,
    ) -> float:
        """Returns the largest tensile strain.

        :return: Ultimate strain
        """

        return min(self.strains)

    def get_unique_strains(
        self,
    ) -> List[float]:
        """Returns an ordered list of unique strains.

        :return: Ordered list of unique strains
        """

        unique_strains = list(set(self.strains))
        unique_strains.sort()

        return unique_strains

    def print_properties(
        self,
        fmt: str = "8.6e",
    ):
        """Prints the stress-strain profile properties to the terminal.

        :param fmt: Number format
        """

        table = Table(title=f"Stress-Strain Profile - {type(self).__name__}")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row(
            "Elastic Modulus", "{:>{fmt}}".format(self.get_elastic_modulus(), fmt=fmt)
        )
        table.add_row(
            "Compressive Strength",
            "{:>{fmt}}".format(self.get_compressive_strength(), fmt=fmt),
        )
        table.add_row(
            "Tensile Strength",
            "{:>{fmt}}".format(-self.get_tensile_strength(), fmt=fmt),
        )
        table.add_row(
            "Ultimate Compressive Strain",
            "{:>{fmt}}".format(self.get_ultimate_compressive_strain(), fmt=fmt),
        )
        table.add_row(
            "Ultimate Tensile Strain",
            "{:>{fmt}}".format(self.get_ultimate_tensile_strain(), fmt=fmt),
        )

        console = Console()
        console.print(table)

    def plot_stress_strain(
        self,
        title: str = "Stress-Strain Profile",
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots the stress-strain profile.

        :param title: Plot title
        :param fmt: Plot format string
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        """

        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (
            fig,
            ax,
        ):
            ax.plot(self.strains, self.stresses, fmt)  # type: ignore
            plt.xlabel("Strain")
            plt.ylabel("Stress")
            plt.grid(True)

        return ax


@dataclass
class ConcreteServiceProfile(StressStrainProfile):
    """Abstract class for a concrete service stress-strain profile.

    :param strains: List of strains (must be increasing or equal)
    :param stresses: List of stresses
    :param ultimate_strain: Concrete strain at failure
    """

    strains: List[float]
    stresses: List[float]
    elastic_modulus: float = field(init=False)
    ultimate_strain: float

    def print_properties(
        self,
        fmt: str = "8.6e",
    ):
        """Prints the stress-strain profile properties to the terminal.

        :param fmt: Number format
        """

        table = Table(title=f"Stress-Strain Profile - {type(self).__name__}")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row(
            "Elastic Modulus", "{:>{fmt}}".format(self.get_elastic_modulus(), fmt=fmt)
        )
        table.add_row(
            "Ultimate Compressive Strain",
            "{:>{fmt}}".format(self.get_ultimate_compressive_strain(), fmt=fmt),
        )

        console = Console()
        console.print(table)

    def get_elastic_modulus(
        self,
    ) -> float:
        """Returns the elastic modulus of the stress-strain profile.

        :return: Elastic modulus
        """

        try:
            return self.elastic_modulus
        except AttributeError:
            return super().get_elastic_modulus()

    def get_compressive_strength(
        self,
    ) -> Union[float, None]:
        """Returns the most positive stress.

        :return: Compressive strength
        """

        return None

    def get_tensile_strength(
        self,
    ) -> Union[float, None]:
        """Returns the most negative stress.

        :return: Tensile strength
        """

        return None

    def get_ultimate_compressive_strain(
        self,
    ) -> float:
        """Returns the largest strain.

        :return: Ultimate strain
        """

        return self.ultimate_strain


@dataclass
class ConcreteLinear(ConcreteServiceProfile):
    """Class for a symmetric linear stress-strain profile.

    :param elastic_modulus: Elastic modulus of the stress-strain profile
    :param ultimate_strain: Concrete strain at failure
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    elastic_modulus: float
    ultimate_strain: float = field(default=1)

    def __post_init__(
        self,
    ):
        self.strains = [-0.001, 0, 0.001]
        self.stresses = [-0.001 * self.elastic_modulus, 0, 0.001 * self.elastic_modulus]


@dataclass
class ConcreteLinearNoTension(ConcreteServiceProfile):
    """Class for a linear stress-strain profile with no tensile strength.

    :param elastic_modulus: Elastic modulus of the stress-strain profile
    :param ultimate_strain: Maximum concrete compressive strain at failure
    :param compressive_strength: Compressive strength of the concrete
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    elastic_modulus: float
    ultimate_strain: float = field(default=1)
    compressive_strength: Union[float, None] = field(default=None)

    def __post_init__(
        self,
    ):
        self.strains = [-0.001, 0, 0.001]
        self.stresses = [0, 0, 0.001 * self.elastic_modulus]

        if self.compressive_strength is not None:
            self.strains[-1] = self.compressive_strength / self.elastic_modulus
            self.stresses[-1] = self.compressive_strength
            self.strains.append(self.ultimate_strain)
            self.stresses.append(self.compressive_strength)


@dataclass
class EurocodeNonLinear(ConcreteServiceProfile):
    r"""Class for a non-linear stress-strain relationship to EC2.

    Tension is modelled with a symmetric ``elastic_modulus`` until failure at
    ``tensile_strength``, after which the tensile stress reduces according to the
    ``tension_softening_stiffness``.

    :param elastic_modulus: Concrete elastic modulus (:math:`E_{cm}`)
    :param ultimate_strain: Maximum concrete compressive strain at failure
        (:math:`\varepsilon_{cu1}`)
    :param compressive_strength: Concrete compressive strength (:math:`f_{cm}`)
    :param compressive_strain: Strain at which the maximum concrete strength is reached
        (:math:`\varepsilon_{c1}`)
    :param tensile_strength: Concrete tensile strength
    :param tension_softening_stiffness: Slope of the linear tension softening
        branch
    :param n_points_1: Number of points to discretise the curve prior to the peak stress
    :param n_points_2: Number of points to discretise the curve after the peak stress
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    elastic_modulus: float
    ultimate_strain: float
    compressive_strength: float
    compressive_strain: float
    tensile_strength: float
    tension_softening_stiffness: float
    n_points_1: int = field(default=10)
    n_points_2: int = field(default=3)

    def __post_init__(
        self,
    ):
        self.strains = []
        self.stresses = []

        # tensile portion of curve
        strain_tension_strength = -self.tensile_strength / self.elastic_modulus
        strain_zero_tension = (
            strain_tension_strength
            - self.tensile_strength / self.tension_softening_stiffness
        )

        self.strains.append(1.1 * strain_zero_tension)
        self.stresses.append(0)
        self.strains.append(strain_zero_tension)
        self.stresses.append(0)
        self.strains.append(strain_tension_strength)
        self.stresses.append(-self.tensile_strength)
        self.strains.append(0)
        self.stresses.append(0)

        # constants
        k = (
            1.05
            * self.elastic_modulus
            * self.compressive_strain
            / self.compressive_strength
        )

        # initialise concrete stress and strain
        conc_strain = 0
        conc_stress = 0

        # prior to peak stress
        for idx in range(self.n_points_1):
            conc_strain = self.compressive_strain / self.n_points_1 * (idx + 1)
            eta = conc_strain / self.compressive_strain
            conc_stress = (
                self.compressive_strength * (k * eta - eta * eta) / (1 + eta * (k - 2))
            )

            self.strains.append(conc_strain)
            self.stresses.append(conc_stress)

        # after peak stress
        for idx in range(self.n_points_2):
            remaining_strain = self.ultimate_strain - self.compressive_strain
            conc_strain = (
                self.compressive_strain + remaining_strain / self.n_points_2 * (idx + 1)
            )
            eta = conc_strain / self.compressive_strain
            conc_stress = (
                self.compressive_strength * (k * eta - eta * eta) / (1 + eta * (k - 2))
            )

            self.strains.append(conc_strain)
            self.stresses.append(conc_stress)

        # close off final stress
        self.strains.append(1.01 * conc_strain)
        self.stresses.append(conc_stress)


@dataclass
class ModifiedMander(ConcreteServiceProfile):
    r"""Class for a non-linear stress-strain relationship based on the Mander
    stress-strain model for confined & unconfined concrete for a rectangular cross
    section. Intended for use with moment-curvature analyses with rectangular or
    circular cross sections.

    Refer to references [1]_ [2]_ [3]_ for further information on the Mander
    stress-strain models for confined and unconfined concrete.

    This stress strain relationship has been specifically modified for use as per the
    modified implementation documented within the NZSEE C5 assessment guidelines.
    However input parameters can also be customised to suit other implementations if
    desired.

    .. tip::
      Optional input variables are only required for defining a confined concrete
      stress-strain relationship. Note if any variables are missed when attempting to
      define a confined concrete stress-strain relationship (using
      ``conc_confined=True``), then the material will default to being defined as an
      unconfined concrete stress-strain relationship with a warning given.

    .. admonition:: Modifications to Mander confined concrete model:-

      The original formulation of the expression for confined concrete presented by
      Mander et al. [1]_ can predict high levels of confined concrete strain dependant
      on the assumed value for the ultimate steel strain for the transverse
      reinforcement. The modified expression given the NZSEE C5 assesment guidelines
      [3]_ provides a correction and is directly implemented in the
      :class:`ModifiedMander` material class.

      These corrections to avoid overestimating the confined concrete limiting strain
      consist of three allowances:-

      - Modifying the maximum steel strain by a factor of 0.6:-

        - :math:`\varepsilon_{s,max}= 0.6\varepsilon_{su} \leq 0.06`

        - Note this 0.6 modifier can be altered via the ``n_steel_strain`` parameter.

        - Note the steel material used for reinforcement is also required to be defined
          with this same limiting fracture strain for a moment-curvature analysis.

      - Modifying the volumetric ratio of confinement reinforcement by a factor of
        0.75. i.e.:-

        - For rectangular sections

          - :math:`\displaystyle{\rho_{st}=\frac{0.75}{s}\left[\frac{A_{v,d}}
            {b_{core}}+\frac{A_{v,b}}{d_{core}}\right]}`

        - For circular sections

          - :math:`\displaystyle{\rho_{st}=\frac{0.75}{s}\frac{4A_v}{d_s}}`

        - Note this 0.75 modifier can be altered via the ``n_confinement`` parameter.

      - For confined concrete utilising a maximum concrete compressive strain of:-

        - :math:`\displaystyle{\varepsilon_{c,max}=0.004+\frac{0.6\rho_{st}f_{yh}
          \varepsilon_{su}}{f'_{cc}}\leq0.05}`

        - Note that the 0.6 factor applied to the ultimate tensile failure strain can
          be modified as noted above.

    .. plot:: ./_static/doc_plots/mander_unconfined_plot.py mander_unconfined_plot
      :include-source: False
      :caption: ModifiedMander Parameters for Unconfined Concrete

    .. plot:: ./_static/doc_plots/mander_confined_plot.py mander_confined_plot
      :include-source: False
      :caption: ModifiedMander Parameters for Confined Concrete

    .. [1] Theoretical Stress-Strain Model For Confined Concrete - Mander, Priestley,
      Park (1988)
    .. [2] Observed Stress-Strain Behavior of Confined Concrete - Mander, Priestley,
      Park (1988)
    .. [3] NZSEE C5 Assessment Guidelines - Part C5 - Concrete Buildings - Technical
      Proposal to Revise the Engineering Assessment Guidelines (2018)

    :param elastic_modulus: Concrete elastic modulus (:math:`E_c`)
    :param compressive_strength: Concrete compressive strength (:math:`f'_c`)
    :param tensile_strength: Concrete tensile strength (:math:`f_t`)
    :param sect_type: The type of concrete cross section for which to create a confined
        concrete stress-strain relationship for:-

        - **rect** = Rectangular section with closed stirrup/tie transverse
          reinforcement

        - **circ_hoop** = Circular section with closed hoop transverse reinforcement

        - **circ_spiral** = Circular section with spiral transverse reinforcement

    :param conc_confined: True to return a confined concrete stress-strain relationship
        based on provided reinforcing parameters, False to return an unconfined concrete
        stress-strain relationship
    :param conc_tension: True to include tension in the concrete within the
        stress-strain relationship (up to the tensile strength of the concrete is
        reached), False to not consider any tension behaviour in the concrete
    :param conc_spalling: True to consider the spalling effect for unconfined concrete,
        False to not consider the spalling branch and truncate the unconfined concrete
        curve at min(:math:`2 \varepsilon_{co},\varepsilon_{c,max}`)
    :param eps_co: Strain at which the maximum concrete stress is obtained for an
        unconfined concrete material (:math:`\varepsilon_{co}`)
    :param eps_c_max_unconfined: Maximum strain that is able to be supported within
        unconfined concrete (:math:`\varepsilon_{c,max}`)
    :param eps_sp: Spalling strain, the strain at which the stress returns to zero for
        unconfined concrete (:math:`\varepsilon_{sp}`)
    :param d: Depth of a rectangular concrete cross section, or diameter of circular
        concrete cross section (:math:`d`)
    :param b: Breadth of a rectangular concrete cross section (:math:`b`)
    :param long_reinf_area: Total area of the longitudinal reinforcement in the concrete
        cross section (:math:`A_{st}`)
    :param w_dash: List of clear spacing between longitudinal reinforcement
        around the full perimeter of a rectangular concrete cross section (:math:`w'`)
    :param cvr: Concrete cover (to confining reinforcement)
    :param trans_spacing: Spacing of transverse confining reinforcement (:math:`s`)
    :param trans_d_b: Diameter of the transverse confining reinforcement (:math:`d_b`)
    :param trans_num_d: Number of legs/cross links parallel to the depth of a
        rectangular concrete cross section
    :param trans_num_b: Number of legs/cross links parallel to the breadth of a
        rectangular concrete cross section
    :param trans_f_y: Yield strength of the transverse confining reinforcement
        (:math:`f_{yh}`)
    :param eps_su: Strain at the ultimate tensile strength of the reinforcement
        (:math:`\varepsilon_{su}`)
    :param n_points: Number of points to discretise the compression part of the
        stress-strain curve between :math:`\varepsilon_{c}=0` & :math:`\varepsilon_{c}
        =2\varepsilon_{co}` for an unconfined concrete, or between
        :math:`\varepsilon_{c}=0` & :math:`\varepsilon_{c}=\varepsilon_{cu}` for a
        confined concrete
    :param n_steel_strain: Modifier for maximum steel reinforcement strain. Steel
        reinforcement material within the concrete cross section should also be defined
        with the same limit for the fracture strain
    :param n_confinement: Modifier for volumetric ratio of confinement reinforcement
    :raises ValueError: If specified section type is not rect, circ_hoop or circ_spiral
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    elastic_modulus: float
    ultimate_strain: float = field(init=False, default=0)
    compressive_strength: float
    tensile_strength: float
    sect_type: Optional[str] = None
    conc_confined: bool = False
    conc_tension: bool = False
    conc_spalling: bool = False
    eps_co: float = 0.002
    eps_c_max_unconfined: float = 0.004
    eps_sp: float = 0.006
    d: Optional[float] = None
    b: Optional[float] = None
    long_reinf_area: Optional[float] = None
    w_dash: Optional[List[float]] = None
    cvr: Optional[float] = None
    trans_spacing: Optional[float] = None
    trans_d_b: Optional[float] = None
    trans_num_d: Optional[int] = None
    trans_num_b: Optional[int] = None
    trans_f_y: Optional[float] = None
    eps_su: Optional[float] = None
    n_points: int = field(default=50)
    n_steel_strain: float = 0.6
    n_confinement: float = 0.75

    def __post_init__(
        self,
    ):
        self.strains = []
        self.stresses = []

        # check section type is valid
        if self.conc_confined and str(self.sect_type).lower() not in [
            "rect",
            "circ_hoop",
            "circ_spiral",
        ]:
            raise ValueError(
                f"The specified section type '{str(self.sect_type).lower()}' should be "
                f"'rect', 'circ_hoop' or 'circ_spiral'."
            )

        if self.conc_confined and self.sect_type in ["circ_hoop", "circ_spiral"]:
            self.b = 0
            self.w_dash = [0]
            self.trans_num_b = 0
            self.trans_num_d = 0

        # if confined concrete required, check that all inputs have been provided,
        # otherwise reset to unconfined stress-strain relationship and provide warning
        input_not_provided = [
            i
            for i in self.__dataclass_fields__.keys()
            if self.__getattribute__(i) is None
        ]
        if self.conc_confined and input_not_provided:
            self.conc_confined = False
            warnings.warn(
                f"Reverting analysis to utilise an unconfined concrete Mander "
                f"stress-strain model, as the following input variables required for a "
                f"confined concrete Mander stress-strain model have not been "
                f"provided:-\n{input_not_provided}"
            )

        # calculate confined/unconfined compressive strength
        if self.conc_confined:
            # calculate clear distance between transverse reinforcement
            s_dash = self.trans_spacing - self.trans_d_b

            if self.sect_type.lower() in ["rect"]:
                # calculate core dimensions (between centrelines of confining transverse
                # reinforcement)
                d_core = self.d - 2 * self.cvr - self.trans_d_b
                b_core = self.b - 2 * self.cvr - self.trans_d_b

                # calculate core area
                A_c = d_core * b_core

                # calculate area of transverse reinforcement in each direction within a depth s
                A_vd = self.trans_num_d * self.trans_d_b**2 * np.pi / 4
                A_vb = self.trans_num_b * self.trans_d_b**2 * np.pi / 4

                # calculate volumetric ratio of confinement reinforcement
                rho_st = (
                    self.n_confinement
                    / self.trans_spacing
                    * (A_vd / b_core + A_vb / d_core)
                )

                # calculate ratio of reinforcement area to core area
                rho_cc = self.long_reinf_area / A_c

                # calculate plan area of ineffectually confined core concrete at the level of
                # the transverse reinforcement
                A_i = 0
                for w in self.w_dash:
                    A_i = A_i + pow(w, 2)

                # calculate confinement effectiveness coefficient
                k_e = (
                    (1 - A_i / (6 * A_c))
                    * (1 - s_dash / (2 * b_core))
                    * (1 - s_dash / (2 * d_core))
                    / (1 - rho_cc)
                )

                # calculate tranverse reinforcement ratios and confining pressures
                # across defined depth
                rho_d = A_vd / (self.trans_spacing * b_core)
                f_ld = k_e * rho_d * self.trans_f_y

                # calculate tranverse reinforcement ratios and confining pressures
                # across defined width
                rho_b = A_vb / (self.trans_spacing * d_core)
                f_lb = k_e * rho_b * self.trans_f_y

                # calculate confined concrete strength
                f_cc = self.compressive_strength * (
                    -1.254
                    + 2.254
                    * (1 + 7.94 * min(f_lb, f_ld) / self.compressive_strength) ** 0.5
                    - 2 * min(f_lb, f_ld) / self.compressive_strength
                )
            else:
                # calculate core diameter
                d_s = self.d - 2 * self.cvr - self.trans_d_b

                # calculate core area
                A_c = d_s**2 * np.pi / 4

                # calculate volumetric ratio of confinement reinforcement
                rho_st = (
                    self.n_confinement
                    / self.trans_spacing
                    * (4 * self.trans_d_b**2 * np.pi / 4 / d_s)
                )
                # calculate ratio of reinforcement area to core area
                rho_cc = self.long_reinf_area / A_c

                # calculate confinement effectiveness coefficient
                exp = 2 if self.sect_type in ["circ_hoop"] else 1
                k_e = (1 - s_dash / (2 * d_s)) ** exp / (1 - rho_cc)

                # calculate tranverse confining pressures
                # rho_b = A_vb / (self.trans_spacing * d_core)
                f_l = k_e * rho_st * self.trans_f_y

                # calculate confined concrete strength
                f_cc = self.compressive_strength * (
                    -1.254
                    + 2.254 * (1 + 7.94 * f_l / self.compressive_strength) ** 0.5
                    - 2 * f_l / self.compressive_strength
                )
        else:
            # calculate unconfined concrete strength
            f_cc = self.compressive_strength

        # calculate strain associated with max confined/unconfined concrete strength
        eps_cc = self.eps_co * (1 + 5 * (f_cc / self.compressive_strength - 1))

        # calculate maximum confined/unconfined compressive strain
        if self.conc_confined:
            eps_c_max = min(
                0.004
                + self.n_steel_strain * rho_st * self.trans_f_y * self.eps_su / f_cc,
                0.05,
            )
        else:
            eps_c_max = self.eps_c_max_unconfined

        # calculate secant modulus
        E_sec = f_cc / eps_cc

        if self.conc_confined:
            self.strains = np.linspace(0, eps_c_max, self.n_points)
            # add eps_cc point corresponding to max stress point at end
            self.strains = np.append(self.strains, eps_cc)
        else:
            self.strains = np.linspace(0, min(2 * eps_cc, eps_c_max), self.n_points)
            # add eps_cc point corresponding to max stress point at end
            self.strains = np.append(self.strains, eps_cc)

        # sort strains numerically
        self.strains.sort()

        # calculate stresses from strains & convert to List
        r = self.elastic_modulus / (self.elastic_modulus - E_sec)
        x = self.strains / eps_cc
        self.strains = self.strains.tolist()
        self.stresses = (f_cc * x * r / (r - 1 + x**r)).tolist()

        # add spalling branch if specified for unconfined curve
        if not self.conc_confined and self.conc_spalling:
            self.strains.append(self.eps_sp)
            self.stresses.append(0)

        # calculate max tension strain based on modulus of rupture/concrete tension
        # strength
        eps_t = self.tensile_strength / self.elastic_modulus

        if self.conc_tension:
            # add tension stress/strain limit
            self.strains.insert(0, -eps_t)
            self.stresses.insert(0, -self.tensile_strength)
            self.strains.insert(0, self.strains[0])
            self.stresses.insert(0, 0)
            self.strains.insert(0, 2 * self.strains[0])
            self.stresses.insert(0, 0)
        else:
            # add flat horizontal tension stress/strain branch
            self.strains.insert(0, -eps_t)
            self.stresses.insert(0, 0)

        # initiate ultimate compressive strain as maximum strain
        self.ultimate_strain = max(self.strains)

        # add small horizontal compressive strain to improve interpolation
        self.strains.append(self.strains[-1] + 1e-12)
        self.stresses.append(self.stresses[-1])


@dataclass
class ConcreteUltimateProfile(StressStrainProfile):
    """Abstract class for a concrete ultimate stress-strain profile.

    :param strains: List of strains (must be increasing or equal)
    :param stresses: List of stresses
    :param compressive_strength: Concrete compressive strength
    """

    strains: List[float]
    stresses: List[float]
    compressive_strength: float

    def get_compressive_strength(
        self,
    ) -> float:
        """Returns the most positive stress.

        :return: Compressive strength
        """

        return self.compressive_strength

    def get_ultimate_compressive_strain(
        self,
    ) -> float:
        """Returns the ultimate strain, or largest compressive strain.

        :return: Ultimate strain
        """

        try:
            return self.ultimate_strain  # type: ignore
        except AttributeError:
            return super().get_ultimate_compressive_strain()

    def print_properties(
        self,
        fmt: str = "8.6e",
    ):
        """Prints the stress-strain profile properties to the terminal.

        :param fmt: Number format
        """

        table = Table(title=f"Stress-Strain Profile - {type(self).__name__}")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row(
            "Compressive Strength",
            "{:>{fmt}}".format(self.get_compressive_strength(), fmt=fmt),
        )
        table.add_row(
            "Ultimate Compressive Strain",
            "{:>{fmt}}".format(self.get_ultimate_compressive_strain(), fmt=fmt),
        )
        console = Console()
        console.print(table)


@dataclass
class RectangularStressBlock(ConcreteUltimateProfile):
    """Class for a rectangular stress block.

    :param compressive_strength: Concrete compressive strength
    :param alpha: Factor that modifies the concrete compressive strength
    :param gamma: Factor that modifies the depth of the stress block
    :param ultimate_strain: Maximum concrete compressive strain at failure
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    compressive_strength: float
    alpha: float
    gamma: float
    ultimate_strain: float

    def __post_init__(
        self,
    ):
        self.strains = [
            0,
            self.ultimate_strain * (1 - self.gamma),
            self.ultimate_strain * (1 - self.gamma),
            self.ultimate_strain,
        ]
        self.stresses = [
            0,
            0,
            self.alpha * self.compressive_strength,
            self.alpha * self.compressive_strength,
        ]

    def get_stress(
        self,
        strain: float,
    ) -> float:
        """Returns a stress given a strain.

        Overrides parent method with small tolerance to aid ultimate stress generation
        at nodes.

        :param strain: Strain at which to return a stress.

        :return: Stress
        """

        if strain >= self.strains[1] - 1e-8:
            return self.stresses[2]
        else:
            return 0


@dataclass
class BilinearStressStrain(ConcreteUltimateProfile):
    """Class for an ultimate bilinear stress-strain relationship.

    :param compressive_strength: Concrete compressive strength
    :param compressive_strain: Strain at which the maximum concrete strength is reached
    :param ultimate_strain: Maximum concrete compressive strain at failure
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    compressive_strength: float
    compressive_strain: float
    ultimate_strain: float

    def __post_init__(
        self,
    ):
        self.strains = [
            -self.compressive_strain,
            0,
            self.compressive_strain,
            self.ultimate_strain,
        ]
        self.stresses = [
            0,
            0,
            self.compressive_strength,
            self.compressive_strength,
        ]


@dataclass
class ParabolicStressStrain(ConcreteUltimateProfile):
    r"""Class for an ultimate parabolic stress-strain relationship.

    Note parabolic portion of the stress-strain relationship is based on the EC2
    derivation, refer to
    :class:`~concreteproperties.stress_strain_profile.EurocodeParabolicUltimate` class
    for further details.

    :param compressive_strength: Concrete compressive strength (:math:`f'_c`)
    :param compressive_strain: Strain at which the maximum concrete strength is reached
        (:math:`\varepsilon_{1}`)
    :param ultimate_strain: Maximum concrete compressive strain at failure
        (:math:`\varepsilon_{u1}`)
    :param n_exp: Parabolic curve exponent
    :param n_points: Number of points to discretise the parabolic segment of the curve

    .. plot:: ./_static/doc_plots/generic_parabolic_ultimate_plot.py
      generic_parabolic_ultimate_plot
      :include-source: False
      :caption: ParabolicStressStrain Parameters
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    compressive_strength: float
    compressive_strain: float
    ultimate_strain: float
    n_exp: float
    n_points: int = field(default=10)

    def __post_init__(
        self,
    ):
        self.strains = []
        self.stresses = []

        # tensile portion of curve
        self.strains.append(-self.compressive_strain)
        self.stresses.append(0)
        self.strains.append(0)
        self.stresses.append(0)

        # parabolic portion of curve
        for idx in range(self.n_points):
            conc_strain = self.compressive_strain / self.n_points * (idx + 1)
            conc_stress = self.compressive_strength * (
                1 - np.power(1 - (conc_strain / self.compressive_strain), self.n_exp)
            )

            self.strains.append(conc_strain)
            self.stresses.append(conc_stress)

        # compressive plateau
        self.strains.append(self.ultimate_strain)
        self.stresses.append(self.compressive_strength)


@dataclass
class EurocodeBilinearUltimate(ConcreteUltimateProfile):
    r"""Class for an ultimate bilinear stress-strain relationship to EC2.

    The stress-strain relationship is defined as follows:-

    With the design concrete compressive strength (:math:`f_{cd}`) being defined in EC2
    CL 3.1.6(1)P as:-

    :math:`\quad f_{cd}=\displaystyle{\frac{\alpha_{cc}f_{ck}}{\gamma_C}}`

    And the stress & strain relationship being defined in EC2 CL 3.1.7(2) as:-

    :math:`\quad\sigma_c=f_{cd}\displaystyle{\bigg[\frac{\varepsilon_c}
    {\varepsilon_{c3}}\bigg]}` for :math:`0\leq\varepsilon_c\leq\varepsilon_{c3}`

    :math:`\quad\sigma_c=f_{cd}` for :math:`\varepsilon_{c3}\leq\varepsilon_c\leq
    \varepsilon_{cu3}`


    .. note::
      The default recommended values for the stress-strain relationship parameters
      :math:`\alpha_{cc}` & :math:`\gamma_C` are taken as per EC2. Note that by default
      the design situation for the default value of :math:`\gamma_C` assumes a
      'Persistent & Transient' load case verses an 'Accidental' load case.

      However, note that a Countries National Annex may modify the value of these
      parameters. If this is the case then provide the values for the variable(s) to
      override the default EC2 value(s) as required.

    .. tip::
      Note, if utilising the
      :class:`~concreteproperties.stress_strain_profile.EurocodeBilinearUltimate`
      stress-strain profile with design codes (other than EC2) which might utilise a
      strength reduction factor (:math:`\phi`) based approach verses a partial factor of
      safety (:math:`\gamma`) based approach, then a
      :class:`~concreteproperties.stress_strain_profile.EurocodeBilinearUltimate`
      stress-strain profile can be adopted provided that consideration is given to the
      following:-

      - The :math:`\gamma_C` factor should generally be taken as being 1.0, and the
        strength reduction factor should be applied to the design ultimate strength or
        material properties in the normal manner associated with the design code.

      - The concrete compressive strength should have an appropriate reduction factor
        applied to it which is consistent with the design code via the
        :math:`\alpha_{cc}` factor.

    .. plot:: ./_static/doc_plots/ec2_bilinear_ultimate_plot.py
      ec2_bilinear_ultimate_plot
      :include-source: False
      :caption: EurocodeBilinearParabolicUltimate Parameters

    :param compressive_strength: Characteristic concrete compressive cylinder strength
        (:math:`f_{ck}`)
    :param limiting_strain: Upper limit on the strain considered in the generated
        stress-strain profile. This variable should be utilised when using the EC2
        bilinear stress-strain relationship with other design codes that have a
        lower limit on the ultimate compressive strain than the maximum compressive
        strain of 0.0035 considered in EC2. In this case the EC2 curve is truncated at
        the limiting strain specified if applicable
    :param alpha_cc: Coefficient to take account of long term effects on the compressive
        strength and of unfavourable effects resulting for the way the load is applied
        (:math:`\alpha_{cc}`)

        - The recommended value in EC2 CL 3.1.6(1)P is 1.0. However, the value of
          :math:`\alpha_{cc}` for use in a Country should lie between 0.8 and 1.0 and
          may be found in that Countries National Annex

    :param gamma_c: Partial factor of safety for concrete (:math:`\gamma_C`) in
        accordance with EC2 CL 2.4.2.4 or a Countries National Annex

        - The recommended values from EC2 CL 2.4.2.4 are as outlined below. However, the
          value of :math:`\gamma_C` may be found in a Countries National Annex

        +------------------------+---------------------------------------+
        | Design Situations      | Partial Factor Of Safety for Concrete |
        +========================+=======================================+
        | Persistent & Transient | 1.5                                   |
        +------------------------+---------------------------------------+
        | Accidental             | 1.2                                   |
        +------------------------+---------------------------------------+

        - Note that the 'Persistent & Transient' value is utilised by default unless a
          user defined value is provided
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    compressive_strength: float
    ultimate_strain: float = field(init=False, default=0)
    limiting_strain: float = 0.0035
    alpha_cc: float = 1.0
    gamma_c: float = 1.5

    def __post_init__(
        self,
    ):
        self.strains = []
        self.stresses = []

        # determine transition and ultimate compressive strains
        epsilon_c3, epsilon_cu3 = self.epsilon_bilinear(
            self.compressive_strength, self.limiting_strain
        )

        # determine design concrete compressive strength
        f_cd = self.alpha_cc * self.compressive_strength / self.gamma_c

        # tensile portion of curve
        self.strains.append(-epsilon_c3)
        self.stresses.append(0)
        self.strains.append(0)
        self.stresses.append(0)

        # linear sloping portion of curve
        self.strains.append(epsilon_c3)
        self.stresses.append(f_cd)

        # compressive plateau
        self.strains.append(epsilon_cu3)
        self.stresses.append(f_cd)

        # initiate ultimate compressive strain as maximum strain
        self.ultimate_strain = max(self.strains)

        # add small horizontal compressive strain to improve interpolation
        self.strains.append(self.strains[-1] + 1e-12)
        self.stresses.append(self.stresses[-1])

    def epsilon_bilinear(
        self, f_ck: float, limiting_strain: float = 0.0035
    ) -> Tuple[float, float]:
        r"""Function to calculate the strain at which the maximum concrete strength is
        reached (:math:`\epsilon_{c3}`) and the ultimate compressive strain
        (:math:`\epsilon_{cu3}`) for the EC2 bilinear stress-strain relationship. Refer
        to EC2 Table 3.1.

        :param f_ck: Characteristic concrete compressive cylinder strength
            (:math:`f_{ck}`)
        :param limiting_strain: Upper limit on the strain considered in the generated
            stress-strain profile. This variable should be utilised when using the EC2
            parabolic stress-strain relationship with other design codes that have a
            lower limit on the ultimate compressive strain than the maximum compressive
            strain of 0.0035 considered in EC2. In this case the returned strains are
            limited to the limiting strain specified if applicable

        :raises ValueError: If concrete strength provided is outside the bounds of 12
            MPa and 90 MPa
        :raises ValueError: If the specified limiting strain is greater than that
            considered by the EC2 design code
        :return: Strain at which maximum concrete strength is reached
            (:math:`\epsilon_{c3}`) & the ultimate compressive strain
            (:math:`\epsilon_{cu3}`)
        """

        # Check if user defined limiting strain is greater than maximum ultimate
        # compressive strain from EC2
        if limiting_strain > 0.0035:
            raise ValueError(
                f"The limiting strain of {limiting_strain} provided is greater "
                f" than the 0.0035 limiting ultimate strain from EC2"
            )

        if 12 <= f_ck <= 90:
            # EC2 table 3.1, epsilon_c2 & epsilon_cu2
            if f_ck < 50:
                epsilon_c = 1.75 / 1000
                epsilon_cu = 3.5 / 1000
            else:
                epsilon_c = (1.75 + 0.55 * ((f_ck - 50) / 40)) / 1000
                epsilon_cu = (2.6 + 35 * ((90 - f_ck) / 100) ** 4) / 1000
        else:
            raise ValueError(
                f"Concrete compressive strength should be between 12 MPa & 90 MPa for "
                f"EC2, a compressive strength of {f_ck:.2f} MPa was provided."
            )

        # TODO check this
        # around a concrete strength of 90MPa, epsilon_c3 creeps over epsilon_cu3 by a
        # very small margin, effectively there is no constant stress plateau at 90MPa
        # concrete strength
        if epsilon_c > epsilon_cu:
            epsilon_c = epsilon_cu

        # limit strain to an upper bound based on the defined limiting strain.
        if epsilon_c > limiting_strain:
            epsilon_c = limiting_strain
        if epsilon_cu > limiting_strain:
            epsilon_cu = limiting_strain

        return epsilon_c, epsilon_cu


@dataclass
class EurocodeParabolicUltimate(ConcreteUltimateProfile):
    r"""Class for an ultimate parabolic stress-strain relationship to EC2.

    The stress-strain relationship is defined as follows:-

    With the design concrete compressive strength (:math:`f_{cd}`) being defined in EC2
    CL 3.1.6(1)P as:-

    :math:`\quad f_{cd}=\displaystyle{\frac{\alpha_{cc}f_{ck}}{\gamma_C}}`

    And the stress & strain relationship being defined in EC2 CL 3.1.7(1) as:-

    :math:`\quad\sigma_c=f_{cd}\displaystyle{\bigg[1-\Big(1-\frac{\varepsilon_c}
    {\varepsilon_{c2}}\Big)^n\bigg]}` for :math:`0\leq\varepsilon_c\leq\varepsilon_{c2}`

    :math:`\quad\sigma_c=f_{cd}` for :math:`\varepsilon_{c2}\leq\varepsilon_c\leq
    \varepsilon_{cu2}`

    .. note::
      The default recommended values for the stress-strain relationship parameters
      :math:`\alpha_{cc}` & :math:`\gamma_C` are taken as per EC2. Note that by default
      the design situation for the default value of :math:`\gamma_C` assumes a
      'Persistent & Transient' load case verses an 'Accidental' load case.

      However, note that a Countries National Annex may modify the value of these
      parameters. If this is the case then provide the values for the variable(s) to
      override the default EC2 value(s) as required.

    .. tip::
      Note, if utilising the
      :class:`~concreteproperties.stress_strain_profile.EurocodeParabolicUltimate`
      stress-strain profile with design codes (other than EC2) which might utilise a
      strength reduction factor (:math:`\phi`) based approach verses a partial factor of
      safety (:math:`\gamma`) based approach, then a
      :class:`~concreteproperties.stress_strain_profile.EurocodeParabolicUltimate`
      stress-strain profile can be adopted provided that consideration is given to the
      following:-

      - The :math:`\gamma_C` factor should generally be taken as being 1.0, and the
        strength reduction factor should be applied to the design ultimate strength or
        material properties in the normal manner associated with the design code.

      - The concrete compressive strength should have an appropriate reduction factor
        applied to it which is consistent with the design code via the
        :math:`\alpha_{cc}` factor.

    .. plot:: ./_static/doc_plots/ec2_parabolic_ultimate_plot.py
      ec2_parabolic_ultimate_plot
      :include-source: False
      :caption: EurocodeParabolicUltimate Parameters

    :param compressive_strength: Characteristic concrete compressive cylinder strength
        (:math:`f_{ck}`)
    :param limiting_strain: Upper limit on the strain considered in the generated
        stress-strain profile. This variable should be utilised when using the EC2
        parabolic stress-strain relationship with other design codes that have a
        lower limit on the ultimate compressive strain than the maximum compressive
        strain of 0.0035 considered in EC2. In this case the EC2 curve is truncated at
        the limiting strain specified if applicable
    :param alpha_cc: Coefficient to take account of long term effects on the compressive
        strength and of unfavourable effects resulting for the way the load is applied
        (:math:`\alpha_{cc}`)

        - The recommended value in EC2 CL 3.1.6(1)P is 1.0. However, the value of
          :math:`\alpha_{cc}` for use in a Country should lie between 0.8 and 1.0 and
          may be found in that Countries National Annex

    :param gamma_c: Partial factor of safety for concrete (:math:`\gamma_C`) in
        accordance with EC2 CL 2.4.2.4 or a Countries National Annex

        - The recommended values from EC2 CL 2.4.2.4 are as outlined below. However, the
          value of :math:`\gamma_C` may be found in a Countries National Annex

        +------------------------+---------------------------------------+
        | Design Situations      | Partial Factor Of Safety for Concrete |
        +========================+=======================================+
        | Persistent & Transient | 1.5                                   |
        +------------------------+---------------------------------------+
        | Accidental             | 1.2                                   |
        +------------------------+---------------------------------------+

        - Note that the 'Persistent & Transient' value is utilised by default unless a
          user defined value is provided

    :param n_points: Number of points to discretise the parabolic segment of the curve
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    compressive_strength: float
    ultimate_strain: float = field(init=False, default=0)
    limiting_strain: float = 0.0035
    alpha_cc: float = 1.0
    gamma_c: float = 1.5
    n_points: int = field(default=10)

    def __post_init__(
        self,
    ):
        self.strains = []
        self.stresses = []

        # determine n exponent
        n_exp = self.n_exp(self.compressive_strength)

        # determine transition and ultimate compressive strains
        epsilon_c2, epsilon_cu2 = self.epsilon_parabolic(
            self.compressive_strength, self.limiting_strain
        )

        # determine design concrete compressive strength
        f_cd = self.alpha_cc * self.compressive_strength / self.gamma_c

        # tensile portion of curve
        self.strains.append(-epsilon_c2)
        self.stresses.append(0)
        self.strains.append(0)
        self.stresses.append(0)

        # parabolic portion of curve
        for idx in range(self.n_points):
            conc_strain = epsilon_c2 / self.n_points * (idx + 1)
            conc_stress = f_cd * (1 - np.power(1 - (conc_strain / epsilon_c2), n_exp))

            self.strains.append(conc_strain)
            self.stresses.append(conc_stress)

        # compressive plateau
        self.strains.append(epsilon_cu2)
        self.stresses.append(f_cd)

        # initiate ultimate compressive strain as maximum strain
        self.ultimate_strain = max(self.strains)

        # add small horizontal compressive strain to improve interpolation
        self.strains.append(self.strains[-1] + 1e-12)
        self.stresses.append(self.stresses[-1])

    def n_exp(self, f_ck: float) -> float:
        """Function to calculate the 'n' exponent in EC2 stress block equation, refer to
        EC2 Table 3.1.

        :param f_ck: Characteristic concrete compressive cylinder strength
            (:math:`f_{ck}`)
        :raises ValueError: If concrete strength provided is outside the EC2 bounds of
            12 MPa and 90 MPa
        :return: :math:`n` exponential
        """

        if 12 <= f_ck <= 90:
            n_exp = 2 if f_ck < 50 else 1.4 + 23.4 * ((90 - f_ck) / 100) ** 4
        else:
            raise ValueError(
                f"Concrete compressive strength should be between 12 MPa & 90 MPa for "
                f"EC2, a compressive strength of {f_ck:.2f} MPa was provided."
            )

        return n_exp

    def epsilon_parabolic(
        self, f_ck: float, limiting_strain: float = 0.0035
    ) -> Tuple[float, float]:
        r"""Function to calculate the strain at which the maximum concrete strength is
        reached (:math:`\epsilon_{c2}`) and the ultimate compressive strain
        (:math:`\epsilon_{cu2}`) for the EC2 parabolic stress-strain relationship. Refer
        to EC2 Table 3.1.

        :param f_ck: Characteristic concrete compressive cylinder strength
            (:math:`f_{ck}`)
        :param limiting_strain: Upper limit on the strain considered in the generated
            stress-strain profile. This variable should be utilised when using the EC2
            parabolic stress-strain relationship with other design codes that have a
            lower limit on the ultimate compressive strain than the maximum compressive
            strain of 0.0035 considered in EC2. In this case the returned strains are
            limited to the limiting strain specified if applicable

        :raises ValueError: If concrete strength provided is outside the bounds of 12
            MPa and 90 MPa
        :raises ValueError: If the specified limiting strain is greater than that
            considered by the EC2 design code
        :return: Strain at which maximum concrete strength is reached
            (:math:`\epsilon_{c2}`) & the ultimate compressive strain
            (:math:`\epsilon_{cu2}`)
        """

        # Check if user defined limiting strain is greater than maximum ultimate
        # compressive strain from EC2
        if limiting_strain > 0.0035:
            raise ValueError(
                f"The limiting strain of {limiting_strain} provided is greater "
                f" than the 0.0035 limiting ultimate strain from EC2"
            )

        if 12 <= f_ck <= 90:
            # EC2 table 3.1, epsilon_c2 & epsilon_cu2
            if f_ck < 50:
                epsilon_c = 2 / 1000
                epsilon_cu = 3.5 / 1000
            else:
                epsilon_c = (2 + 0.085 * (f_ck - 50) ** 0.53) / 1000
                epsilon_cu = (2.6 + 35 * ((90 - f_ck) / 100) ** 4) / 1000
        else:
            raise ValueError(
                f"Concrete compressive strength should be between 12 MPa & 90 MPa for "
                f"EC2, a compressive strength of {f_ck:.2f} MPa was provided."
            )

        # around a concrete strength of 90MPa, epsilon_c2 creeps over epsilon_cu2 by a
        # very small margin, effectively there is no constant stress plateau at 90MPa
        # concrete strength
        if epsilon_c > epsilon_cu:
            epsilon_c = epsilon_cu

        # limit strain to an upper bound based on the defined limiting strain.
        if epsilon_c > limiting_strain:
            epsilon_c = limiting_strain
        if epsilon_cu > limiting_strain:
            epsilon_cu = limiting_strain

        return epsilon_c, epsilon_cu


@dataclass
class SteelProfile(StressStrainProfile):
    """Abstract class for a steel stress-strain profile.

    :param strains: List of strains (must be increasing or equal)
    :param stresses: List of stresses
    :param yield_strength: Steel yield strength
    :param elastic_modulus: Steel elastic modulus
    :param fracture_strain: Steel fracture strain
    """

    strains: List[float]
    stresses: List[float]
    yield_strength: float
    elastic_modulus: float
    fracture_strain: float

    def get_elastic_modulus(
        self,
    ) -> float:
        """Returns the elastic modulus of the stress-strain profile.

        :return: Elastic modulus
        """

        return self.elastic_modulus

    def get_yield_strength(
        self,
    ) -> float:
        """Returns the yield strength of the stress-strain profile.

        :return: Yield strength
        """

        return self.yield_strength

    def print_properties(
        self,
        fmt: str = "8.6e",
    ):
        """Prints the stress-strain profile properties to the terminal.

        :param fmt: Number format
        """

        table = Table(title=f"Stress-Strain Profile - {type(self).__name__}")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row(
            "Elastic Modulus", "{:>{fmt}}".format(self.get_elastic_modulus(), fmt=fmt)
        )
        table.add_row(
            "Yield Strength", "{:>{fmt}}".format(self.yield_strength, fmt=fmt)
        )
        table.add_row(
            "Tensile Strength",
            "{:>{fmt}}".format(-self.get_tensile_strength(), fmt=fmt),
        )
        table.add_row(
            "Fracture Strain",
            "{:>{fmt}}".format(self.get_ultimate_tensile_strain(), fmt=fmt),
        )

        console = Console()
        console.print(table)


@dataclass
class SteelElasticPlastic(SteelProfile):
    """Class for a perfectly elastic-plastic steel stress-strain profile.

    :param yield_strength: Steel yield strength
    :param elastic_modulus: Steel elastic modulus
    :param fracture_strain: Steel fracture strain
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    yield_strength: float
    elastic_modulus: float
    fracture_strain: float

    def __post_init__(
        self,
    ):
        yield_strain = self.yield_strength / self.elastic_modulus
        self.strains = [
            -self.fracture_strain,
            -yield_strain,
            0,
            yield_strain,
            self.fracture_strain,
        ]
        self.stresses = [
            -self.yield_strength,
            -self.yield_strength,
            0,
            self.yield_strength,
            self.yield_strength,
        ]


@dataclass
class SteelHardening(SteelProfile):
    """Class for a steel stress-strain profile with strain hardening.

    :param yield_strength: Steel yield strength
    :param elastic_modulus: Steel elastic modulus
    :param fracture_strain: Steel fracture strain
    :param ultimate_strength: Steel ultimate strength
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    yield_strength: float
    elastic_modulus: float
    fracture_strain: float
    ultimate_strength: float

    def __post_init__(
        self,
    ):
        yield_strain = self.yield_strength / self.elastic_modulus
        self.strains = [
            -self.fracture_strain,
            -yield_strain,
            0,
            yield_strain,
            self.fracture_strain,
        ]
        self.stresses = [
            -self.ultimate_strength,
            -self.yield_strength,
            0,
            self.yield_strength,
            self.ultimate_strength,
        ]


@dataclass
class StrandProfile(StressStrainProfile):
    """Abstract class for a steel strand stress-strain profile.

    Implements a piecewise linear stress-strain profile. Positive stresses & strains are
    compression.

    :param strains: List of strains (must be increasing or equal)
    :param stresses: List of stresses
    :param yield_strength: Strand yield strength
    """

    strains: List[float]
    stresses: List[float]
    yield_strength: float

    def __post_init__(self) -> None:
        return super().__post_init__()

    def get_strain(
        self,
        stress: float,
    ) -> float:
        """Returns a strain given a stress.

        :param stress: Stress at which to return a strain.

        :return: Strain
        """

        # create interpolation function
        strain_function = interp1d(
            x=self.stresses,
            y=self.strains,
            kind="linear",
            fill_value="extrapolate",  # type: ignore
        )

        return strain_function(stress)

    def get_yield_strength(self) -> float:
        """Returns the yield strength of the stress-strain profile.

        :return: Yield strength
        """

        return self.yield_strength

    def print_properties(
        self,
        fmt: str = "8.6e",
    ) -> None:
        """Prints the stress-strain profile properties to the terminal.

        :param fmt: Number format
        """

        table = Table(title=f"Stress-Strain Profile - {type(self).__name__}")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row(
            "Elastic Modulus", "{:>{fmt}}".format(self.get_elastic_modulus(), fmt=fmt)
        )
        table.add_row(
            "Yield Strength", "{:>{fmt}}".format(self.yield_strength, fmt=fmt)
        )
        table.add_row(
            "Breaking Strength",
            "{:>{fmt}}".format(-self.get_tensile_strength(), fmt=fmt),
        )
        table.add_row(
            "Fracture Strain",
            "{:>{fmt}}".format(self.get_ultimate_tensile_strain(), fmt=fmt),
        )

        console = Console()
        console.print(table)


@dataclass
class StrandHardening(StrandProfile):
    """Class for a strand stress-strain profile with strain hardening.

    :param yield_strength: Strand yield strength
    :param elastic_modulus: Strand elastic modulus
    :param fracture_strain: Strand fracture strain
    :param breaking_strength: Strand breaking strength
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    yield_strength: float
    elastic_modulus: float
    fracture_strain: float
    breaking_strength: float

    def __post_init__(self) -> None:
        yield_strain = self.yield_strength / self.elastic_modulus
        self.strains = [
            -self.fracture_strain,
            -yield_strain,
            0,
            yield_strain,
            self.fracture_strain,
        ]
        self.stresses = [
            -self.breaking_strength,
            -self.yield_strength,
            0,
            self.yield_strength,
            self.breaking_strength,
        ]

        return super().__post_init__()

    def get_elastic_modulus(
        self,
    ) -> float:
        """Returns the elastic modulus of the stress-strain profile.

        :return: Elastic modulus
        """

        return self.elastic_modulus


@dataclass
class StrandPCI1992(StrandProfile):
    """Class for a strand stress-strain profile by R. Devalapura and M. Tadros from the
    March-April issue of the PCI Journal.

    :param yield_strength: Strand yield strength
    :param elastic_modulus: Strand elastic modulus
    :param fracture_strain: Strand fracture strain
    :param breaking_strength: Strand breaking strength
    :param bilinear_yield_ratio: Ratio between the stress at the intersection of a
        bilinear profile, and the yield strength
    :param strain_cps: Strain control points, generates the following strain segments:
        ``[0, strain_cps[0], strain_cps[1], fracture_strain]``. Length must be equal to
        2.
    :param n_points: Number of points to discretise within each strain segment. Length
        must be equal to 3.
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    yield_strength: float
    elastic_modulus: float
    fracture_strain: float
    breaking_strength: float
    bilinear_yield_ratio: float = 1.04
    strain_cps: List[float] = field(default_factory=lambda: [0.005, 0.015])
    n_points: List[int] = field(default_factory=lambda: [5, 14, 5])

    def __post_init__(self) -> None:
        # validate control points
        if len(self.strain_cps) != 2:
            raise ValueError("Length of strain_cps must be equal to 2.")

        if len(self.n_points) != 3:
            raise ValueError("Length of n_points must be equal to 3.")

        # determine constants
        f_so = self.bilinear_yield_ratio * self.yield_strength
        const_c = self.elastic_modulus / f_so
        const_a = (
            self.elastic_modulus
            * (self.breaking_strength - f_so)
            / (self.fracture_strain * self.elastic_modulus - f_so)
        )
        const_b = self.elastic_modulus - const_a

        # function that determines the stress
        def stress_eq(a, b, c, d, eps_ps, f_pu):
            if eps_ps != 0:
                sign = eps_ps / abs(eps_ps)  # get sign of strain
            else:
                sign = 1

            eps_ps = abs(eps_ps)  # ensure strain is positive
            denom = pow(1 + pow(c * eps_ps, d), 1 / d)  # calculate denominator
            stress = min(eps_ps * (a + b / denom), f_pu)  # calculate stress

            return sign * stress

        # determine constant D that yields the yield strength at a strain of 0.01
        def find_d(const_d):
            return (
                stress_eq(
                    a=const_a,
                    b=const_b,
                    c=const_c,
                    d=const_d,
                    eps_ps=0.01,
                    f_pu=self.breaking_strength,
                )
                - self.yield_strength
            )

        const_d = brentq(f=find_d, a=1, b=20)

        # generate stresses and strains
        seg1 = np.linspace(
            start=0, stop=self.strain_cps[0], num=self.n_points[0], endpoint=False
        )
        seg2 = np.linspace(
            start=self.strain_cps[0],
            stop=self.strain_cps[1],
            num=self.n_points[1],
            endpoint=False,
        )
        seg3 = np.linspace(
            start=self.strain_cps[1], stop=self.fracture_strain, num=self.n_points[2]
        )
        strain_list = seg1.tolist() + seg2.tolist() + seg3.tolist()

        # generate compressive region of profile
        strains_c = []
        stresses_c = []

        for strain in strain_list:
            strains_c.append(strain)
            stresses_c.append(
                stress_eq(
                    a=const_a,
                    b=const_b,
                    c=const_c,
                    d=const_d,
                    eps_ps=strain,
                    f_pu=self.breaking_strength,
                )
            )

        # generate tensile region of profile
        strains_t = [eps * -1.0 for eps in strains_c[:0:-1]]
        stresses_t = [sig * -1.0 for sig in stresses_c[:0:-1]]

        # combine lists
        self.strains = strains_t + strains_c
        self.stresses = stresses_t + stresses_c

        return super().__post_init__()
