from __future__ import annotations

from typing import List, TYPE_CHECKING
import warnings
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from rich.console import Console
from rich.table import Table

from concreteproperties.post import plotting_context

if TYPE_CHECKING:
    import matplotlib


@dataclass
class StressStrainProfile:
    """Abstract base class for a material stress-strain profile.

    Implements a piecewise linear stress-strain profile. Positive stresses & strains are
    compression.

    :param strains: List of strains (must be increasing or equal)
    :type strains: List[float]
    :param stresses: List of stresses
    :type stresses: List[float]
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

        :param float strain: Strain at which to return a stress.

        :return: Stress
        :rtype: float
        """

        # create interpolation function
        stress_function = interp1d(
            x=self.strains,
            y=self.stresses,
            kind="linear",
            fill_value="extrapolate",
        )

        return stress_function(strain)

    def get_elastic_modulus(
        self,
    ) -> float:
        """Returns the elastic modulus of the stress-strain profile.

        :return: Elastic modulus
        :rtype: float
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
        :rtype: float
        """

        return max(self.stresses)

    def get_tensile_strength(
        self,
    ) -> float:
        """Returns the most negative stress.

        :return: Tensile strength
        :rtype: float
        """

        return min(self.stresses)

    def get_ultimate_strain(
        self,
    ) -> float:
        """Returns the largest strain.

        :return: Ultimate strain
        :rtype: float
        """

        return max(self.strains)

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
        fmt: Optional[str] = "8.6e",
    ):
        """Prints the stress-strain profile properties to the terminal.

        :param fmt: Number format
        :type fmt: Optional[str]
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
            "Ultimate Strain", "{:>{fmt}}".format(self.get_ultimate_strain(), fmt=fmt)
        )

        console = Console()
        console.print(table)

    def plot_stress_strain(
        self,
        title: Optional[str] = "Stress-Strain Profile",
        **kwargs,
    ) -> matplotlib.axes._subplots.AxesSubplot:
        """Plots the stress-strain profile.

        :param title: Plot title
        :type title: Optional[str]
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes._subplots.AxesSubplot`
        """

        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (
            fig,
            ax,
        ):
            ax.plot(self.strains, self.stresses, "o-", markersize=3)
            plt.xlabel("Strain")
            plt.ylabel("Stress")
            plt.grid(True)

        return ax


@dataclass
class SteelProfile(StressStrainProfile):
    """Abstract class for a steel stress-strain profile.

    :param strains: List of strains (must be increasing or equal)
    :type strains: List[float]
    :param stresses: List of stresses
    :type stresses: List[float]
    :param float yield_strength: Steel yield strength
    :param float elastic_modulus: Steel elastic modulus
    :param float fracture_strain: Steel fracture strain
    """

    yield_strength: float
    elastic_modulus: float
    fracture_strain: float

    def print_properties(
        self,
        fmt: Optional[str] = "8.6e",
    ):
        """Prints the stress-strain profile properties to the terminal.

        :param fmt: Number format
        :type fmt: Optional[str]
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
            "Fracture Strain", "{:>{fmt}}".format(self.get_ultimate_strain(), fmt=fmt)
        )

        console = Console()
        console.print(table)


@dataclass
class SteelElasticPlastic(SteelProfile):
    """Class for a perfectly elastic-plastic steel stress-strain profile.

    :param float yield_strength: Steel yield strength
    :param float elastic_modulus: Steel elastic modulus
    :param float fracture_strain: Steel fracture strain
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

    :param float yield_strength: Steel yield strength
    :param float elastic_modulus: Steel elastic modulus
    :param float fracture_strain: Steel fracture strain
    :param float ultimate_strength: Steel ultaimte strength
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
class ConcreteServiceProfile(StressStrainProfile):
    """Abstract class for a concrete service stress-strain profile.

    :param strains: List of strains (must be increasing or equal)
    :type strains: List[float]
    :param stresses: List of stresses
    :type stresses: List[float]
    """

    def print_properties(
        self,
        fmt: Optional[str] = "8.6e",
    ):
        """Prints the stress-strain profile properties to the terminal.

        :param fmt: Number format
        :type fmt: Optional[str]
        """

        table = Table(title=f"Stress-Strain Profile - {type(self).__name__}")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row(
            "Elastic Modulus", "{:>{fmt}}".format(self.get_elastic_modulus(), fmt=fmt)
        )

        console = Console()
        console.print(table)

    def get_compressive_strength(
        self,
    ) -> float:
        """Returns the most positive stress.

        :return: Compressive strength
        :rtype: float
        """

        return None

    def get_tensile_strength(
        self,
    ) -> float:
        """Returns the most negative stress.

        :return: Tensile strength
        :rtype: float
        """

        return None

    def get_ultimate_strain(
        self,
    ) -> float:
        """Returns the largest strain.

        :return: Ultimate strain
        :rtype: float
        """

        return None


@dataclass
class ConcreteLinearProfile(ConcreteServiceProfile):
    """Class for a symmetric linear stress-strain profile.

    :param float elastic_modulus: Elastic modulus of the stress-strain profile
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    elastic_modulus: float

    def __post_init__(
        self,
    ):
        self.strains = [-0.001, 0, 0.001]
        self.stresses = [-0.001 * self.elastic_modulus, 0, 0.001 * self.elastic_modulus]


@dataclass
class ConcreteUltimateProfile(StressStrainProfile):
    """Abstract class for a concrete ultimate stress-strain profile.

    :param strains: List of strains (must be increasing or equal)
    :type strains: List[float]
    :param stresses: List of stresses
    :type stresses: List[float]
    :param float compressive_strength: Concrete compressive strength
    """

    compressive_strength: float

    def get_compressive_strength(
        self,
    ) -> float:
        """Returns the most positive stress.

        :return: Compressive strength
        :rtype: float
        """

        return self.compressive_strength

    def print_properties(
        self,
        fmt: Optional[str] = "8.6e",
    ):
        """Prints the stress-strain profile properties to the terminal.

        :param fmt: Number format
        :type fmt: Optional[str]
        """

        table = Table(title=f"Stress-Strain Profile - {type(self).__name__}")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row(
            "Compressive Strength",
            "{:>{fmt}}".format(self.get_compressive_strength(), fmt=fmt),
        )
        table.add_row(
            "Ultimate Strain", "{:>{fmt}}".format(self.get_ultimate_strain(), fmt=fmt)
        )
        console = Console()
        console.print(table)


@dataclass
class WhitneyStressBlock(ConcreteUltimateProfile):
    """Class for a Whitney (rectangular) stress block.

    :param float compressive_strength: Concrete compressive strength
    :param float alpha_2: Factor that modifies the concrete compressive strength
    :param float gamma: Factor that modifies the depth of the stress block
    :param float ultimate_strain: Strain at the extreme compression fibre
    """

    strains: List[float] = field(init=False)
    stresses: List[float] = field(init=False)
    alpha_2: float
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
            self.alpha_2 * self.compressive_strength,
            self.alpha_2 * self.compressive_strength,
        ]

    def get_stress(
        self,
        strain: float,
    ) -> float:
        """Returns a stress given a strain.

        Overrides parent method with small tolerance to aid ultimate stress generation
        at nodes.

        :param float strain: Strain at which to return a stress.

        :return: Stress
        :rtype: float
        """

        if strain >= self.strains[1] - 1e-12:
            return self.stresses[2]
        else:
            return 0
