from typing import List

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class StressStrainProfile:
    """Abstract base class for a material stress-strain profile.

    Implements a piecewise linear stress-strain profile.

    Positive stresses & strains are compression.
    """

    def __init__(
        self,
        strains: List[float],
        stresses: List[float],
    ):
        """Inits the StressStrainProfile class.

        :param strains: List of strains (must be increasing or equal)
        :type strains: List[float]
        :param stresses: List of stresses (must be increasing or equal)
        :type stresses: List[float]
        """

        # validate input - same length lists
        if len(strains) != len(stresses):
            raise ValueError("Length of strains must equal length of stresses")

        # validate input - length > 1
        if len(strains) < 2:
            raise ValueError("Length of strains and stresses must be greater than 1")

        # validate input - increasing values
        prev_strain = strains[0]
        prev_stress = stresses[0]

        for idx in range(len(strains)):
            if idx != 0:
                if strains[idx] < prev_strain or stresses[idx] < prev_stress:
                    msg = "strains and stresses must containing increasing values."
                    raise ValueError(msg)

                prev_strain = strains[idx]
                prev_stress = stresses[idx]

        self.strains = strains
        self.stresses = stresses

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

    def plot_stress_strain(
        self,
    ):
        """Plots the stress-strain profile."""

        raise NotImplementedError


class BilinearProfile(StressStrainProfile):
    """Class for a symmetric bilinear stress-strain profile."""

    def __init__(
        self,
        strain1: float,
        strain2: float,
        stress1: float,
        stress2: float,
    ):
        """Inits the BilinearProfile class.

        :param float strain1: Strain at kink in bilinear curve
        :param float strain2: Strain at end of bilinear curve
        :param float stress1: Stress at kink in bilinear curve
        :param float stress2: Stress at end of bilinear curve
        """

        super().__init__(
            strains=[-strain2, -strain1, 0, strain1, strain2],
            stresses=[-stress2, -stress1, 0, stress1, stress2],
        )


class WhitneyStressBlock(StressStrainProfile):
    """Class for a Whitney (rectangular) stress block."""

    def __init__(
        self,
        alpha_2: float,
        gamma: float,
        compressive_strength: float,
        ultimate_strain: float,
    ):
        """Inits the WhitneyStressBlock class.

        :param float alpha_2: Factor that modifies the concrete compressive strength
        :param float gamma: Factor that modifies the depth of the stress block
        :param float compressive_strength: Concrete compressive strength
        :param float ultimate_strain: Strain at the extreme compression fibre
        """

        super().__init__(
            strains=[
                0,
                ultimate_strain * (1 - gamma),
                ultimate_strain * (1 - gamma),
                ultimate_strain,
            ],
            stresses=[
                0,
                0,
                alpha_2 * compressive_strength,
                alpha_2 * compressive_strength,
            ],
        )


class ParabolicStressBlock(StressStrainProfile):
    pass


class PCAStressProfile(StressStrainProfile):
    pass


class SteelElasticPlastic(BilinearProfile):
    """Class for a perfectly elastic-plastic steel stress-strain profile."""

    def __init__(
        self,
        yield_strength: float,
        elastic_modulus: float,
        fracture_strain: float,
    ):
        """Inits the SteelElasticPlastic class.

        :param float yield_strength: Steel yield stress
        :param float elastic_modulus: Steel elastic modulus
        :param float fracture_strain: Steel fracture strain
        """

        self.yield_strength = yield_strength
        self.elastic_modulus = elastic_modulus
        self.fracture_strain = fracture_strain

        super().__init__(
            strain1=yield_strength / elastic_modulus,
            strain2=fracture_strain,
            stress1=yield_strength,
            stress2=yield_strength,
        )
