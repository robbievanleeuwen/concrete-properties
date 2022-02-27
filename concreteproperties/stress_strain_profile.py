from typing import List
from scipy.interpolate import interp1d


class StressStrainProfile:
    """Abstract base class for a material stress-strain profile."""

    def get_stress(
        self,
        strain: float,
    ):
        """Returns a stress given a strain.

        :param float strain: Strain at which to return a stress
        """

        raise NotImplementedError

    def plot_stress_strain(
        self,
    ):
        """Plots the stress-strain profile."""

        raise NotImplementedError


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

        self.alpha_2 = alpha_2
        self.gamma = gamma
        self.compressive_strength = compressive_strength
        self.ultimate_strain = ultimate_strain

    def get_stress(
        self,
        strain: float,
    ):
        """Returns a stress given a strain.

        :param float strain: Strain at which to return a stress.
        """

        if strain > self.ultimate_strain * (1 - self.gamma):
            return self.alpha_2 * self.compressive_strength
        else:
            return 0


class ParabolicStressBlock(StressStrainProfile):
    pass


class PCAStressProfile(StressStrainProfile):
    pass


class PiecewiseLinearProfile(StressStrainProfile):
    """Class for a piecewise linear stress-strain profile."""

    def __init__(
        self,
        strains: List[float],
        stresses: List[float],
    ):
        """Inits the PiecewiseLinearProfile class.

        :param strains: List of strains (must start with 0 and must be increasing)
        :type strains: List[float]
        :param stresses: List of stresses (must start with 0 and must be increasing or
            equal)
        :type stresses: List[float]
        """

        # validate input - first value zero
        if strains[0] != 0 or stresses[0] != 0:
            raise ValueError("First value of strains and stresses must be zero.")

        # validate input - same length lists
        if len(strains) != len(stresses):
            raise ValueError("Length of strains must equal length of stresses")

        # validate input - length > 1
        if len(strains) < 2:
            raise ValueError("Length of strains and stresses must be greater than 1")

        # validate input - increasing values
        prev_strain = 0
        prev_stress = 0

        for idx in range(len(strains)):
            if idx != 0:
                if strains[idx] <= prev_strain or stresses[idx] < prev_stress:
                    msg = "strains and stresses must containing increasing values."
                    raise ValueError(msg)

                prev_strain = strains[idx]
                prev_stress = stresses[idx]

        self.strains = strains
        self.stresses = stresses

    def get_stress(
        self,
        strain: float,
    ):
        """Returns a stress given a strain for the bilinear profile.

        :param float strain: Strain at which to return a stress.
        """

        # determine if strain is positive or negative
        if strain > 0:
            mult = 1
        else:
            mult = -1

        # create interpolation function
        stress_function = interp1d(
            x=self.strains,
            y=self.stresses,
            kind="linear",
            fill_value="extrapolate",
        )

        return mult * stress_function(abs(strain))


class BilinearProfile(PiecewiseLinearProfile):
    """Class for a bilinear stress-strain profile."""

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
            strains=[0, strain1, strain2],
            stresses=[0, stress1, stress2],
        )


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
