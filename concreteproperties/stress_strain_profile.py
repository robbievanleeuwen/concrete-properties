from typing import List
from scipy.interpolate import interp1d


class StressStrainProfile:
    """Abstract base class for a material stress-strain profile."""

    def get_stress(
        self,
        strain: float,
    ):
        """Returns a stress given a strain.

        :param float strain: Strain at which to return a stress.
        """

        raise NotImplementedError


class WhitneyStressBlock(StressStrainProfile):
    pass


class ParabolicStressBlock(StressStrainProfile):
    pass


class PCAStressProfile(StressStrainProfile):
    pass


class PiecewiseLinearProfile(StressStrainProfile):
    """Class for a piecewise linear stress strain profile."""

    def __init__(
        self,
        strain: List[float],
        stress: List[float],
    ):
        """Inits the BilinearProfile class.

        :param strain: List of strains (must start with 0 and must be increasing)
        :type strain: List[float]
        :param stress: List of stresses (must start with 0 and must be increasing)
        :type stress: List[float]
        """

        # validate input - first value zero
        if strain[0] != 0 or stress[0] != 0:
            raise ValueError("First value of strain and stress must be zero.")

        # validate input - same length lists
        if len(strain) != len(stress):
            raise ValueErorr("Length of strain must equal length of stress")

        # validate input - increasing values
        prev_strain = 0
        prev_stress = 0

        for idx in range(len(strain)):
            if idx != 0:
                if strain[idx] < prev_strain or stress[idx] < prev_stress:
                    msg = "strain and strain must containing increasing values."
                    raise ValueError(msg)

        self.strain = strain
        self.stress = stress

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
            x=strain,
            y=stress,
            kind="linear",
            fill_value="extrapolate",
        )

        return mult * stress_function(abs(strain))


class BilinearProfile(PiecewiseLinearProfile):
    """Class for a bilinear stress strain profile."""

    def __init__(
        self,
        strain1: float,
        strain2: float,
        stress1: float,
        stress2: float,
    ):
        """Inits the BilinearProfile class.

        :param float strain1: Strain at kink in bilinear curve.
        :param float strain2: Strain at end of bilinear curve.
        :param float stress1: Stress at kink in bilinear curve.
        :param float stress2: Stress at end of bilinear curve.
        """

        super.__init__(
            strain=[0, strain1, strain2],
            stress=[0, stress1, stress2],
        )
