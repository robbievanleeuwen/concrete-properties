from dataclasses import dataclass, field
import sectionproperties.pre.pre as sp_pre
from concreteproperties.stress_strain_profile import StressStrainProfile


@dataclass(eq=True, frozen=True)
class Concrete(sp_pre.Material):
    """Class for a concrete material.

    :param string name: Concrete material name
    :param float elastic_modulus: Concrete modulus of elasticity
    :param float compressive_strength: Concrete compressive strength
    :param float alpha_1: Factor that modifies the concrete compressive strength at
        squash load
    :param float density: Concrete density (mass per unit volume)
    :param stress_strain_profile: Concrete stress-strain profile
    :type stress_strain_profile:
        :class:`~concreteproperties.stress_strain_profile.StressStrainProfile`
    """

    compressive_strength: float
    alpha_1: float
    stress_strain_profile: StressStrainProfile
    yield_strength: float = field(init=False)
    poissons_ratio: float = field(init=False)
    color: float = field(init=False)

    def __post_init__(self):
        super().__init__(
            name=self.name,
            elastic_modulus=self.elastic_modulus,
            poissons_ratio=0.2,
            yield_strength=self.compressive_strength,
            density=self.density,
            color="lightgrey",
        )


@dataclass(eq=True, frozen=True)
class Steel(sp_pre.Material):
    """Class for a steel material.

    :param string name: Steel material name
    :param float elastic_modulus: Steel modulus of elasticity
    :param float yield_strength: Steel yield strength
    :param float density: Steel density (mass per unit volume)
    :param stress_strain_profile: Steel stress-strain profile
    :type stress_strain_profile:
        :class:`~concreteproperties.stress_strain_profile.StressStrainProfile`
    """

    stress_strain_profile: StressStrainProfile
    poissons_ratio: float = field(init=False)
    color: float = field(init=False)

    def __post_init__(self):
        super().__init__(
            name=self.name,
            elastic_modulus=self.elastic_modulus,
            poissons_ratio=0.3,
            yield_strength=self.yield_strength,
            density=self.density,
            color="grey",
        )
