from dataclasses import dataclass, field
import sectionproperties.pre.pre as sp_pre
from concreteproperties.stress_strain_profile import StressStrainProfile


@dataclass(eq=True, frozen=True)
class Material(sp_pre.Material):
    """Class for a *concreteproperties* material.

    :param string name: Material name
    :param float elastic_modulus: Material modulus of elasticity
    :param float poissons_ratio: Material Poisson's ratio
    :param float yield_strength: Material yield strength
    :param float density: Material density (mass per unit volume)
    :param str color: Colour of the material for rendering
    :param stress_strain_profile: Concrete stress-strain profile
    :type stress_strain_profile:
        :class:`~concreteproperties.stress_strain_profile.StressStrainProfile`
    """

    stress_strain_profile: StressStrainProfile


@dataclass(eq=True, frozen=True)
class Concrete(Material):
    """Class for a concrete material.

    :param string name: Concrete material name
    :param float elastic_modulus: Concrete modulus of elasticity
    :param float poissons_ratio: Material Poisson's ratio
    :param float compressive_strength: Concrete compressive strength
    :param float alpha_1: Factor that modifies the concrete compressive strength at
        squash load
    :param float density: Concrete density (mass per unit volume)
    :param str color: Colour of the material for rendering
    :param stress_strain_profile: Concrete stress-strain profile
    :type stress_strain_profile:
        :class:`~concreteproperties.stress_strain_profile.StressStrainProfile`
    """

    compressive_strength: float
    alpha_1: float
    yield_strength: float = field(init=False)

    def __post_init__(self):
        super().__init__(
            name=self.name,
            elastic_modulus=self.elastic_modulus,
            poissons_ratio=self.poissons_ratio,
            yield_strength=self.compressive_strength,
            density=self.density,
            color=self.color,
            stress_strain_profile=self.stress_strain_profile,
        )


@dataclass(eq=True, frozen=True)
class Steel(Material):
    """Class for a steel material.

    :param string name: Steel material name
    :param float elastic_modulus: Steel modulus of elasticity
    :param float poissons_ratio: Material Poisson's ratio
    :param float yield_strength: Steel yield strength
    :param float density: Steel density (mass per unit volume)
    :param str color: Colour of the material for rendering
    :param stress_strain_profile: Steel stress-strain profile
    :type stress_strain_profile:
        :class:`~concreteproperties.stress_strain_profile.StressStrainProfile`
    """

    def __post_init__(self):
        super().__init__(
            name=self.name,
            elastic_modulus=self.elastic_modulus,
            poissons_ratio=self.poissons_ratio,
            yield_strength=self.yield_strength,
            density=self.density,
            color=self.color,
            stress_strain_profile=self.stress_strain_profile,
        )
