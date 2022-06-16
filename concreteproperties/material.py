from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, field

from sectionproperties.pre.pre import Material

if TYPE_CHECKING:
    from concreteproperties.stress_strain_profile import StressStrainProfile


@dataclass(eq=True, frozen=True)
class Concrete(Material):
    """Class for a concrete material.

    :param string name: Concrete material name
    :param float elastic_modulus: Concrete modulus of elasticity
    :param float poissons_ratio: Material Poisson's ratio
    :param float density: Concrete density (mass per unit volume)
    :param str color: Colour of the material for rendering
    :param stress_strain_profile: Concrete stress-strain profile
    :type stress_strain_profile:
        :class:`~concreteproperties.stress_strain_profile.StressStrainProfile`
    :param float compressive_strength: Concrete compressive strength
    :param float alpha_1: Factor that modifies the concrete compressive strength at
        squash load
    """

    compressive_strength: float
    alpha_1: float
    stress_strain_profile: StressStrainProfile
    yield_strength: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "yield_strength", self.compressive_strength)


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

    stress_strain_profile: StressStrainProfile
