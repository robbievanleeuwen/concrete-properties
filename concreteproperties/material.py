from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from concreteproperties.stress_strain_profile import StressStrainProfile


@dataclass(eq=True)
class Concrete:
    """Class for a concrete material.

    :param string name: Concrete material name
    :param float elastic_modulus: Concrete modulus of elasticity
    :param float density: Concrete density (mass per unit volume)
    :param ultimate_stress_strain_profile: Ultimate concrete stress-strain profile
    :type ultimate_stress_strain_profile:
        :class:`~concreteproperties.stress_strain_profile.StressStrainProfile`
    :param float alpha_1: Factor that modifies the concrete compressive strength at
        squash load
    :param float flexural_tensile_strength: Concrete flexural tensile strength
    :param float residual_shrinkage_stress: Concrete residual shrinkage stress
    :param str colour: Colour of the material for rendering
    """

    name: str
    elastic_modulus: float
    density: float
    ultimate_stress_strain_profile: StressStrainProfile
    alpha_1: float
    flexural_tensile_strength: float
    residual_shrinkage_stress: float
    colour: str


@dataclass(eq=True)
class Steel:
    """Class for a steel material.

    :param string name: Steel material name
    :param float elastic_modulus: Steel modulus of elasticity
    :param float density: Steel density (mass per unit volume)
    :param float yield_strength: Steel yield stress
    :param ultimate_stress_strain_profile: Ultimate steel stress-strain profile
    :type ultimate_stress_strain_profile:
        :class:`~concreteproperties.stress_strain_profile.StressStrainProfile`
    :param str colour: Colour of the material for rendering
    """

    name: str
    elastic_modulus: float
    density: float
    yield_strength: float
    ultimate_stress_strain_profile: StressStrainProfile
    colour: str
