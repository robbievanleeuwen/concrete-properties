from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, field

from concreteproperties.stress_strain_profile import (
    ConcreteServiceProfile,
    ConcreteUltimateProfile,
    SteelProfile,
)


@dataclass(eq=True)
class Concrete:
    """Class for a concrete material.

    :param str name: Concrete material name
    :param float density: Concrete density (mass per unit volume)
    :param stress_strain_profile: Service concrete stress-strain profile
    :type stress_strain_profile:
        :class:`~concreteproperties.stress_strain_profile.ConcreteServiceProfile`
    :param ultimate_stress_strain_profile: Ultimate concrete stress-strain profile
    :type ultimate_stress_strain_profile:
        :class:`~concreteproperties.stress_strain_profile.ConcreteUltimateProfile`
    :param float alpha_squash: Factor that modifies the concrete compressive strength at
        squash load
    :param float flexural_tensile_strength: Absolute value of the concrete flexural
        tensile strength
    :param str colour: Colour of the material for rendering
    """

    name: str
    density: float
    stress_strain_profile: ConcreteServiceProfile
    ultimate_stress_strain_profile: ConcreteUltimateProfile
    alpha_squash: float
    flexural_tensile_strength: float
    colour: str

    def __post_init__(self):
        self.elastic_modulus = self.stress_strain_profile.get_elastic_modulus()

        if not isinstance(self.stress_strain_profile, ConcreteServiceProfile):
            raise ValueError(
                "Concrete stress_strain_profile must be a ConcreteServiceProfile object"
            )

        if not isinstance(self.ultimate_stress_strain_profile, ConcreteUltimateProfile):
            raise ValueError(
                "Concrete ultimate_stress_strain_profile must be a ConcreteUltimateProfile object"
            )


@dataclass(eq=True)
class Steel:
    """Class for a steel material.

    :param str name: Steel material name
    :param float density: Steel density (mass per unit volume)
    :param stress_strain_profile: Ultimate steel stress-strain profile
    :type ultimate_stress_strain_profile:
        :class:`~concreteproperties.stress_strain_profile.SteelProfile`
    :param str colour: Colour of the material for rendering
    """

    name: str
    density: float
    stress_strain_profile: SteelProfile
    colour: str

    def __post_init__(self):
        self.elastic_modulus = self.stress_strain_profile.get_elastic_modulus()

        if not isinstance(self.stress_strain_profile, SteelProfile):
            raise ValueError(
                "Steel stress_strain_profile must be a SteelProfile object"
            )
