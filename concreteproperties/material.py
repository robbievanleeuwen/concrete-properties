from __future__ import annotations

from dataclasses import dataclass, field

import concreteproperties.stress_strain_profile as ssp


@dataclass
class Material:
    """Generic class for a *concreteproperties* material.

    :param name: Material name
    :param density: Material density (mass per unit volume)
    :param stress_strain_profile: Material stress-strain profile
    :param colour: Colour of the material for rendering
    :param meshed: If set to True, the entire material region is meshed; if set to
        False, the material region is treated as a lumped circular mass at its centroid
    """

    name: str
    density: float
    stress_strain_profile: ssp.StressStrainProfile
    colour: str
    meshed: bool

    def __post_init__(self):
        # set elastic modulus
        self.elastic_modulus = self.stress_strain_profile.get_elastic_modulus()


@dataclass
class Concrete(Material):
    """Class for a concrete material.

    :param name: Concrete material name
    :param density: Concrete density (mass per unit volume)
    :param stress_strain_profile: Service concrete stress-strain profile
    :param ultimate_stress_strain_profile: Ultimate concrete stress-strain profile
    :param flexural_tensile_strength: Absolute value of the concrete flexural
        tensile strength
    :param colour: Colour of the material for rendering
    """

    name: str
    density: float
    stress_strain_profile: ssp.ConcreteServiceProfile
    ultimate_stress_strain_profile: ssp.ConcreteUltimateProfile
    flexural_tensile_strength: float
    colour: str
    meshed: bool = field(default=True, init=False)

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.stress_strain_profile, ssp.ConcreteServiceProfile):
            msg = "Concrete stress_strain_profile must be a "
            msg += "ConcreteServiceProfile object."
            raise ValueError(msg)

        if not isinstance(
            self.ultimate_stress_strain_profile, ssp.ConcreteUltimateProfile
        ):
            msg = "Concrete ultimate_stress_strain_profile must be a "
            msg += "ConcreteUltimateProfile object."
            raise ValueError(msg)


@dataclass
class Steel(Material):
    """Class for a steel material with the entire region meshed to allow for strain
    variation across the section, e.g. structural steel profiles.

    :param name: Steel material name
    :param density: Steel density (mass per unit volume)
    :param stress_strain_profile: Steel stress-strain profile
    :param colour: Colour of the material for rendering
    """

    name: str
    density: float
    stress_strain_profile: ssp.StressStrainProfile
    colour: str
    meshed: bool = field(default=True, init=False)


@dataclass
class SteelBar(Steel):
    """Class for a steel bar material, treated as a lumped circular mass with a constant
    strain.

    :param name: Steel bar material name
    :param density: Steel bar density (mass per unit volume)
    :param stress_strain_profile: Steel bar stress-strain profile
    :param colour: Colour of the material for rendering
    """

    name: str
    density: float
    stress_strain_profile: ssp.StressStrainProfile
    colour: str
    meshed: bool = field(default=False, init=False)


# @dataclass
# class SteelStrand(Steel):
#     # placeholder
#     pass
