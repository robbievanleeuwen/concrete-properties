"""Defines material objects to be used with concreteproperties."""

from __future__ import annotations

from dataclasses import dataclass, field

import concreteproperties.stress_strain_profile as ssp


@dataclass
class Material:
    """Generic class for a ``concreteproperties`` material.

    Args:
        name: Material name
        density: Material density (mass per unit volume)
        stress_strain_profile: Material stress-strain profile
        colour: Colour of the material for rendering, see
            https://matplotlib.org/stable/gallery/color/named_colors.html for a list of
            named colours
        meshed: If set to True, the entire material region is meshed; if set to False,
            the material region is treated as a lumped circular mass at its centroid
    """

    name: str
    density: float
    stress_strain_profile: ssp.StressStrainProfile
    colour: str
    meshed: bool

    def __post_init__(self) -> None:
        """Post init method."""
        # set elastic modulus
        self.elastic_modulus = self.stress_strain_profile.get_elastic_modulus()


@dataclass
class Concrete(Material):
    """Class for a concrete material.

    Args:
        name: Concrete material name
        density: Concrete density (mass per unit volume)
        stress_strain_profile: Service concrete stress-strain profile
        ultimate_stress_strain_profile: Ultimate concrete stress-strain profile
        flexural_tensile_strength: Absolute value of the concrete flexural tensile
            strength
        colour: Colour of the material for rendering, see
            https://matplotlib.org/stable/gallery/color/named_colors.html for a list of
            named colours

    Raises:
        ValueError: If concrete stress_strain_profile is not a ConcreteServiceProfile
            object
        ValueError: If concrete ultimate_stress_strain_profile is not a
            ConcreteUltimateProfile object
    """

    name: str
    density: float
    stress_strain_profile: ssp.ConcreteServiceProfile
    ultimate_stress_strain_profile: ssp.ConcreteUltimateProfile
    flexural_tensile_strength: float
    colour: str
    meshed: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        """Post init method.

        Raises:
            ValueError: If concrete stress_strain_profile is not a
                ConcreteServiceProfile object
            ValueError: If concrete ultimate_stress_strain_profile is not a
                ConcreteUltimateProfile object
        """
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
    """Class for a steel material.

    This steel material has the entire region meshed to allow for strain variation
    across the section, e.g. structural steel profiles in composite sections.

    Args:
        name: Steel material name
        density: Steel density (mass per unit volume)
        stress_strain_profile: Steel stress-strain profile
        colour: Colour of the material for rendering, see
            https://matplotlib.org/stable/gallery/color/named_colors.html for a list of
            named colours
    """

    name: str
    density: float
    stress_strain_profile: ssp.StressStrainProfile
    colour: str
    meshed: bool = field(default=True, init=False)


@dataclass
class SteelBar(Steel):
    """Class for a steel bar material.

    This steel material is treated as a lumped circular mass with a constant strain.

    Args:
        name: Steel bar material name
        density: Steel bar density (mass per unit volume)
        stress_strain_profile: Steel bar stress-strain profile
        Colour of the material for rendering, see
            https://matplotlib.org/stable/gallery/color/named_colors.html for a list of
            named colours
    """

    name: str
    density: float
    stress_strain_profile: ssp.StressStrainProfile
    colour: str
    meshed: bool = field(default=False, init=False)


@dataclass
class SteelStrand(Steel):
    """Class for a steel strand material.

    This steel strand material is treated as a lumped circular mass with a constant
    strain.

    .. note::

      A :class:`~concreteproperties.stress_strain_profile.StrandProfile` must be used
      if using a :class:`~concreteproperties.material.SteelStrand` object.

    .. note::

      The strand is assumed to be bonded to the concrete.

    Args:
        name: Steel strand material name
        density: Steel strand density (mass per unit volume)
        stress_strain_profile: Steel strand stress-strain profile
        Colour of the material for rendering, see
            https://matplotlib.org/stable/gallery/color/named_colors.html for a list of
            named colours
        prestress_stress: Prestressing stress applied to the strand
    """

    name: str
    density: float
    stress_strain_profile: ssp.StrandProfile
    colour: str
    prestress_stress: float = 0
    meshed: bool = field(default=False, init=False)

    def get_prestress_stress(self) -> float:
        """Returns the prestress stress.

        Returns:
            Prestress stress
        """
        return self.prestress_stress

    def get_prestress_strain(self) -> float:
        """Returns the prestress strain.

        Returns:
            Prestress strain
        """
        stress = self.get_prestress_stress()

        return self.stress_strain_profile.get_strain(stress=stress)
