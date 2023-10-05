"""Abstract class for a design code."""

from __future__ import annotations

from typing import TYPE_CHECKING

import concreteproperties.results as res


if TYPE_CHECKING:
    from concreteproperties.concrete_section import ConcreteSection
    from concreteproperties.material import Concrete, SteelBar


class DesignCode:
    """Abstract class for a design code object."""

    def __init__(self) -> None:
        """Inits the DesignCode class."""
        pass

    def assign_concrete_section(
        self,
        concrete_section: ConcreteSection,
    ) -> None:
        """Assigns a concrete section to the design code.

        Args:
            concrete_section: Concrete section object to analyse
        """
        self.concrete_section = concrete_section

    def create_concrete_material(
        self,
        compressive_strength: float,
        colour: str = "lightgrey",
    ) -> Concrete:
        """Returns a concrete material object.

        List assumptions of material properties here...

        Args:
            compressive_strength: Concrete compressive strength
            colour: Colour of the concrete for rendering

        Raises:
            NotImplementedError: If this method has not been implemented by the child
                class
        """
        raise NotImplementedError

    def create_steel_material(
        self,
        yield_strength: float,
        colour: str = "grey",
    ) -> SteelBar:
        """Returns a steel bar material object.

        List assumptions of material properties here...

        Args:
            yield_strength: Steel yield strength
            colour: Colour of the steel for rendering

        Raises:
            NotImplementedError: If this method has not been implemented by the child
                class
        """
        raise NotImplementedError

    def get_gross_properties(
        self,
        **kwargs,
    ) -> res.GrossProperties:
        """Returns the gross section properties of the reinforced concrete section.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`~concreteproperties.concrete_section.ConcreteSection.get_gross_properties`

        Returns:
            Concrete properties object
        """
        return self.concrete_section.get_gross_properties(**kwargs)

    def get_transformed_gross_properties(
        self,
        **kwargs,
    ) -> res.TransformedGrossProperties:
        """Transforms gross section properties.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`~concreteproperties.concrete_section.ConcreteSection.get_transformed_gross_properties`

        Returns:
            Transformed concrete properties object
        """
        return self.concrete_section.get_transformed_gross_properties(**kwargs)

    def calculate_cracked_properties(
        self,
        **kwargs,
    ) -> res.CrackedResults:
        """Calculates cracked section properties.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_cracked_properties`

        Returns:
            Cracked results object
        """
        return self.concrete_section.calculate_cracked_properties(**kwargs)

    def moment_curvature_analysis(
        self,
        **kwargs,
    ) -> res.MomentCurvatureResults:
        """Performs a moment curvature analysis (no reduction factors applied).

        Args:
            kwargs: Keyword arguments passed to
                :meth:`~concreteproperties.concrete_section.ConcreteSection.moment_curvature_analysis`

        Returns:
            Moment curvature results object
        """
        return self.concrete_section.moment_curvature_analysis(**kwargs)

    def ultimate_bending_capacity(
        self,
        **kwargs,
    ) -> res.UltimateBendingResults:
        """Calculates the ultimate bending capacity.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`~concreteproperties.concrete_section.ConcreteSection.ultimate_bending_capacity`

        Returns:
            Ultimate bending results object
        """
        return self.concrete_section.ultimate_bending_capacity(**kwargs)

    def moment_interaction_diagram(
        self,
        **kwargs,
    ) -> res.MomentInteractionResults:
        """Generates a moment interaction diagram.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`~concreteproperties.concrete_section.ConcreteSection.moment_interaction_diagram`

        Returns:
            Moment interaction results object
        """
        return self.concrete_section.moment_interaction_diagram(**kwargs)

    def biaxial_bending_diagram(
        self,
        **kwargs,
    ) -> res.BiaxialBendingResults:
        """Generates a biaxial bending diagram.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`~concreteproperties.concrete_section.ConcreteSection.biaxial_bending_diagram`

        Returns:
            Biaxial bending results
        """
        return self.concrete_section.biaxial_bending_diagram(**kwargs)

    def calculate_uncracked_stress(
        self,
        **kwargs,
    ) -> res.StressResult:
        """Calculates uncracked stresses within the reinforced concrete section.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_uncracked_stress`

        Returns:
            Stress results object
        """
        return self.concrete_section.calculate_uncracked_stress(**kwargs)

    def calculate_cracked_stress(
        self,
        **kwargs,
    ) -> res.StressResult:
        """Calculates cracked stresses within the reinforced concrete section.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_cracked_stress`

        Returns:
            Stress results object
        """
        return self.concrete_section.calculate_cracked_stress(**kwargs)

    def calculate_service_stress(
        self,
        **kwargs,
    ) -> res.StressResult:
        """Calculates service stresses within the reinforced concrete section.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_service_stress`

        Returns:
            Stress results object
        """
        return self.concrete_section.calculate_service_stress(**kwargs)

    def calculate_ultimate_stress(
        self,
        **kwargs,
    ) -> res.StressResult:
        """Calculates ultimate stresses within the reinforced concrete section.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_ultimate_stress`

        Returns:
            Stress results object
        """
        return self.concrete_section.calculate_ultimate_stress(**kwargs)
