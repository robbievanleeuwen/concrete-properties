from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt

from concreteproperties.post import plotting_context

if TYPE_CHECKING:
    from sectionproperties.pre.geometry import Geometry
    import matplotlib.axes


@dataclass
class ConcreteProperties:
    """Class for storing gross concrete section properties."""

    # section areas
    total_area: float = 0
    concrete_area: float = 0
    steel_area: float = 0
    e_a: float = 0
    e_a: float = 0

    # section mass
    mass: float = 0

    # section perimeter
    perimeter: float = 0

    # first moments of area
    e_qx: float = 0
    e_qy: float = 0

    # centroids
    cx: float = 0
    cy: float = 0

    # second moments of area
    e_ixx_g: float = 0
    e_iyy_g: float = 0
    e_ixy_g: float = 0
    e_ixx_c: float = 0
    e_iyy_c: float = 0
    e_ixy_c: float = 0
    e_i11_c: float = 0
    e_i22_c: float = 0

    # principal axis angle
    phi: float = 0

    # section moduli
    e_zxx_plus: float = 0
    e_zxx_minus: float = 0
    e_zyy_plus: float = 0
    e_zyy_minus: float = 0
    e_z11_plus: float = 0
    e_z11_minus: float = 0
    e_z22_plus: float = 0
    e_z22_minus: float = 0

    # plastic properties
    squash_load: float = 0
    tensile_load: float = 0
    axial_pc_x: float = 0
    axial_pc_y: float = 0
    conc_ultimate_strain: float = 0


@dataclass
class TransformedConcreteProperties:
    """Class for storing transformed gross concrete section properties.

    :param concrete_properties: Concrete properties object
    :type concrete_properties:
        :class:`~concreteproperties.concrete_section.ConcreteProperties`
    :param float elastic_modulus: Reference elastic modulus
    """

    concrete_properties: ConcreteProperties = field(repr=False)
    elastic_modulus: float

    # first moments of area
    qx: float = 0
    qy: float = 0

    # second moments of area
    ixx_g: float = 0
    iyy_g: float = 0
    ixy_g: float = 0
    ixx_c: float = 0
    iyy_c: float = 0
    ixy_c: float = 0
    i11_c: float = 0
    i22_c: float = 0

    # section moduli
    zxx_plus: float = 0
    zxx_minus: float = 0
    zyy_plus: float = 0
    zyy_minus: float = 0
    z11_plus: float = 0
    z11_minus: float = 0
    z22_plus: float = 0
    z22_minus: float = 0

    def __post_init__(
        self,
    ):

        self.qx = self.concrete_properties.e_qx / self.elastic_modulus
        self.qy = self.concrete_properties.e_qy / self.elastic_modulus
        self.ixx_g = self.concrete_properties.e_ixx_g / self.elastic_modulus
        self.iyy_g = self.concrete_properties.e_iyy_g / self.elastic_modulus
        self.ixy_g = self.concrete_properties.e_ixy_g / self.elastic_modulus
        self.ixx_c = self.concrete_properties.e_ixx_c / self.elastic_modulus
        self.iyy_c = self.concrete_properties.e_iyy_c / self.elastic_modulus
        self.ixy_c = self.concrete_properties.e_ixy_c / self.elastic_modulus
        self.i11_c = self.concrete_properties.e_i11_c / self.elastic_modulus
        self.i22_c = self.concrete_properties.e_i22_c / self.elastic_modulus
        self.zxx_plus = self.concrete_properties.e_zxx_plus / self.elastic_modulus
        self.zxx_minus = self.concrete_properties.e_zxx_minus / self.elastic_modulus
        self.zyy_plus = self.concrete_properties.e_zyy_plus / self.elastic_modulus
        self.zyy_minus = self.concrete_properties.e_zyy_minus / self.elastic_modulus
        self.z11_plus = self.concrete_properties.e_z11_plus / self.elastic_modulus
        self.z11_minus = self.concrete_properties.e_z11_minus / self.elastic_modulus
        self.z22_plus = self.concrete_properties.e_z22_plus / self.elastic_modulus
        self.z22_minus = self.concrete_properties.e_z22_minus / self.elastic_modulus


@dataclass
class CrackedResults:
    """Class for storing cracked concrete section properties."""

    theta: float
    m_cr: float = 0
    d_nc: float = 0
    cracked_geometries: List[Geometry] = field(default_factory=list, repr=False)
    e_a_cr: float = 0
    e_qx_cr: float = 0
    e_qy_cr: float = 0
    e_ixx_g_cr: float = 0
    e_iyy_g_cr: float = 0
    e_ixy_g_cr: float = 0
    e_ixx_c_cr: float = 0
    e_iyy_c_cr: float = 0
    e_ixy_c_cr: float = 0
    e_iuu_cr: float = 0

    # transformed properties
    a_cr: float = 0
    qx_cr: float = 0
    qy_cr: float = 0
    ixx_g_cr: float = 0
    iyy_g_cr: float = 0
    ixy_g_cr: float = 0
    ixx_c_cr: float = 0
    iyy_c_cr: float = 0
    ixy_c_cr: float = 0
    iuu_cr: float = 0

    def calculate_transformed_properties(
        self,
        elastic_modulus: float,
    ):
        """Calculates and stores transformed cracked properties using a reference
        elastic modulus.

        :param float elastic_modulus: Reference elastic modulus
        """

        self.a_cr = self.e_a_cr / elastic_modulus
        self.qx_cr = self.e_qx_cr / elastic_modulus
        self.qy_cr = self.e_qy_cr / elastic_modulus
        self.ixx_g_cr = self.e_ixx_g_cr / elastic_modulus
        self.iyy_g_cr = self.e_iyy_g_cr / elastic_modulus
        self.ixy_g_cr = self.e_ixy_g_cr / elastic_modulus
        self.ixx_c_cr = self.e_ixx_c_cr / elastic_modulus
        self.iyy_c_cr = self.e_iyy_c_cr / elastic_modulus
        self.ixy_c_cr = self.e_ixy_c_cr / elastic_modulus
        self.iuu_cr = self.e_iuu_cr / elastic_modulus


@dataclass
class MomentCurvatureResults:
    """Class for storing moment curvature results.

    :param float theta: Angle the neutral axis makes with the horizontal axis
    """

    # results
    theta: float
    kappa: List[float] = field(default_factory=list)
    moment: List[float] = field(default_factory=list)

    # for analysis
    _n_i: float = field(default=0, repr=False)
    _m_i: float = field(default=0, repr=False)
    _failure: bool = field(default=False, repr=False)

    def __post_init__(
        self,
    ):
        self.kappa.append(0)
        self.moment.append(0)

    def plot_results(
        self,
        m_scale: float = 1e-6,
        **kwargs,
    ) -> matplotlib.axes._subplots.AxesSubplot:
        """Plots the moment curvature results.

        :param float m_scale: Scaling factor to apply to bending moment
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes._subplots.AxesSubplot`
        """

        # scale moments
        moments = np.array(self.moment) * m_scale

        # create plot and setup the plot
        with plotting_context(title="Moment-Curvature", **kwargs) as (
            fig,
            ax,
        ):
            ax.plot(self.kappa, moments, "o-", markersize=3)
            plt.xlabel("Curvature")
            plt.ylabel("Moment")
            plt.grid(True)

        return ax


@dataclass
class UltimateBendingResults:
    """Class for storing ultimate bending results.

    :param float theta: Angle the neutral axis makes with the horizontal axis
    """

    theta: float
    d_n: float = 0
    n: float = 0
    mx: float = 0
    my: float = 0
    mv: float = 0


@dataclass
class MomentInteractionResults:
    """Class for storing moment interaction results."""

    n: List[float] = field(default_factory=list)
    m: List[float] = field(default_factory=list)

    def plot_diagram(
        self,
        n_scale: float = 1e-3,
        m_scale: float = 1e-6,
        **kwargs,
    ) -> matplotlib.axes._subplots.AxesSubplot:
        """Plots a moment interaction diagram.

        :param float n_scale: Scaling factor to apply to axial force
        :param float m_scale: Scaling factor to apply to bending moment
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes._subplots.AxesSubplot`
        """

        # create plot and setup the plot
        with plotting_context(title="Moment Interaction Diagram", **kwargs) as (
            fig,
            ax,
        ):
            # scale results
            forces = np.array(self.n) * n_scale
            moments = np.array(self.m) * m_scale

            ax.plot(moments, forces, "o-", markersize=3)

            plt.xlabel("Bending Moment")
            plt.ylabel("Axial Force")
            plt.grid(True)

        return ax

    @staticmethod
    def plot_multiple_diagrams(
        moment_interaction_results: List[MomentInteractionResults],
        labels: List[str],
        n_scale: float = 1e-3,
        m_scale: float = 1e-6,
        **kwargs,
    ) -> matplotlib.axes._subplots.AxesSubplot:
        """Plots multiple moment interaction diagrams.

        :param moment_interaction_results: List of moment interaction results objects
        :type moment_interaction_results:
            List[:class:`~concreteproperties.results.MomentInteractionResults`]
        :param labels: List of labels for each moment interaction diagram
        :type labels: List[str]
        :param float n_scale: Scaling factor to apply to axial force
        :param float m_scale: Scaling factor to apply to bending moment
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes._subplots.AxesSubplot`
        """

        # create plot and setup the plot
        with plotting_context(title="Moment Interaction Diagram", **kwargs) as (
            fig,
            ax,
        ):
            # for each M-N curve
            for idx, mi_result in enumerate(moment_interaction_results):
                # scale results
                forces = np.array(mi_result.n) * n_scale
                moments = np.array(mi_result.m) * m_scale

                ax.plot(moments, forces, "o-", label=labels[idx], markersize=3)

            plt.xlabel("Bending Moment")
            plt.ylabel("Axial Force")
            plt.grid(True)

            # if there is more than one curve show legend
            if idx > 0:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return ax

@dataclass
class BiaxialBendingResults:
    """Class for storing biaxial bending results.

    :param float n: Net axial force
    """

    n: float
    mx: List[float] = field(default_factory=list)
    my: List[float] = field(default_factory=list)

    def plot_diagram(
        self,
        m_scale: float = 1e-6,
        **kwargs,
    ) -> matplotlib.axes._subplots.AxesSubplot:
        """Plots a biaxial bending diagram.

        :param float m_scale: Scaling factor to apply to bending moment
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes._subplots.AxesSubplot`
        """

        # create plot and setup the plot
        with plotting_context(title=f"Biaxial Bending Diagram, $N = {self.n:.3e}$", **kwargs) as (
            fig,
            ax,
        ):
            # scale results
            mx = np.array(self.mx) * m_scale
            my = np.array(self.my) * m_scale

            ax.plot(mx, my, "o-", markersize=3)

            plt.xlabel("Bending Moment $M_x$")
            plt.ylabel("Bending Moment $M_y$")
            plt.grid(True)

        return ax
