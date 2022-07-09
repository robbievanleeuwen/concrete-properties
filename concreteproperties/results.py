from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as tri
import matplotlib.cm as cm
from matplotlib.colors import CenteredNorm
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.interpolate import interp1d
from rich.console import Console
from rich.table import Table

from concreteproperties.post import plotting_context

if TYPE_CHECKING:
    import matplotlib
    from concreteproperties.concrete_section import ConcreteSection
    from concreteproperties.analysis_section import AnalysisSection
    from sectionproperties.pre.geometry import Geometry


@dataclass
class ConcreteProperties:
    """Class for storing gross concrete section properties.

    All properties with an `e_` preceding the property are multiplied by the elastic
    modulus. In order to obtain transformed properties, call the
    :meth:`~concreteproperties.concrete_section.ConcreteSection.get_transformed_gross_properties`
    method.
    """

    # section areas
    total_area: float = 0
    concrete_area: float = 0
    steel_area: float = 0
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
    e_i11: float = 0
    e_i22: float = 0

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

    def print_results(
        self,
        fmt: Optional[str] = "8.6e",
    ):
        """Prints the gross concrete section properties to the terminal.

        :param fmt: Number format
        :type fmt: Optional[str]
        """

        table = Table(title="Gross Concrete Section Properties")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row("Total Area", "{:>{fmt}}".format(self.total_area, fmt=fmt))
        table.add_row("Concrete Area", "{:>{fmt}}".format(self.concrete_area, fmt=fmt))
        table.add_row("Steel Area", "{:>{fmt}}".format(self.steel_area, fmt=fmt))
        table.add_row("Axial Rigidity (EA)", "{:>{fmt}}".format(self.e_a, fmt=fmt))
        table.add_row("Mass (per unit length)", "{:>{fmt}}".format(self.mass, fmt=fmt))
        table.add_row("Perimeter", "{:>{fmt}}".format(self.perimeter, fmt=fmt))
        table.add_row("E.Qx", "{:>{fmt}}".format(self.e_qx, fmt=fmt))
        table.add_row("E.Qy", "{:>{fmt}}".format(self.e_qy, fmt=fmt))
        table.add_row("x-Centroid", "{:>{fmt}}".format(self.cx, fmt=fmt))
        table.add_row("y-Centroid", "{:>{fmt}}".format(self.cy, fmt=fmt))
        table.add_row("E.Ixx_g", "{:>{fmt}}".format(self.e_ixx_g, fmt=fmt))
        table.add_row("E.Iyy_g", "{:>{fmt}}".format(self.e_iyy_g, fmt=fmt))
        table.add_row("E.Ixy_g", "{:>{fmt}}".format(self.e_ixy_g, fmt=fmt))
        table.add_row("E.Ixx_c", "{:>{fmt}}".format(self.e_ixx_c, fmt=fmt))
        table.add_row("E.Iyy_c", "{:>{fmt}}".format(self.e_iyy_c, fmt=fmt))
        table.add_row("E.Ixy_c", "{:>{fmt}}".format(self.e_ixy_c, fmt=fmt))
        table.add_row("E.I11", "{:>{fmt}}".format(self.e_i11, fmt=fmt))
        table.add_row("E.I22", "{:>{fmt}}".format(self.e_i22, fmt=fmt))
        table.add_row("Principal Axis Angle", "{:>{fmt}}".format(self.phi, fmt=fmt))
        table.add_row("E.Zxx+", "{:>{fmt}}".format(self.e_zxx_plus, fmt=fmt))
        table.add_row("E.Zxx-", "{:>{fmt}}".format(self.e_zxx_minus, fmt=fmt))
        table.add_row("E.Zyy+", "{:>{fmt}}".format(self.e_zyy_plus, fmt=fmt))
        table.add_row("E.Zyy-", "{:>{fmt}}".format(self.e_zyy_minus, fmt=fmt))
        table.add_row("E.Z11+", "{:>{fmt}}".format(self.e_z11_plus, fmt=fmt))
        table.add_row("E.Z11-", "{:>{fmt}}".format(self.e_z11_minus, fmt=fmt))
        table.add_row("E.Z22+", "{:>{fmt}}".format(self.e_z22_plus, fmt=fmt))
        table.add_row("E.Z22-", "{:>{fmt}}".format(self.e_z22_minus, fmt=fmt))
        table.add_row("Squash Load", "{:>{fmt}}".format(self.squash_load, fmt=fmt))
        table.add_row("Tensile Load", "{:>{fmt}}".format(self.tensile_load, fmt=fmt))
        table.add_row(
            "x-Axial Plastic Centroid", "{:>{fmt}}".format(self.axial_pc_x, fmt=fmt)
        )
        table.add_row(
            "y-Axial Plastic Centroid", "{:>{fmt}}".format(self.axial_pc_y, fmt=fmt)
        )
        table.add_row(
            "Ultimate Concrete Strain",
            "{:>{fmt}}".format(self.conc_ultimate_strain, fmt=fmt),
        )

        console = Console()
        console.print(table)


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

    # area
    area: float = 0

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
    i11: float = 0
    i22: float = 0

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
        self.area = self.concrete_properties.total_area / self.elastic_modulus
        self.qx = self.concrete_properties.e_qx / self.elastic_modulus
        self.qy = self.concrete_properties.e_qy / self.elastic_modulus
        self.ixx_g = self.concrete_properties.e_ixx_g / self.elastic_modulus
        self.iyy_g = self.concrete_properties.e_iyy_g / self.elastic_modulus
        self.ixy_g = self.concrete_properties.e_ixy_g / self.elastic_modulus
        self.ixx_c = self.concrete_properties.e_ixx_c / self.elastic_modulus
        self.iyy_c = self.concrete_properties.e_iyy_c / self.elastic_modulus
        self.ixy_c = self.concrete_properties.e_ixy_c / self.elastic_modulus
        self.i11 = self.concrete_properties.e_i11 / self.elastic_modulus
        self.i22 = self.concrete_properties.e_i22 / self.elastic_modulus
        self.zxx_plus = self.concrete_properties.e_zxx_plus / self.elastic_modulus
        self.zxx_minus = self.concrete_properties.e_zxx_minus / self.elastic_modulus
        self.zyy_plus = self.concrete_properties.e_zyy_plus / self.elastic_modulus
        self.zyy_minus = self.concrete_properties.e_zyy_minus / self.elastic_modulus
        self.z11_plus = self.concrete_properties.e_z11_plus / self.elastic_modulus
        self.z11_minus = self.concrete_properties.e_z11_minus / self.elastic_modulus
        self.z22_plus = self.concrete_properties.e_z22_plus / self.elastic_modulus
        self.z22_minus = self.concrete_properties.e_z22_minus / self.elastic_modulus

    def print_results(
        self,
        fmt: Optional[str] = "8.6e",
    ):
        """Prints the transformed gross concrete section properties to the terminal.

        :param fmt: Number format
        :type fmt: Optional[str]
        """

        table = Table(title="Transformed Gross Concrete Section Properties")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row("E_ref", "{:>{fmt}}".format(self.elastic_modulus, fmt=fmt))
        table.add_row("Area", "{:>{fmt}}".format(self.area, fmt=fmt))
        table.add_row("Qx", "{:>{fmt}}".format(self.qx, fmt=fmt))
        table.add_row("Qy", "{:>{fmt}}".format(self.qy, fmt=fmt))
        table.add_row("Ixx_g", "{:>{fmt}}".format(self.ixx_g, fmt=fmt))
        table.add_row("Iyy_g", "{:>{fmt}}".format(self.iyy_g, fmt=fmt))
        table.add_row("Ixy_g", "{:>{fmt}}".format(self.ixy_g, fmt=fmt))
        table.add_row("Ixx_c", "{:>{fmt}}".format(self.ixx_c, fmt=fmt))
        table.add_row("Iyy_c", "{:>{fmt}}".format(self.iyy_c, fmt=fmt))
        table.add_row("Ixy_c", "{:>{fmt}}".format(self.ixy_c, fmt=fmt))
        table.add_row("I11", "{:>{fmt}}".format(self.i11, fmt=fmt))
        table.add_row("I22", "{:>{fmt}}".format(self.i22, fmt=fmt))
        table.add_row("Zxx+", "{:>{fmt}}".format(self.zxx_plus, fmt=fmt))
        table.add_row("Zxx-", "{:>{fmt}}".format(self.zxx_minus, fmt=fmt))
        table.add_row("Zyy+", "{:>{fmt}}".format(self.zyy_plus, fmt=fmt))
        table.add_row("Zyy-", "{:>{fmt}}".format(self.zyy_minus, fmt=fmt))
        table.add_row("Z11+", "{:>{fmt}}".format(self.z11_plus, fmt=fmt))
        table.add_row("Z11-", "{:>{fmt}}".format(self.z11_minus, fmt=fmt))
        table.add_row("Z22+", "{:>{fmt}}".format(self.z22_plus, fmt=fmt))
        table.add_row("Z22-", "{:>{fmt}}".format(self.z22_minus, fmt=fmt))

        console = Console()
        console.print(table)


@dataclass
class CrackedResults:
    """Class for storing cracked concrete section properties.

    All properties with an `e_` preceding the property are multiplied by the elastic
    modulus. In order to obtain transformed properties, call the
    :meth:`~concreteproperties.results.CrackedResults.calculate_transformed_properties`
    method.

    :param float theta: Angle (in radians) the neutral axis makes with the horizontal axis (-pi <= theta <= pi)
    """

    theta: float
    m_cr: float = 0
    d_nc: float = 0
    cracked_geometries: List[Geometry] = field(default_factory=list, repr=False)
    e_a_cr: float = 0
    e_qx_cr: float = 0
    e_qy_cr: float = 0
    cx: float = 0
    cy: float = 0
    e_ixx_g_cr: float = 0
    e_iyy_g_cr: float = 0
    e_ixy_g_cr: float = 0
    e_ixx_c_cr: float = 0
    e_iyy_c_cr: float = 0
    e_ixy_c_cr: float = 0
    e_iuu_cr: float = 0

    # transformed properties
    elastic_modulus_ref: float = None
    a_cr: float = None
    qx_cr: float = None
    qy_cr: float = None
    ixx_g_cr: float = None
    iyy_g_cr: float = None
    ixy_g_cr: float = None
    ixx_c_cr: float = None
    iyy_c_cr: float = None
    ixy_c_cr: float = None
    iuu_cr: float = None

    def calculate_transformed_properties(
        self,
        elastic_modulus: float,
    ):
        """Calculates and stores transformed cracked properties using a reference
        elastic modulus.

        :param float elastic_modulus: Reference elastic modulus
        """

        self.elastic_modulus_ref = elastic_modulus
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

    def print_results(
        self,
        fmt: Optional[str] = "8.6e",
    ):
        """Prints the cracked concrete section properties to the terminal.

        :param fmt: Number format
        :type fmt: Optional[str]
        """

        table = Table(title="Cracked Concrete Section Properties")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row("theta", "{:>{fmt}}".format(self.theta, fmt=fmt))

        if self.elastic_modulus_ref:
            table.add_row(
                "E_ref", "{:>{fmt}}".format(self.elastic_modulus_ref, fmt=fmt)
            )

        table.add_row("M_cr", "{:>{fmt}}".format(self.m_cr, fmt=fmt))
        table.add_row("d_nc", "{:>{fmt}}".format(self.d_nc, fmt=fmt))

        if self.a_cr:
            table.add_row("A_cr", "{:>{fmt}}".format(self.a_cr, fmt=fmt))

        table.add_row("E.A_cr", "{:>{fmt}}".format(self.e_a_cr, fmt=fmt))

        if self.qx_cr:
            table.add_row("Qx_cr", "{:>{fmt}}".format(self.qx_cr, fmt=fmt))
            table.add_row("Qy_cr", "{:>{fmt}}".format(self.qy_cr, fmt=fmt))

        table.add_row("E.Qx_cr", "{:>{fmt}}".format(self.e_qx_cr, fmt=fmt))
        table.add_row("E.Qy_cr", "{:>{fmt}}".format(self.e_qy_cr, fmt=fmt))
        table.add_row("x-Centroid", "{:>{fmt}}".format(self.cx, fmt=fmt))
        table.add_row("y-Centroid", "{:>{fmt}}".format(self.cy, fmt=fmt))

        if self.ixx_g_cr:
            table.add_row("Ixx_g_cr", "{:>{fmt}}".format(self.ixx_g_cr, fmt=fmt))
            table.add_row("Iyy_g_cr", "{:>{fmt}}".format(self.iyy_g_cr, fmt=fmt))
            table.add_row("Ixy_g_cr", "{:>{fmt}}".format(self.ixy_g_cr, fmt=fmt))
            table.add_row("Ixx_c_cr", "{:>{fmt}}".format(self.ixx_c_cr, fmt=fmt))
            table.add_row("Iyy_c_cr", "{:>{fmt}}".format(self.iyy_c_cr, fmt=fmt))
            table.add_row("Ixy_c_cr", "{:>{fmt}}".format(self.ixy_c_cr, fmt=fmt))
            table.add_row("Iuu_cr", "{:>{fmt}}".format(self.iuu_cr, fmt=fmt))

        table.add_row("E.Ixx_g_cr", "{:>{fmt}}".format(self.e_ixx_g_cr, fmt=fmt))
        table.add_row("E.Iyy_g_cr", "{:>{fmt}}".format(self.e_iyy_g_cr, fmt=fmt))
        table.add_row("E.Ixy_g_cr", "{:>{fmt}}".format(self.e_ixy_g_cr, fmt=fmt))
        table.add_row("E.Ixx_c_cr", "{:>{fmt}}".format(self.e_ixx_c_cr, fmt=fmt))
        table.add_row("E.Iyy_c_cr", "{:>{fmt}}".format(self.e_iyy_c_cr, fmt=fmt))
        table.add_row("E.Ixy_c_cr", "{:>{fmt}}".format(self.e_ixy_c_cr, fmt=fmt))
        table.add_row("E.Iuu_cr", "{:>{fmt}}".format(self.e_iuu_cr, fmt=fmt))

        console = Console()
        console.print(table)


@dataclass
class MomentCurvatureResults:
    """Class for storing moment curvature results.

    :param float theta: Angle (in radians) the neutral axis makes with the horizontal axis (-pi <= theta <= pi)
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
        m_scale: Optional[float] = 1e-6,
        **kwargs,
    ) -> matplotlib.axes._subplots.AxesSubplot:
        """Plots the moment curvature results.

        :param m_scale: Scaling factor to apply to bending moment
        :type m_scale: Optional[float]
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

    def get_curvature(
        self,
        moment: float,
    ) -> float:
        """Given a moment, uses the moment-curvature results to interpolate a curvature.

        Raises a ValueError if supplied moment is outside bounds of moment-curvature
        results. (TODO)

        :param float moment: Bending moment at which to obtain curvature

        :return: Curvature
        :rtype: float
        """

        f_kappa = interp1d(
            x=self.moment,
            y=self.kappa,
            kind="linear",
        )

        return float(f_kappa(moment))


@dataclass
class UltimateBendingResults:
    """Class for storing ultimate bending results.

    :param float theta: Angle (in radians) the neutral axis makes with the horizontal axis (-pi <= theta <= pi)
    """

    # bending angle
    theta: float

    # ultimate neutral axis depth
    d_n: float = None
    k_u: float = None

    # resultant actions
    n: float = None
    mx: float = None
    my: float = None
    mv: float = None

    def print_results(
        self,
        fmt: Optional[str] = "8.6e",
    ):
        """Prints the ultimate bending results to the terminal.

        :param fmt: Number format
        :type fmt: Optional[str]
        """

        table = Table(title="Ultimate Bending Results")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row("Bending Angle - theta", "{:>{fmt}}".format(self.theta, fmt=fmt))
        table.add_row("Neutral Axis Depth - d_n", "{:>{fmt}}".format(self.d_n, fmt=fmt))
        table.add_row(
            "Neutral Axis Parameter- k_u", "{:>{fmt}}".format(self.k_u, fmt=fmt)
        )
        table.add_row("Axial Force", "{:>{fmt}}".format(self.n, fmt=fmt))
        table.add_row("Bending Capacity - mx", "{:>{fmt}}".format(self.mx, fmt=fmt))
        table.add_row("Bending Capacity - my", "{:>{fmt}}".format(self.my, fmt=fmt))
        table.add_row("Bending Capacity - mv", "{:>{fmt}}".format(self.mv, fmt=fmt))

        console = Console()
        console.print(table)


@dataclass
class MomentInteractionResults:
    """Class for storing moment interaction results."""

    n: List[float] = field(default_factory=list)
    m: List[float] = field(default_factory=list)

    def plot_diagram(
        self,
        n_scale: Optional[float] = 1e-3,
        m_scale: Optional[float] = 1e-6,
        **kwargs,
    ) -> matplotlib.axes._subplots.AxesSubplot:
        """Plots a moment interaction diagram.

        :param n_scale: Scaling factor to apply to axial force
        :type n_scale: Optional[float]
        :param n_scale: Scaling factor to apply to axial force
        :type m_scale: Optional[float]
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
        n_scale: Optional[float] = 1e-3,
        m_scale: Optional[float] = 1e-6,
        **kwargs,
    ) -> matplotlib.axes._subplots.AxesSubplot:
        """Plots multiple moment interaction diagrams.

        :param moment_interaction_results: List of moment interaction results objects
        :type moment_interaction_results:
            List[:class:`~concreteproperties.results.MomentInteractionResults`]
        :param labels: List of labels for each moment interaction diagram
        :type labels: List[str]
        :param float n_scale: Scaling factor to apply to axial force
        :type n_scale: Optional[float]
        :param float m_scale: Scaling factor to apply to bending moment
        :type m_scale: Optional[float]
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
        m_scale: Optional[float] = 1e-6,
        **kwargs,
    ) -> matplotlib.axes._subplots.AxesSubplot:
        """Plots a biaxial bending diagram.

        :param m_scale: Scaling factor to apply to bending moment
        :type m_scale: Optional[float]
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes._subplots.AxesSubplot`
        """

        # create plot and setup the plot
        with plotting_context(
            title=f"Biaxial Bending Diagram, $N = {self.n:.3e}$", **kwargs
        ) as (
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


@dataclass
class StressResult:
    """Class for storing stress results."""

    concrete_section: ConcreteSection
    concrete_analysis_sections: List[AnalysisSection]
    concrete_stresses: List[np.ndarray]
    concrete_forces: List[Tuple[float]]
    steel_geometries: List[Geometry]
    steel_stresses: List[float]
    steel_strains: List[float]
    steel_forces: List[Tuple[float]]

    def plot_stress(
        self,
        title: Optional[str] = "Stress",
        conc_cmap: Optional[str] = "RdGy",
        steel_cmap: Optional[str] = "bwr",
        **kwargs,
    ) -> matplotlib.axes._subplots.AxesSubplot:
        """Plots concrete and steel stresses on a concrete section.

        :param title: Plot title
        :type title: Optional[str]
        :param conc_cmap: Colour map for the concrete stress
        :type conc_cmap: Optional[str]
        :param steel_cmap: Colour map for the steel stress
        :type steel_cmap: Optional[str]
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes._subplots.AxesSubplot`
        """

        with plotting_context(
            title=title,
            **dict(
                kwargs, nrows=1, ncols=3, gridspec_kw={"width_ratios": [1, 0.08, 0.08]}
            ),
        ) as (fig, ax):
            # plot background
            self.concrete_section.plot_section(
                background=True, **dict(kwargs, ax=fig.axes[0])
            )

            # set up the colormaps
            cmap_conc = cm.get_cmap(name=conc_cmap)
            cmap_steel = cm.get_cmap(name=steel_cmap)

            # determine minimum and maximum stress values for the contour list
            # add tolerance for plotting stress blocks
            conc_sig_min = min([min(x) for x in self.concrete_stresses]) - 1e-12
            conc_sig_max = max([max(x) for x in self.concrete_stresses]) + 1e-12
            steel_sig_min = min(self.steel_stresses)
            steel_sig_max = max(self.steel_stresses)

            # set up ticks
            v_conc = np.linspace(conc_sig_min, conc_sig_max, 15, endpoint=True)
            v_steel = np.linspace(steel_sig_min, steel_sig_max, 15, endpoint=True)

            if np.isclose(v_conc[0], v_conc[-1], atol=1e-12):
                v_conc = 15
                ticks_conc = None
            else:
                ticks_conc = v_conc

            if np.isclose(v_steel[0], v_steel[-1], atol=1e-12):
                ticks_steel = None
                steel_tick_same = True
            else:
                ticks_steel = v_steel
                steel_tick_same = False

            # plot the concrete stresses
            for idx, sig in enumerate(self.concrete_stresses):
                # check region has a force
                if abs(self.concrete_forces[idx][0]) > 1e-8:
                    # create triangulation
                    triang = tri.Triangulation(
                        self.concrete_analysis_sections[idx].mesh_nodes[:, 0],
                        self.concrete_analysis_sections[idx].mesh_nodes[:, 1],
                        self.concrete_analysis_sections[idx].mesh_elements[:, 0:3],
                    )

                    # plot the filled contour
                    trictr = fig.axes[0].tricontourf(
                        triang, sig, v_conc, cmap=cmap_conc, norm=CenteredNorm()
                    )

                    # plot a zero stress contour, supressing warning
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="No contour levels were found within the data range.",
                        )

                        # set zero stress for neutral axis contour
                        zero_level = 0

                        if min(sig) > 0:
                            if min(sig) < 1e-3:
                                zero_level = min(sig) + 1e-12

                        if max(sig) < 0:
                            if max(sig) > -1e-3:
                                zero_level = max(sig) - 1e-12

                        if min(sig) == 0:
                            zero_level = 1e-12

                        if max(sig) == 0:
                            zero_level = -1e-12

                        CS = fig.axes[0].tricontour(
                            triang, sig, [zero_level], linewidths=1, linestyles="dashed"
                        )

            # plot the steel stresses
            steel_patches = []
            colours = []

            for idx, sig in enumerate(self.steel_stresses):
                steel_patches.append(
                    mpatches.Polygon(
                        xy=list(
                            self.concrete_section.steel_geometries[
                                idx
                            ].geom.exterior.coords
                        )
                    )
                )
                colours.append(sig)

            patch = PatchCollection(steel_patches, cmap=cmap_steel)
            patch.set_array(colours)
            if steel_tick_same:
                patch.set_clim([0.99 * v_steel[0], 1.01 * v_steel[-1]])
            fig.axes[0].add_collection(patch)

            # add the colour bars
            fig.colorbar(
                trictr,
                label="Concrete Stress",
                format="%.2e",
                ticks=ticks_conc,
                cax=fig.axes[1],
            )
            fig.colorbar(
                patch,
                label="Steel Stress",
                format="%.2e",
                ticks=ticks_steel,
                cax=fig.axes[2],
            )

            ax.set_aspect("equal", anchor="C")

        return ax
