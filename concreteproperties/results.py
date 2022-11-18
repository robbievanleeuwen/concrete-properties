from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import CenteredNorm  # type: ignore
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from rich.console import Console
from rich.table import Table
from scipy.interpolate import interp1d
from sectionproperties.pre.geometry import CompoundGeometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from concreteproperties.post import plotting_context

if TYPE_CHECKING:
    import matplotlib

    from concreteproperties.analysis_section import AnalysisSection
    from concreteproperties.concrete_section import ConcreteSection
    from concreteproperties.pre import CPGeom


@dataclass
class GrossProperties:
    """Class for storing gross concrete section properties.

    All properties with an `e_` preceding the property are multiplied by the elastic
    modulus. In order to obtain transformed properties, call the
    :meth:`~concreteproperties.concrete_section.ConcreteSection.get_transformed_gross_properties`
    method.
    """

    # section areas
    total_area: float = 0
    concrete_area: float = 0
    reinf_meshed_area: float = 0
    reinf_lumped_area: float = 0
    e_a: float = 0

    # section mass
    mass: float = 0

    # section perimeter
    perimeter: float = 0

    # first moments of area
    e_qx: float = 0
    e_qy: float = 0
    qx_gross: float = 0
    qy_gross: float = 0

    # centroids
    cx: float = 0
    cy: float = 0
    cx_gross: float = 0
    cy_gross: float = 0

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

    # other properties
    conc_ultimate_strain: float = 0

    def print_results(
        self,
        fmt: str = "8.6e",
    ):
        """Prints the gross concrete section properties to the terminal.

        :param fmt: Number format
        """

        table = Table(title="Gross Concrete Section Properties")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row("Total Area", "{:>{fmt}}".format(self.total_area, fmt=fmt))
        table.add_row("Concrete Area", "{:>{fmt}}".format(self.concrete_area, fmt=fmt))
        table.add_row(
            "Meshed Reinforcement Area",
            "{:>{fmt}}".format(self.reinf_meshed_area, fmt=fmt),
        )
        table.add_row(
            "Lumped Reinforcement Area",
            "{:>{fmt}}".format(self.reinf_lumped_area, fmt=fmt),
        )
        table.add_row("Axial Rigidity (EA)", "{:>{fmt}}".format(self.e_a, fmt=fmt))
        table.add_row("Mass (per unit length)", "{:>{fmt}}".format(self.mass, fmt=fmt))
        table.add_row("Perimeter", "{:>{fmt}}".format(self.perimeter, fmt=fmt))
        table.add_row("E.Qx", "{:>{fmt}}".format(self.e_qx, fmt=fmt))
        table.add_row("E.Qy", "{:>{fmt}}".format(self.e_qy, fmt=fmt))
        table.add_row("x-Centroid", "{:>{fmt}}".format(self.cx, fmt=fmt))
        table.add_row("y-Centroid", "{:>{fmt}}".format(self.cy, fmt=fmt))
        table.add_row("x-Centroid (Gross)", "{:>{fmt}}".format(self.cx_gross, fmt=fmt))
        table.add_row("y-Centroid (Gross)", "{:>{fmt}}".format(self.cy_gross, fmt=fmt))
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
        table.add_row(
            "Ultimate Concrete Strain",
            "{:>{fmt}}".format(self.conc_ultimate_strain, fmt=fmt),
        )

        console = Console()
        console.print(table)


@dataclass
class TransformedGrossProperties:
    """Class for storing transformed gross concrete section properties.

    :param concrete_properties: Concrete properties object
    :param elastic_modulus: Reference elastic modulus
    """

    concrete_properties: GrossProperties = field(repr=False)
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
        self.area = self.concrete_properties.e_a / self.elastic_modulus
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
        fmt: str = "8.6e",
    ):
        """Prints the transformed gross concrete section properties to the terminal.

        :param fmt: Number format
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
    r"""Class for storing cracked concrete section properties.

    All properties with an `e_` preceding the property are multiplied by the elastic
    modulus. In order to obtain transformed properties, call the
    :meth:`~concreteproperties.results.CrackedResults.calculate_transformed_properties`
    method.

    :param theta: Angle (in radians) the neutral axis makes with the horizontal
        axis (:math:`-\pi \leq \theta \leq \pi`)
    """

    theta: float
    m_cr: float = 0
    d_nc: float = 0
    cracked_geometries: List[CPGeom] = field(default_factory=list, repr=False)
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
    e_i11_cr: float = 0
    e_i22_cr: float = 0
    phi_cr: float = 0

    # transformed properties
    elastic_modulus_ref: Optional[float] = None
    a_cr: Optional[float] = None
    qx_cr: Optional[float] = None
    qy_cr: Optional[float] = None
    ixx_g_cr: Optional[float] = None
    iyy_g_cr: Optional[float] = None
    ixy_g_cr: Optional[float] = None
    ixx_c_cr: Optional[float] = None
    iyy_c_cr: Optional[float] = None
    ixy_c_cr: Optional[float] = None
    iuu_cr: Optional[float] = None
    i11_cr: Optional[float] = None
    i22_cr: Optional[float] = None

    def calculate_transformed_properties(
        self,
        elastic_modulus: float,
    ):
        """Calculates and stores transformed cracked properties using a reference
        elastic modulus.

        :param elastic_modulus: Reference elastic modulus
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
        self.i11_cr = self.e_i11_cr / elastic_modulus
        self.i22_cr = self.e_i22_cr / elastic_modulus

    def plot_cracked_geometries(
        self,
        title: str = "Cracked Geometries",
        **kwargs,
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots the geometries that remain (are in compression or are reinforcement)
        after a cracked analysis.

        :param title: Plot title
        :param kwargs: Passed to
            :meth:`~sectionproperties.pre.geometry.CompoundGeometry.plot_geometry`

        :return: Matplotlib axes object
        """

        return CompoundGeometry(
            [geom.to_sp_geom() for geom in self.cracked_geometries]
        ).plot_geometry(title=title, **kwargs)

    def print_results(
        self,
        fmt: str = "8.6e",
    ):
        """Prints the cracked concrete section properties to the terminal.

        :param fmt: Number format
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
            table.add_row("I11_cr", "{:>{fmt}}".format(self.i11_cr, fmt=fmt))
            table.add_row("I22_cr", "{:>{fmt}}".format(self.i22_cr, fmt=fmt))

        table.add_row("E.Ixx_g_cr", "{:>{fmt}}".format(self.e_ixx_g_cr, fmt=fmt))
        table.add_row("E.Iyy_g_cr", "{:>{fmt}}".format(self.e_iyy_g_cr, fmt=fmt))
        table.add_row("E.Ixy_g_cr", "{:>{fmt}}".format(self.e_ixy_g_cr, fmt=fmt))
        table.add_row("E.Ixx_c_cr", "{:>{fmt}}".format(self.e_ixx_c_cr, fmt=fmt))
        table.add_row("E.Iyy_c_cr", "{:>{fmt}}".format(self.e_iyy_c_cr, fmt=fmt))
        table.add_row("E.Ixy_c_cr", "{:>{fmt}}".format(self.e_ixy_c_cr, fmt=fmt))
        table.add_row("E.Iuu_cr", "{:>{fmt}}".format(self.e_iuu_cr, fmt=fmt))
        table.add_row("E.I11_cr", "{:>{fmt}}".format(self.e_i11_cr, fmt=fmt))
        table.add_row("E.I22_cr", "{:>{fmt}}".format(self.e_i22_cr, fmt=fmt))
        table.add_row("phi_cr", "{:>{fmt}}".format(self.phi_cr, fmt=fmt))

        console = Console()
        console.print(table)


@dataclass
class MomentCurvatureResults:
    r"""Class for storing moment curvature results.

    :param theta: Angle (in radians) the neutral axis makes with the horizontal
        axis (:math:`-\pi \leq \theta \leq \pi`)
    :param kappa: List of curvatures
    :param n: List of axial forces
    :param m_x: List of bending moments about the x-axis
    :param m_y: List of bending moments about the y-axis
    :param m_xy: List of resultant bending moments
    :param failure_geometry: Geometry object of the region of the cross-section that
        failed, ending the moment curvature analysis
    :param convergence: The critical ratio between the strain and the failure strain
        within the cross-section for each curvature step in the analysis. A value of one
        indicates failure.
    """

    # results
    theta: float
    kappa: List[float] = field(default_factory=list)
    n: List[float] = field(default_factory=list)
    m_x: List[float] = field(default_factory=list)
    m_y: List[float] = field(default_factory=list)
    m_xy: List[float] = field(default_factory=list)
    failure_geometry: CPGeom = field(init=False)
    convergence: List[float] = field(default_factory=list)

    # for analysis
    _kappa: float = field(default=0, repr=False)
    _n_i: float = field(default=0, repr=False)
    _m_x_i: float = field(default=0, repr=False)
    _m_y_i: float = field(default=0, repr=False)
    _failure: bool = field(default=False, repr=False)
    _failure_convergence: float = field(default=0, repr=False)

    def plot_results(
        self,
        m_scale: float = 1e-6,
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots the moment curvature results.

        :param m_scale: Scaling factor to apply to bending moment
        :param fmt: Plot format string
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        """

        # scale moments
        moments = np.array(self.m_xy) * m_scale

        # create plot and setup the plot
        with plotting_context(title="Moment-Curvature", **kwargs) as (
            fig,
            ax,
        ):
            ax.plot(self.kappa, moments, fmt)  # type: ignore
            plt.xlabel("Curvature")
            plt.ylabel("Moment")
            plt.grid(True)

        return ax

    @staticmethod
    def plot_multiple_results(
        moment_curvature_results: List[MomentCurvatureResults],
        labels: List[str],
        m_scale: float = 1e-6,
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots multiple moment curvature results.

        :param moment_curvature_results: List of moment curvature results objects
        :param labels: List of labels for each moment curvature diagram
        :param m_scale: Scaling factor to apply to bending moment
        :param fmt: Plot format string
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        """

        # create plot and setup the plot
        with plotting_context(title="Moment-Curvature", **kwargs) as (
            fig,
            ax,
        ):
            idx = 0

            # for each M-k curve
            for idx, mk_result in enumerate(moment_curvature_results):
                # scale results
                kappas = np.array(mk_result.kappa)
                moments = np.array(mk_result.m_xy) * m_scale

                ax.plot(kappas, moments, fmt, label=labels[idx])  # type: ignore

            plt.xlabel("Curvature")
            plt.ylabel("Moment")
            plt.grid(True)

            # if there is more than one curve show legend
            if idx > 0:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # type: ignore

        return ax

    def plot_failure_geometry(
        self,
        title: str = "Failure Geometry",
        **kwargs,
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots the geometry that fails in the moment curvature analysis.

        :param title: Plot title
        :param kwargs: Passed to
            :meth:`~sectionproperties.pre.geometry.CompoundGeometry.plot_geometry`

        :return: Matplotlib axes object
        """

        return self.failure_geometry.plot_geometry(title=title, **kwargs)

    def get_curvature(
        self,
        moment: float,
    ) -> float:
        """Given a moment, uses the moment-curvature results to interpolate a curvature.

        :param moment: Bending moment at which to obtain curvature

        :raises ValueError: If supplied moment is outside bounds of moment-curvature
            results.

        :return: Curvature
        """

        # check moment is within bounds of results
        m_min = min(self.m_xy)
        m_max = max(self.m_xy)

        if moment > m_max or moment < m_min:
            raise ValueError(
                "moment must be within the bounds of the moment-curvature results."
            )

        f_kappa = interp1d(
            x=self.m_xy,
            y=self.kappa,
            kind="linear",
        )

        return float(f_kappa(moment))


@dataclass(order=True)
class UltimateBendingResults:
    r"""Class for storing ultimate bending results.

    :param theta: Angle (in radians) the neutral axis makes with the horizontal
        axis (:math:`-\pi \leq \theta \leq \pi`)
    :param d_n: Ultimate neutral axis depth
    :param k_u: Neutral axis parameter *(d_n / d)*
    :param n: Resultant axial force
    :param m_x: Resultant bending moment about the x-axis
    :param m_y: Resultant bending moment about the y-axis
    :param m_xy: Resultant bending moment
    :param label: Result label
    """

    # bending angle
    theta: float

    # ultimate neutral axis depth
    d_n: float = 0
    k_u: float = 0

    # resultant actions
    n: float = 0
    m_x: float = 0
    m_y: float = 0
    m_xy: float = 0

    # label
    label: Optional[str] = field(default=None, compare=False)

    def print_results(
        self,
        fmt: str = "8.6e",
    ):
        """Prints the ultimate bending results to the terminal.

        :param fmt: Number format
        """

        table = Table(title="Ultimate Bending Results")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        if self.label:
            table.add_row("Label", self.label)

        table.add_row("Bending Angle - theta", "{:>{fmt}}".format(self.theta, fmt=fmt))
        table.add_row("Neutral Axis Depth - d_n", "{:>{fmt}}".format(self.d_n, fmt=fmt))
        table.add_row(
            "Neutral Axis Parameter - k_u", "{:>{fmt}}".format(self.k_u, fmt=fmt)
        )
        table.add_row("Axial Force", "{:>{fmt}}".format(self.n, fmt=fmt))
        table.add_row("Bending Capacity - m_x", "{:>{fmt}}".format(self.m_x, fmt=fmt))
        table.add_row("Bending Capacity - m_y", "{:>{fmt}}".format(self.m_y, fmt=fmt))
        table.add_row("Bending Capacity - m_xy", "{:>{fmt}}".format(self.m_xy, fmt=fmt))

        console = Console()
        console.print(table)


@dataclass
class MomentInteractionResults:
    """Class for storing moment interaction results.

    :param results: List of ultimate bending result objects
    """

    results: List[UltimateBendingResults] = field(default_factory=list)

    def sort_results(self) -> None:
        """Sorts the results by decreasing axial force."""

        self.results.sort(reverse=True)

        # remove duplicates from sorted list
        new_results = []

        for res in self.results:
            if res not in new_results:
                new_results.append(res)

        self.results = new_results

    def get_results_lists(
        self,
        moment: str,
    ) -> Tuple[List[float], List[float]]:
        """Returns a list of axial forces and moments.

        :param moment: Which moment to plot, acceptable values are ``"m_x"``, ``"m_y"``
            or ``"m_xy"``
        :return: List of axial forces and moments *(n, m)*
        """

        # build list of results
        n_list = []
        m_list = []

        for result in self.results:
            n_list.append(result.n)

            if moment == "m_x":
                m_list.append(result.m_x)
            elif moment == "m_y":
                m_list.append(result.m_y)
            elif moment == "m_xy":
                m_list.append(result.m_xy)
            else:
                raise ValueError(f"{moment} not an acceptable value for moment.")

        return n_list, m_list

    def plot_diagram(
        self,
        n_scale: float = 1e-3,
        m_scale: float = 1e-6,
        moment: str = "m_x",
        fmt: str = "o-",
        labels: bool = False,
        label_offset: bool = False,
        **kwargs,
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots a moment interaction diagram.

        :param n_scale: Scaling factor to apply to axial force
        :param m_scale: Scaling factor to apply to the bending moment
        :param moment: Which moment to plot, acceptable values are ``"m_x"``, ``"m_y"``
            or ``"m_xy"``
        :param fmt: Plot format string
        :param labels: If set to True, also plots labels on the diagram
        :param label_offset: If set to True, attempts to offset the label from the
            diagram
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        """

        # create plot and setup the plot
        with plotting_context(title="Moment Interaction Diagram", **kwargs) as (
            fig,
            ax,
        ):
            # get results
            n_list, m_list = self.get_results_lists(moment=moment)

            # scale results
            forces = np.array(n_list) * n_scale
            moments = np.array(m_list) * m_scale

            # plot diagram
            ax.plot(moments, forces, fmt)  # type: ignore

            # plot labels
            if labels:
                if label_offset:
                    # compute gradients of curve and aspect ratio of plot
                    grad = np.gradient([moments, forces], axis=1)
                    x_diff = ax.get_xlim()  # type: ignore
                    y_diff = ax.get_ylim()  # type: ignore
                    ar = (y_diff[1] - y_diff[0]) / (x_diff[1] - x_diff[0])

                for idx, m in enumerate(m_list):
                    if self.results[idx].label:
                        # get x,y position on plot
                        x = m * m_scale
                        y = n_list[idx] * n_scale

                        if label_offset:
                            # calculate text offset
                            grad_pt = grad[1, idx] / grad[0, idx] / ar  # type: ignore
                            if grad_pt == 0:
                                norm_angle = np.pi / 2
                            else:
                                norm_angle = np.arctan2(-1 / grad_pt, 1)
                            x_t = np.cos(norm_angle) * 20
                            y_t = np.sin(norm_angle) * 20
                            annotate_dict = {
                                "xytext": (x_t, y_t),
                                "textcoords": "offset points",
                                "arrowprops": dict(
                                    arrowstyle="->",
                                    connectionstyle="angle,angleA=0,angleB=90,rad=10",
                                ),
                                "bbox": dict(boxstyle="round", fc="0.8"),
                            }
                        else:
                            annotate_dict = {}

                        # plot text
                        ax.annotate(  # type: ignore
                            text=self.results[idx].label, xy=(x, y), **annotate_dict
                        )

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
        moment: str = "m_x",
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots multiple moment interaction diagrams.

        :param moment_interaction_results: List of moment interaction results objects
        :param labels: List of labels for each moment interaction diagram
        :param n_scale: Scaling factor to apply to axial force
        :param m_scale: Scaling factor to apply to bending moment
        :param moment: Which moment to plot, acceptable values are ``"m_x"``, ``"m_y"``
            or ``"m_xy"``
        :param fmt: Plot format string
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        """

        # create plot and setup the plot
        with plotting_context(title="Moment Interaction Diagram", **kwargs) as (
            fig,
            ax,
        ):
            idx = 0

            # for each M-N curve
            for idx, mi_result in enumerate(moment_interaction_results):
                n_list, m_list = mi_result.get_results_lists(moment=moment)

                # scale results
                forces = np.array(n_list) * n_scale
                moments = np.array(m_list) * m_scale

                ax.plot(moments, forces, fmt, label=labels[idx])  # type: ignore

            plt.xlabel("Bending Moment")
            plt.ylabel("Axial Force")
            plt.grid(True)

            # if there is more than one curve show legend
            if idx > 0:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # type: ignore

        return ax

    def point_in_diagram(
        self,
        n: float,
        m: float,
        moment: str = "m_x",
    ) -> bool:
        """Determines whether or not the combination of axial force and moment lies
        within the moment interaction diagram.

        :param n: Axial force
        :param m: Bending moment
        :param moment: Which moment to analyse, acceptable values are ``"m_x"``,
            ``"m_y"`` or ``"m_xy"``

        :returns: True, if combination of axial force and moment is within the diagram
        """

        # get results
        n_list, m_list = self.get_results_lists(moment=moment)

        # create a polygon from points on diagram
        poly_points = []

        for idx, mom in enumerate(m_list):
            poly_points.append((mom, n_list[idx]))

        poly = Polygon(poly_points)
        point = Point(m, n)

        return poly.contains(point)


@dataclass
class BiaxialBendingResults:
    """Class for storing biaxial bending results.

    :param n: Net axial force
    :param results: List of ultimate bending result objects
    """

    n: float
    results: List[UltimateBendingResults] = field(default_factory=list)

    def get_results_lists(
        self,
    ) -> Tuple[List[float], List[float]]:
        """Returns a list and moments about the ``x`` and ``y`` axes.

        :return: List of axial forces and moments *(mx, my)*
        """

        # build list of results
        m_x_list = []
        m_y_list = []

        for result in self.results:
            m_x_list.append(result.m_x)
            m_y_list.append(result.m_y)

        return m_x_list, m_y_list

    def plot_diagram(
        self,
        m_scale: float = 1e-6,
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots a biaxial bending diagram.

        :param m_scale: Scaling factor to apply to bending moment
        :param fmt: Plot format string
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        """

        m_x_list, m_y_list = self.get_results_lists()

        # create plot and setup the plot
        with plotting_context(
            title=f"Biaxial Bending Diagram, $N = {self.n:.3e}$", **kwargs
        ) as (
            fig,
            ax,
        ):
            # scale results
            m_x = np.array(m_x_list) * m_scale
            m_y = np.array(m_y_list) * m_scale

            ax.plot(m_x, m_y, fmt)  # type: ignore

            plt.xlabel("Bending Moment $M_x$")
            plt.ylabel("Bending Moment $M_y$")
            plt.grid(True)

        return ax

    @staticmethod
    def plot_multiple_diagrams_2d(
        biaxial_bending_results: List[BiaxialBendingResults],
        labels: Optional[List[str]] = None,
        m_scale: float = 1e-6,
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots multiple biaxial bending diagrams in a 2D plot.

        :param biaxial_bending_results: List of biaxial bending results objects
        :param labels: List of labels for each biaxial bending diagram, if not provided
            labels are axial forces
        :param m_scale: Scaling factor to apply to bending moment
        :param fmt: Plot format string
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        """

        # create plot and setup the plot
        with plotting_context(title="Biaxial Bending Diagram", **kwargs) as (
            fig,
            ax,
        ):
            idx = 0

            # generate default labels
            if labels is None:
                labels = []
                default_labels = True
            else:
                default_labels = False

            # for each M-N curve
            for idx, bb_result in enumerate(biaxial_bending_results):
                m_x_list, m_y_list = bb_result.get_results_lists()

                # scale results
                m_x_list = np.array(m_x_list) * m_scale
                m_y_list = np.array(m_y_list) * m_scale

                # generate default labels
                if default_labels:
                    labels.append(f"N = {bb_result.n:.3e}")

                ax.plot(m_x_list, m_y_list, fmt, label=labels[idx])  # type: ignore

            plt.xlabel("Bending Moment $M_x$")
            plt.ylabel("Bending Moment $M_y$")
            plt.grid(True)

            # if there is more than one curve show legend
            if idx > 0:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # type: ignore

        return ax

    @staticmethod
    def plot_multiple_diagrams_3d(
        biaxial_bending_results: List[BiaxialBendingResults],
        n_scale: float = 1e-3,
        m_scale: float = 1e-6,
        fmt: str = "-",
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots multiple biaxial bending diagrams in a 3D plot.

        :param biaxial_bending_results: List of biaxial bending results objects
        :param n_scale: Scaling factor to apply to axial force
        :param m_scale: Scaling factor to apply to bending moment
        :param fmt: Plot format string

        :return: Matplotlib axes object
        """

        # make 3d plot
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        # for each curve
        for bb_result in biaxial_bending_results:
            m_x_list, m_y_list = bb_result.get_results_lists()

            # scale results
            n_list = bb_result.n * n_scale * np.ones(len(m_x_list))
            m_x_list = np.array(m_x_list) * m_scale
            m_y_list = np.array(m_y_list) * m_scale

            ax.plot3D(m_x_list, m_y_list, n_list, fmt)  # type: ignore

        ax.set_xlabel("Bending Moment $M_x$")
        ax.set_ylabel("Bending Moment $M_y$")
        ax.set_zlabel("Axial Force $N$")  # type: ignore
        plt.show()

        return ax

    def point_in_diagram(
        self,
        m_x: float,
        m_y: float,
    ) -> bool:
        """Determines whether or not the combination of bending moments lies within the
        biaxial bending diagram.

        :param m_x: Bending moment about the x-axis
        :param m_y: Bending moment about the y-axis

        :returns: True, if combination of bendings moments is within the diagram
        """

        # create a polygon from points on diagram
        poly_points = []

        for ult_res in self.results:
            poly_points.append((ult_res.m_x, ult_res.m_y))

        poly = Polygon(poly_points)
        point = Point(m_x, m_y)

        return poly.contains(point)


@dataclass
class StressResult:
    """Class for storing stress results.

    The lever arm is computed to the elastic centroid.

    :param concrete_analysis_sections: List of concrete analysis section objects
        present in the stress analysis, which can be visualised by calling the
        :meth:`~concreteproperties.analysis_section.AnalysisSection.plot_mesh` or
        :meth:`~concreteproperties.analysis_section.AnalysisSection.plot_shape`
    :param concrete_stresses: List of concrete stresses at the nodes of each concrete
        analysis section
    :param concrete_forces: List of net forces for each concrete analysis section and
        its lever arm (``force``, ``d_x``, ``d_y``)
    :param meshed_reinforcement_sections: List of meshed reinforcement section objects
        present in the stress analysis
    :param meshed_reinforcement_stresses: List of meshed reinforcement stresses at the
        nodes of each meshed reinforcement analysis section
    :param meshed_reinforcement_forces: List of net forces for each meshed reinforcement
         analysis section and its lever arm (``force``, ``d_x``, ``d_y``)
    :param lumped_reinforcement_geometries: List of lumped reinforcement geometry
        objects present in the stress analysis
    :param lumped_reinforcement_stresses: List of lumped reinforcement stresses for
        each lumped geometry
    :param lumped_reinforcement_strains: List of lumped reinforcement strains for each
        lumped geometry
    :param lumped_reinforcement_forces: List of net forces for each lumped reinforcement
         geometry and its lever arm (``force``, ``d_x``, ``d_y``)
    """

    concrete_section: ConcreteSection
    concrete_analysis_sections: List[AnalysisSection]
    concrete_stresses: List[np.ndarray]
    concrete_forces: List[Tuple[float, float, float]]
    meshed_reinforcement_sections: List[AnalysisSection]
    meshed_reinforcement_stresses: List[np.ndarray]
    meshed_reinforcement_forces: List[Tuple[float, float, float]]
    lumped_reinforcement_geometries: List[CPGeom]
    lumped_reinforcement_stresses: List[float]
    lumped_reinforcement_strains: List[float]
    lumped_reinforcement_forces: List[Tuple[float, float, float]]

    def plot_stress(
        self,
        title: str = "Stress",
        conc_cmap: str = "RdGy",
        reinf_cmap: str = "bwr",
        **kwargs,
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots concrete and steel stresses on a concrete section.

        :param title: Plot title
        :param conc_cmap: Colour map for the concrete stress
        :param reinf_cmap: Colour map for the reinforcement stress
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        """

        with plotting_context(
            title=title,
            aspect=True,
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
            cmap_reinf = cm.get_cmap(name=reinf_cmap)

            # determine minimum and maximum stress values for the contour list
            # add tolerance for plotting stress blocks
            conc_sig_min = min([min(x) for x in self.concrete_stresses]) - 1e-12
            conc_sig_max = max([max(x) for x in self.concrete_stresses]) + 1e-12

            # if there is meshed reinforcement, calculate min and max
            if self.meshed_reinforcement_stresses:
                meshed_reinf_sig_min = (
                    min([min(x) for x in self.meshed_reinforcement_stresses]) - 1e-12
                )
                meshed_reinf_sig_max = (
                    max([max(x) for x in self.meshed_reinforcement_stresses]) + 1e-12
                )
            else:
                meshed_reinf_sig_min = None
                meshed_reinf_sig_max = None

            # if there is lumped reinforcement, calculate min and max
            if self.lumped_reinforcement_stresses:
                lumped_reinf_sig_min = min(self.lumped_reinforcement_stresses)
                lumped_reinf_sig_max = max(self.lumped_reinforcement_stresses)
            else:
                lumped_reinf_sig_min = None
                lumped_reinf_sig_max = None

            # determine min and max reinforcement stresess
            if (
                meshed_reinf_sig_min
                and meshed_reinf_sig_max
                and lumped_reinf_sig_min
                and lumped_reinf_sig_max
            ):
                reinf_sig_min = min(meshed_reinf_sig_min, lumped_reinf_sig_min)
                reinf_sig_max = max(meshed_reinf_sig_max, lumped_reinf_sig_max)
            elif meshed_reinf_sig_min and meshed_reinf_sig_max:
                reinf_sig_min = meshed_reinf_sig_min
                reinf_sig_max = meshed_reinf_sig_max
            elif lumped_reinf_sig_min and lumped_reinf_sig_max:
                reinf_sig_min = lumped_reinf_sig_min
                reinf_sig_max = lumped_reinf_sig_max
            else:
                reinf_sig_min = 0
                reinf_sig_max = 0

            # set up ticks
            v_conc = np.linspace(conc_sig_min, conc_sig_max, 15, endpoint=True)
            v_reinf = np.linspace(reinf_sig_min, reinf_sig_max, 15, endpoint=True)

            if np.isclose(v_conc[0], v_conc[-1], atol=1e-12):
                v_conc = 15
                ticks_conc = None
            else:
                ticks_conc = v_conc

            if np.isclose(v_reinf[0], v_reinf[-1], atol=1e-12):
                ticks_reinf = None
                reinf_tick_same = True
            else:
                ticks_reinf = v_reinf
                reinf_tick_same = False

            # plot the concrete stresses
            for idx, sig in enumerate(self.concrete_stresses):
                # check region has a force
                if abs(self.concrete_forces[idx][0]) > 1e-8:
                    # create triangulation
                    triang_conc = tri.Triangulation(
                        self.concrete_analysis_sections[idx].mesh_nodes[:, 0],
                        self.concrete_analysis_sections[idx].mesh_nodes[:, 1],
                        self.concrete_analysis_sections[idx].mesh_elements[:, 0:3],  # type: ignore
                    )

                    # plot the filled contour
                    trictr_conc = fig.axes[0].tricontourf(
                        triang_conc, sig, v_conc, cmap=cmap_conc, norm=CenteredNorm()
                    )  # type: ignore

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
                            triang_conc,
                            sig,
                            [zero_level],
                            linewidths=1,
                            linestyles="dashed",
                        )  # type: ignore

            # plot the meshed reinforcement stresses
            trictr_reinf = None

            for idx, sig in enumerate(self.meshed_reinforcement_stresses):
                # check region has a force
                if abs(self.meshed_reinforcement_forces[idx][0]) > 1e-8:
                    # create triangulation
                    triang_reinf = tri.Triangulation(
                        self.meshed_reinforcement_sections[idx].mesh_nodes[:, 0],
                        self.meshed_reinforcement_sections[idx].mesh_nodes[:, 1],
                        self.meshed_reinforcement_sections[idx].mesh_elements[:, 0:3],  # type: ignore
                    )

                    # plot the filled contour
                    trictr_reinf = fig.axes[0].tricontourf(
                        triang_reinf, sig, v_reinf, cmap=cmap_reinf, norm=CenteredNorm()
                    )  # type: ignore

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
                            triang_reinf,
                            sig,
                            [zero_level],
                            linewidths=1,
                            linestyles="dashed",
                        )  # type: ignore

            # plot the lumped reinforcement stresses
            lumped_reinf_patches = []
            colours = []

            for idx, sig in enumerate(self.lumped_reinforcement_stresses):
                lumped_reinf_patches.append(
                    mpatches.Polygon(
                        xy=list(self.lumped_reinforcement_geometries[idx].geom.exterior.coords)  # type: ignore
                    )
                )
                colours.append(sig)

            patch = PatchCollection(lumped_reinf_patches, cmap=cmap_reinf)
            patch.set_array(colours)
            if reinf_tick_same:
                patch.set_clim(vmin=0.99 * v_reinf[0], vmax=1.01 * v_reinf[-1])
            else:
                patch.set_clim(vmin=v_reinf[0], vmax=v_reinf[-1])
            fig.axes[0].add_collection(patch)  # type: ignore

            # add the colour bars
            fig.colorbar(
                trictr_conc,  # type: ignore
                label="Concrete Stress",
                format="%.2e",
                ticks=ticks_conc,
                cax=fig.axes[1],
            )

            if trictr_reinf:
                mappable = trictr_reinf
            else:
                mappable = patch

            fig.colorbar(
                mappable,
                label="Reinforcement Stress",
                format="%.2e",
                ticks=ticks_reinf,
                cax=fig.axes[2],
            )

        return ax

    def sum_forces(
        self,
    ) -> float:
        """Returns the sum of the internal forces.

        :return: Sum of internal forces
        """

        force_sum = 0

        # sum concrete forces
        for conc_force in self.concrete_forces:
            force_sum += conc_force[0]

        # sum meshed reinf stresses
        for meshed_reinf_force in self.meshed_reinforcement_forces:
            force_sum += meshed_reinf_force[0]

        # sum lumped reinf forces
        for lumped_reinf_force in self.lumped_reinforcement_forces:
            force_sum += lumped_reinf_force[0]

        return force_sum

    def sum_moments(
        self,
    ) -> Tuple[float, float, float]:
        """Returns the sum of the internal moments.

        :return: Sum of internal moments about each axis and resultant moment
            (``m_x``, ``m_y``, ``m``)
        """

        moment_sum_x = 0
        moment_sum_y = 0

        # sum concrete forces
        for conc_force in self.concrete_forces:
            moment_sum_x += conc_force[0] * conc_force[2]
            moment_sum_y += conc_force[0] * conc_force[1]

        # sum meshed reinf stresses
        for meshed_reinf_force in self.meshed_reinforcement_forces:
            moment_sum_x += meshed_reinf_force[0] * meshed_reinf_force[2]
            moment_sum_y += meshed_reinf_force[0] * meshed_reinf_force[1]

        # sum lumped reinf forces
        for lumped_reinf_force in self.lumped_reinforcement_forces:
            moment_sum_x += lumped_reinf_force[0] * lumped_reinf_force[2]
            moment_sum_y += lumped_reinf_force[0] * lumped_reinf_force[1]

        moment_sum = np.sqrt(moment_sum_x * moment_sum_x + moment_sum_y * moment_sum_y)

        return moment_sum_x, moment_sum_y, moment_sum
